import torch
import cv2
import numpy as np
from pymavlink import mavutil
import time


YOLO_MODEL = 'C:/Users/nayak/Documents/Bullseye_Detection/yolov5/runs/train/bullseye_detector8/weights/best.pt'
TARGET_CLASS = 'hotspot-y5zB'   
CAMERA_SOURCE = 0
MAVLINK_CONNECTION = 'tcp:127.0.0.1:5762'
SPEED = 3                      
KP_YAW = 1.0                   
KP_ALT = 0.7                   
KP_VX = 0.5                    
TARGET_ALTITUDE = 15           
CENTER_THRESHOLD_X = 0.05      
MAVLINK_SIGN_KEY = b'supersecurekey1234'
SIGNING_ID = 1


print("Loading YOLOv5 model")
model = torch.hub.load('ultralytics/yolov5', 'custom', YOLO_MODEL)
model.conf = 0.4


print("Connecting to MAVLink")
boot_time = time.time()
master = mavutil.mavlink_connection(
    MAVLINK_CONNECTION, baud=115200, source_system=1, mavlink20=True
)
master.wait_heartbeat()
print(f"Connected to system (sys {master.target_system}, comp {master.target_component})")

master.signing = True
master.signing_secret_key = MAVLINK_SIGN_KEY
master.signing_link_id = SIGNING_ID
master.signing_timestamp = int(time.time())


def arm_and_takeoff():
    print("Setting GUIDED mode...")
    mode_guided = master.mode_mapping()['GUIDED']
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE,
        0,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_guided, 0, 0, 0, 0, 0
    )
    master.recv_match(type='COMMAND_ACK', blocking=True)

    print("Arm")
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 0, 0, 0, 0, 0, 0
    )
    master.recv_match(type='COMMAND_ACK', blocking=True)
    time.sleep(2)

    print(f" Taking off to {TARGET_ALTITUDE}m...")
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, TARGET_ALTITUDE
    )
    master.recv_match(type='COMMAND_ACK', blocking=True)
   

arm_and_takeoff()
time.sleep(10)


def send_velocity(vx, vy, vz, yaw_rate):
    elapsed = (time.time() - boot_time) * 1e3
    master.mav.set_position_target_local_ned_send(
        int(elapsed),
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        0b0000111111000111,
        0, 0, 0,
        vx, vy, vz,
        0, 0, 0,
        0, yaw_rate
    )
    print(f"vx={vx:.2f}, vy=0, vz={vz:.2f}, yaw_rate={yaw_rate:.2f}")


print(" Opening camera")
cap = cv2.VideoCapture(CAMERA_SOURCE)
if not cap.isOpened():
    raise IOError("Cannot open camera/stream")


try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame received")
            send_velocity(0, 0, 0, 0)
            continue

        results = model(frame)
        detections = results.pandas().xyxy[0]
        print("Detected classes:", detections['name'].unique())

        target = detections[detections['name'] == TARGET_CLASS]
        if not target.empty:
            target = target.assign(area=(target['xmax'] - target['xmin']) * (target['ymax'] - target['ymin']))
            target = target.iloc[target['area'].idxmax()]
            x_center = (target['xmin'] + target['xmax']) / 2
            y_center = (target['ymin'] + target['ymax']) / 2
            bbox_area = target['area']

            frame_h, frame_w, _ = frame.shape
            cx, cy = frame_w / 2, frame_h / 2

            offset_x = (x_center - cx) / cx
            offset_y = (y_center - cy) / cy

            vz = offset_y * KP_ALT

            if abs(offset_y) >= CENTER_THRESHOLD_X:
                vx = offset_y * KP_VX * SPEED
            else:
                vx = 0

            if abs(offset_x) >= CENTER_THRESHOLD_X:
                vy = offset_x * KP_VX * SPEED
            else:
                vy = 0

            print(f"Moving: vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}")
            send_velocity(vx, vy, vz, 0)

            
            label = f"{target['name']} {target['confidence']:.2f}"
            cv2.rectangle(frame, (int(target['xmin']), int(target['ymin'])),
                          (int(target['xmax']), int(target['ymax'])), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(target['xmin']), int(target['ymin']) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            print("Target not found - hovering")
            send_velocity(0, 0, 0, 0)

        cv2.imshow('YOLOv5 Detection', np.squeeze(results.render()))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n[Interrupted by user")

finally:
    print("Landing...")
    send_velocity(0, 0, 0, 0)
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
        0, 0, 0, 0, 0, 0, 0, 0
    )
    cap.release()
    cv2.destroyAllWindows()
