import torch
import cv2
import numpy as np
from pymavlink import mavutil
import time

YOLO_MODEL = 'C:/Users/nayak/Documents/Bullseye_Detection/yolov5/runs/train/bullseye_detector8/weights/best.pt'
TARGET_CLASS = 'person'
CAMERA_SOURCE = 0
MAVLINK_CONNECTION = 'tcp:127.0.0.1:5762'  
SPEED = 3                     
KP_YAW = 1.5                   
KP_ALT = 1                  
TARGET_ALTITUDE = 15           
MAVLINK_SIGN_KEY = b'supersecurekey1234'  
SIGNING_ID = 1                            

print("[INFO] Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'custom', YOLO_MODEL)
model.conf = 0.4

print("[INFO] Connecting to MAVLink (MAVLink2)...")
boot_time = time.time()
master = mavutil.mavlink_connection(
    MAVLINK_CONNECTION, baud=115200, source_system=1, mavlink20=True
)
master.wait_heartbeat()
print(f"[INFO] Connected to system (sys {master.target_system}, comp {master.target_component})")

print("[INFO] Enabling MAVLink2 message signing...")
master.signing = True
master.signing_secret_key = MAVLINK_SIGN_KEY
master.signing_link_id = SIGNING_ID
master.signing_timestamp = int(time.time())
print(f"[INFO] Message signing enabled with link_id={SIGNING_ID}")

def send_fake_position_estimate():
    print("[INFO] Sending fake position estimates for EKF init...")
    for _ in range(50):  # 5 seconds
        master.mav.vision_position_estimate_send(
            int((time.time() - boot_time) * 1e6),  
                                    0, 0, -1,               
            0, 0, 0                  
        )
        time.sleep(0.1)

send_fake_position_estimate()


print("[INFO] Setting GUIDED mode...")
mode_guided = master.mode_mapping()['GUIDED']
master.mav.command_long_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_CMD_DO_SET_MODE,
    0,
    mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
    mode_guided, 0, 0, 0, 0, 0
)
ack = master.recv_match(type='COMMAND_ACK', blocking=True, timeout=5)
print(f"[INFO] GUIDED mode ACK: {ack}")

time.sleep(2)

print("[INFO] Arming drone...")
master.mav.command_long_send(
    master.target_system, master.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    0, 1, 0, 0, 0, 0, 0, 0
)
ack = master.recv_match(type='COMMAND_ACK', blocking=True, timeout=5)
print(f"[INFO] ARM ACK: {ack}")

time.sleep(2)


print(f"[INFO] Sending takeoff command to {TARGET_ALTITUDE}m...")
master.mav.command_long_send(
    master.target_system, master.target_component,
    mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
    0, 0, 0, 0, 0, 0, 0, TARGET_ALTITUDE
)
ack = master.recv_match(type='COMMAND_ACK', blocking=True, timeout=5)
print(f"[INFO] TAKEOFF ACK: {ack}")


print("[INFO] Waiting to reach target altitude...")
while True:
    msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=5)
    if msg:
        current_alt = msg.relative_alt / 1000.0  # mm to m
        print(f"[DEBUG] Current Altitude: {current_alt:.2f}m")
        if abs(current_alt - TARGET_ALTITUDE) < 0.5:
            print(f"[INFO] Target altitude {TARGET_ALTITUDE}m reached.")
            break
    else:
        print("[WARN] No altitude data, retrying...")


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


print("[INFO] Opening camera stream...")
cap = cv2.VideoCapture(CAMERA_SOURCE)
if not cap.isOpened():
    raise IOError("Cannot open camera/stream")


print("[INFO] Starting target tracking loop...")
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] No frame received")
            send_velocity(0, 0, 0, 0)  
            continue

        
        results = model(frame)
        detections = results.pandas().xyxy[0]

        target = detections[detections['name'] == TARGET_CLASS]
        if not target.empty:
            target = target.iloc[target['area'].idxmax()]
            x_center = (target['xmin'] + target['xmax']) / 2
            y_center = (target['ymin'] + target['ymax']) / 2
            bbox_area = (target['xmax'] - target['xmin']) * (target['ymax'] - target['ymin'])

            frame_h, frame_w, _ = frame.shape
            cx, cy = frame_w / 2, frame_h / 2

            offset_x = (x_center - cx) / cx  # [-1, 1]
            offset_y = (y_center - cy) / cy  # [-1, 1]

            yaw_rate = -offset_x * KP_YAW
            vz = offset_y * KP_ALT

            AREA_FAR = 0.05 * frame_w * frame_h
            AREA_CLOSE = 0.2 * frame_w * frame_h
            if bbox_area < AREA_FAR:
                vx = SPEED
            elif bbox_area > AREA_CLOSE:
                vx = -SPEED / 2
            else:
                vx = 0

            print(f"[DEBUG] vx={vx:.2f}, vz={vz:.2f}, yaw_rate={yaw_rate:.2f}")

            send_velocity(vx, 0, vz, yaw_rate)

            label = f"{target['name']} {target['confidence']:.2f}"
            cv2.rectangle(frame, (int(target['xmin']), int(target['ymin'])),
                          (int(target['xmax']), int(target['ymax'])), (0,255,0), 2)
            cv2.putText(frame, label, (int(target['xmin']), int(target['ymin'])-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        else:
            print("[INFO] Target lost - hovering")
            send_velocity(0, 0, 0, 0)

        cv2.imshow('YOLOv5 Detection', np.squeeze(results.render()))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user")

finally:
    print("[INFO] Landing...")
    send_velocity(0, 0, 0, 0)
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND,
        0, 0, 0, 0, 0, 0, 0, 0
    )
    cap.release()
    cv2.destroyAllWindows()
