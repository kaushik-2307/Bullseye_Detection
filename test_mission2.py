import torch
import cv2
import numpy as np
from pymavlink import mavutil
import time


YOLO_MODEL = 'C:/Bullseye_Detection/runs/train/helipad19/weights/best.pt'
TARGET_CLASS = 'hotspot-y5zB'   
CAMERA_SOURCE = 0
MAVLINK_CONNECTION = 'tcp:127.0.0.1:5762'
SPEED = 1.5                    # Increased base speed
KP_YAW = 1.0                   # Yaw control gain
KP_ALT = 0.7                   # Altitude control gain
KP_VX = 0.8                    # Increased horizontal control gain
TARGET_ALTITUDE = 15           # Target altitude in meters
CENTER_THRESHOLD_X = 0.02      # Reduced threshold (2% of frame) to allow finer movements
MAVLINK_SIGN_KEY = b'supersecurekey1234'
SIGNING_ID = 1


print("Loading YOLOv5 model")
model = torch.hub.load('ultralytics/yolov5', 'custom', YOLO_MODEL)
# Force CPU usage and float32 precision
model.conf = 0.7
model.cpu().float()  # Move to CPU and use float32
torch.set_num_threads(4)  # Limit CPU threads for stable performance


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
    # Type mask: set which dimensions to use
    # bit 1: x velocity
    # bit 2: y velocity
    # bit 3: z velocity
    # bit 9: yaw rate
    type_mask = (
        1 +  # Position x
        2 +  # Position y
        4 +  # Position z
        64 +  # Accel x
        128 +  # Accel y
        256 +  # Accel z
        1024 + # Force
        2048  # Yaw
    )
    
    elapsed = (time.time() - boot_time) * 1e3
    master.mav.set_position_target_local_ned_send(
        int(elapsed),
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,  # Use body frame
        type_mask,  # Use velocity components
        0, 0, 0,  # Position (ignored)
        vx, vy, vz,  # Velocity in m/s
        0, 0, 0,  # Acceleration (ignored)
        0, yaw_rate  # Yaw angle and yaw rate
    )
    print(f"Sending velocity: vx={vx:.2f} (fwd/back), vy={vy:.2f} (right/left), vz={vz:.2f} (down/up), yaw_rate={yaw_rate:.2f}")


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

            # Calculate normalized offsets (-1 to 1)
            offset_x = (x_center - cx) / cx  # Positive when target is to the right
            offset_y = (y_center - cy) / cy  # Positive when target is below

            # In body frame (FRD - Forward Right Down):
            # vx = forward/-backward
            # vy = right/left
            # vz = down/up (positive is DOWN in NED)
            
            # Move forward/backward based on target size (maintain distance)
            target_size = np.sqrt(bbox_area / (frame_w * frame_h))  # Normalized 0-1
            desired_size = 0.3  # Target should occupy ~30% of frame
            vx = (target_size - desired_size) * SPEED
            
            # Strafe left/right to center target horizontally
            vy = -offset_x * KP_VX * SPEED  # Negative because right is positive
            
            # Move up/down to center target vertically
            vz = offset_y * KP_ALT  # Positive offset_y (target below) = positive vz (move down)
            
            # Add yaw correction to keep target centered
            yaw_rate = offset_x * KP_YAW
            
            # Apply dead zone to prevent tiny movements
            if abs(offset_x) < CENTER_THRESHOLD_X:
                vy = 0
                yaw_rate = 0
            if abs(offset_y) < CENTER_THRESHOLD_X:
                vz = 0
            if abs(vx) < 0.1:  # Small forward/backward deadzone
                vx = 0
            print(f"[INFO] Moving: vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}, yaw_rate={yaw_rate:.2f}")
            send_velocity(vx, vy, vz, yaw_rate)
            
            label = f"{target['name']} {target['confidence']:.2f}"
            cv2.rectangle(frame, (int(target['xmin']), int(target['ymin'])),
                          (int(target['xmax']), int(target['ymax'])), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(target['xmin']), int(target['ymin']) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw movement debug info
            h, w = frame.shape[:2]
            # Draw center crosshair
            cv2.line(frame, (w//2, h//2-20), (w//2, h//2+20), (0,0,255), 2)
            cv2.line(frame, (w//2-20, h//2), (w//2+20, h//2), (0,0,255), 2)
            # Draw target position
            cv2.circle(frame, (int(x_center), int(y_center)), 5, (255,0,0), -1)
            # Draw movement arrows and text
            if abs(vx) > 0.1:
                text = "⟩" if vx > 0 else "⟨"
                cv2.putText(frame, f"Fwd/Back: {vx:.2f}", (10, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            if abs(vy) > 0.1:
                text = "→" if vy > 0 else "←"
                cv2.putText(frame, f"Right/Left: {vy:.2f}", (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            if abs(vz) > 0.1:
                text = "↓" if vz > 0 else "↑"
                cv2.putText(frame, f"Up/Down: {vz:.2f}", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        else:
            print("Target not found - hovering")
            send_velocity(0, 0, 0, 0)

        debug_frame = frame.copy()
        cv2.imshow('YOLOv5 Detection', debug_frame)
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