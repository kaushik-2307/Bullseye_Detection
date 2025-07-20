# 🚀 YOLOv5 + PyMAVLink real-time drone control
import torch
import cv2
import numpy as np
from pymavlink import mavutil
import time

# === CONFIGURATION ===
YOLO_MODEL = 'C:/Users/nayak/Documents/Bullseye_Detection/yolov5/runs/train/bullseye_detector8/weights/best.pt'     # Or yolov5n.pt for nano model
TARGET_CLASS = 'person'       # Class name to track
CAMERA_SOURCE = 0             # 0=USB Cam, or RTSP/UDP URL e.g., 'udp://0.0.0.0:5600'
MAVLINK_CONNECTION = 'tcp:127.0.0.1:5762'  # Or serial e.g., 'serial:/dev/serial0:115200'
SPEED = 5                   # m/s forward speed
KP_YAW = 0.005                # Proportional gain for yaw
KP_ALT = 0.005                # Proportional gain for altitude

# === LOAD YOLOv5 MODEL ===
print("[INFO] Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'custom', YOLO_MODEL)
model.conf = 0.4  # Confidence threshold

# === CONNECT TO AUTOPILOT ===
print("[INFO] Connecting to MAVLink...")
master = mavutil.mavlink_connection(MAVLINK_CONNECTION)
master.wait_heartbeat()
print(f"[INFO] Connected to system (system {master.target_system}, component {master.target_component})")
print(f"[INFO] Connected to system (system {master.target_system}, component {master.target_component})")

# === SEND FAKE POSITION ESTIMATE FOR EKF INIT (SITL workaround) ===
def send_fake_position_estimate():
    print("[INFO] Sending fake position estimates for EKF init...")
    for i in range(50):  # send for ~5 seconds
        master.mav.vision_position_estimate_send(
            int(time.time() * 1e6),  # usec
            0, 0, -1,                # x, y, z (NED, z negative up)
            0, 0, 0                  # roll, pitch, yaw
        )
        time.sleep(0.1)

send_fake_position_estimate()


# === SET GUIDED MODE FIRST ===
print("[INFO] Setting GUIDED mode...")
print(f"[DEBUG] Sending GUIDED mode to system {master.target_system}, component {master.target_component}")
master.set_mode('GUIDED')
ack = master.recv_match(type='COMMAND_ACK', blocking=True)
print(f"[INFO] GUIDED mode ACK: {ack}")
if hasattr(ack, 'result'):
    print(f"[DEBUG] GUIDED mode ACK result: {ack.result}")
time.sleep(10)  # Short delay to ensure mode change

# === ARM DRONE AFTER GUIDED ===
master.mav.command_long_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    0,  # 0=disarm, 1=arm
    1, 0, 0, 0, 0, 0, 0, 0
)
print("[INFO] Drone armed, ready for takeoff")
time.sleep(5)  # Short delay to ensure arming

# === TAKEOFF ===
print("[INFO] Sending takeoff command...")
print(f"[DEBUG] Sending TAKEOFF to system {master.target_system}, component {master.target_component}")
master.mav.command_long_send(master.target_system,master.target_component,mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 15, 15)  # param7=15m altitude, param8=0

# Wait for takeoff confirmation and print ACK
ack = master.recv_match(type='COMMAND_ACK', blocking=True)
print(f"[INFO] Takeoff ACK: {ack}")
if hasattr(ack, 'result'):
    print(f"[DEBUG] Takeoff ACK result: {ack.result}")
print("[INFO] Takeoff command sent, drone ascending...")

# === HELPER FUNCTION: Send velocity ===
def send_velocity(vx, vy, vz, yaw_rate):
    master.mav.set_position_target_local_ned_send(
        0,  # time_boot_ms
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED, int(0b0000111111000111), 0, 0, 0, vx, vy, vz, 0, 0, 0, 0, 0)

# === OPEN CAMERA STREAM ===
print("[INFO] Opening camera stream...")
cap = cv2.VideoCapture(CAMERA_SOURCE)
if not cap.isOpened():
    raise IOError("Cannot open camera/stream")

# === MAIN LOOP ===
print("[INFO] Starting detection and control loop...")
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] No frame received from camera")
            continue

        # Run YOLOv5 detection
        results = model(frame)
        detections = results.pandas().xyxy[0]  # Pandas DataFrame

        # Find target class
        target = detections[detections['name'] == TARGET_CLASS]
        if not target.empty:
            # Take the largest detection (closest object)
            target = target.iloc[target['area'].idxmax()]

            # Bounding box center
            x_center = (target['xmin'] + target['xmax']) / 2
            y_center = (target['ymin'] + target['ymax']) / 2

            # Frame center
            frame_h, frame_w, _ = frame.shape
            cx = frame_w / 2
            cy = frame_h / 2

            # Calculate offsets
            offset_x = x_center - cx
            offset_y = y_center - cy

            # Normalize offsets
            norm_x = offset_x / cx  # [-1, 1]
            norm_y = offset_y / cy  # [-1, 1]

            # Simple proportional controller
            yaw_rate = -norm_x * KP_YAW  # Yaw to center
            vz = norm_y * KP_ALT         # Up/down to center
            vx = SPEED                   # Forward

            # Send MAVLink command
            send_velocity(vx, 0, vz, yaw_rate)

            # Draw detection
            label = f"{target['name']} {target['confidence']:.2f}"
            cv2.rectangle(frame, (int(target['xmin']), int(target['ymin'])),
                          (int(target['xmax']), int(target['ymax'])), (0,255,0), 2)
            cv2.putText(frame, label, (int(target['xmin']), int(target['ymin'])-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        else:
            # No target found, hover
            send_velocity(0, 0, 0, 0)

        # Show frame
        cv2.imshow('YOLOv5 Detection', np.squeeze(results.render()))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user")

finally:
    print("[INFO] Landing and closing...")
    send_velocity(0, 0, 0, 0)  # Stop movement
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND,
        0, 0, 0, 0, 0, 0, 0, 0)
    cap.release()
    cv2.destroyAllWindows()
