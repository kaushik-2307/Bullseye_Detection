import torch
import cv2
import numpy as np
from pymavlink import mavutil
import time
import threading

# Configuration
YOLO_MODEL = 'C:/Bullseye_Detection/runs/train/helipad19/weights/best.pt'
TARGET_CLASS = 'hotspot-y5zB'
CAMERA_SOURCE = 0
MAVLINK_CONNECTION = 'tcp:127.0.0.1:5762'
SPEED = 1.5
KP_YAW = 1.0
KP_ALT = 0.7
KP_VX = 0.8
TARGET_ALTITUDE = 15
CENTER_THRESHOLD_X = 0.02
COMMAND_RATE = 5  # Hz - send commands at this rate

# Global variables for velocity commands
current_vx = 0
current_vy = 0
current_vz = 0
current_yaw_rate = 0
command_lock = threading.Lock()
boot_time = None

print("Loading YOLOv5 model")
model = torch.hub.load('ultralytics/yolov5', 'custom', YOLO_MODEL)
model.conf = 0.7
model.cpu().float()
torch.set_num_threads(4)

print("Connecting to MAVLink")
master = mavutil.mavlink_connection(
    MAVLINK_CONNECTION, baud=115200, source_system=1, mavlink20=True
)

boot_time = time.time()
master.wait_heartbeat()
print(f"Connected to system (sys {master.target_system}, comp {master.target_component})")

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
    
    print("Arming...")
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 0, 0, 0, 0, 0, 0
    )
    master.recv_match(type='COMMAND_ACK', blocking=True)
    time.sleep(2)
    
    print(f"Taking off to {TARGET_ALTITUDE}m...")
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, TARGET_ALTITUDE
    )
    master.recv_match(type='COMMAND_ACK', blocking=True)

def send_velocity_command():
    """Send velocity command - called by continuous thread"""
    with command_lock:
        vx, vy, vz, yaw_rate = current_vx, current_vy, current_vz, current_yaw_rate
    
    type_mask = (
        1 + 2 + 4 +      # Position x,y,z ignored
        64 + 128 + 256 + # Accel x,y,z ignored
        1024 +           # Force ignored
        2048             # Yaw ignored (using yaw_rate)
    )
    
    elapsed = (time.time() - boot_time) * 1e3
    
    master.mav.set_position_target_local_ned_send(
        int(elapsed),
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        type_mask,
        0, 0, 0,
        vx, vy, vz,
        0, 0, 0,
        0, yaw_rate
    )

def command_sender_thread():
    """Continuously send velocity commands at fixed rate"""
    loop_delay = 1.0 / COMMAND_RATE
    
    while True:
        send_velocity_command()
        time.sleep(loop_delay)

def set_velocity(vx, vy, vz, yaw_rate):
    """Update velocity commands thread-safely"""
    with command_lock:
        global current_vx, current_vy, current_vz, current_yaw_rate
        current_vx = vx
        current_vy = vy
        current_vz = vz
        current_yaw_rate = yaw_rate

# Start continuous command sender thread
command_thread = threading.Thread(target=command_sender_thread, daemon=True)
command_thread.start()

# Arm and takeoff
arm_and_takeoff()
time.sleep(10)

print("Opening camera")
cap = cv2.VideoCapture(CAMERA_SOURCE)
if not cap.isOpened():
    raise IOError("Cannot open camera/stream")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame received")
            set_velocity(0, 0, 0, 0)
            continue
        
        results = model(frame)
        detections = results.pandas().xyxy[0]
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
            
            target_size = np.sqrt(bbox_area / (frame_w * frame_h))
            desired_size = 0.3
            
            vx = (target_size - desired_size) * SPEED
            vy = -offset_x * KP_VX * SPEED
            vz = offset_y * KP_ALT
            yaw_rate = offset_x * KP_YAW
            
            # Apply dead zones
            if abs(offset_x) < CENTER_THRESHOLD_X:
                vy = 0
                yaw_rate = 0
            if abs(offset_y) < CENTER_THRESHOLD_X:
                vz = 0
            if abs(vx) < 0.1:
                vx = 0
            
            print(f"[INFO] Target found - vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}, yaw_rate={yaw_rate:.2f}")
            set_velocity(vx, vy, vz, yaw_rate)
            
            # Draw detection
            label = f"{target['name']} {target['confidence']:.2f}"
            cv2.rectangle(frame, (int(target['xmin']), int(target['ymin'])),
                         (int(target['xmax']), int(target['ymax'])), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(target['xmin']), int(target['ymin']) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            print("Target not found - hovering")
            set_velocity(0, 0, 0, 0)
        
        cv2.imshow('YOLOv5 Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    print("Landing...")
    set_velocity(0, 0, 0, 0)
    time.sleep(1)
    
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
        0, 0, 0, 0, 0, 0, 0, 0
    )
    
    cap.release()
    cv2.destroyAllWindows()
