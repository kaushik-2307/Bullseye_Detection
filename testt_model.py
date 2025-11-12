
"""
Drone takeoff, detect helipad with YOLOv5, move towards it, and RTL when centered
"""

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
CENTER_THRESHOLD_Y = 0.02
COMMAND_RATE = 5  # Hz - send commands at this rate

# Global variables
current_vx = 0
current_vy = 0
current_vz = 0
current_yaw_rate = 0
command_lock = threading.Lock()
boot_time = None
master = None

def arm_and_takeoff():
    """Arm and take off to target altitude"""
    print("Setting GUIDED mode...")
    # Ensure mode mapping contains GUIDED
    mapping = master.mode_mapping()
    if 'GUIDED' not in mapping:
        print(f"[WARN] GUIDED mode not in mapping: {mapping}")
    mode_guided = mapping.get('GUIDED', 4)

    # send mode change
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        int(mode_guided), 0, 0, 0, 0, 0
    )
    ack = master.recv_match(type='COMMAND_ACK', blocking=True, timeout=5)
    print(f"Mode change ACK: {ack}")
    time.sleep(1)

    print("Arming...")
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
        1, 0, 0, 0, 0, 0, 0
    )
    ack = master.recv_match(type='COMMAND_ACK', blocking=True, timeout=5)
    print(f"Arm command ACK: {ack}")

    # wait up to 10s for motors to report armed via heartbeat
    armed = False
    for _ in range(20):
        hb = master.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
        if hb is None:
            continue
        base_mode = getattr(hb, 'base_mode', 0)
        # MAV_MODE_FLAG_SAFETY_ARMED == 128
        if base_mode & 128:
            armed = True
            break
    if not armed:
        print("[ERROR] Vehicle did not arm within timeout. Check safety, pre-arm checks, or permissions.")
        return
    print("Vehicle armed")

    print(f"Taking off to {TARGET_ALTITUDE}m...")
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0,
        0, 0, 0, 0, 0, 0, float(TARGET_ALTITUDE)
    )
    ack = master.recv_match(type='COMMAND_ACK', blocking=True, timeout=5)
    print(f"Takeoff command ACK: {ack}")

    # Optionally wait and confirm altitude increase via GLOBAL_POSITION_INT or VFR_HUD
    print("Waiting for climb...")
    climbed = False
    for _ in range(30):
        gps = master.recv_match(type=['GLOBAL_POSITION_INT', 'VFR_HUD'], blocking=True, timeout=1)
        if gps is None:
            continue
        if gps.get_type() == 'GLOBAL_POSITION_INT':
            # alt in millimeters
            alt_m = getattr(gps, 'relative_alt', None)
            if alt_m is not None:
                alt_m = alt_m / 1000.0
                if alt_m >= TARGET_ALTITUDE * 0.5:
                    climbed = True
                    break
        elif gps.get_type() == 'VFR_HUD':
            alt = getattr(gps, 'alt', None)
            if alt is not None and alt >= TARGET_ALTITUDE * 0.5:
                climbed = True
                break
    if climbed:
        print("Takeoff confirmed (altitude increasing)")
    else:
        print("Takeoff not confirmed - vehicle may still be climbing or telemetry unavailable")

def send_velocity_command():
    """Send velocity command at fixed rate"""
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
    global current_vx, current_vy, current_vz, current_yaw_rate
    with command_lock:
        current_vx = vx
        current_vy = vy
        current_vz = vz
        current_yaw_rate = yaw_rate

def return_to_launch():
    """Trigger return to launch"""
    print("Helipad centered! Triggering RTL...")
    set_velocity(0, 0, 0, 0)
    time.sleep(1)
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH, 0,
        0, 0, 0, 0, 0, 0, 0
    )

def main():
    global master, boot_time
    
    print("Loading YOLOv5 model (forcing CPU)...")
    # Force CPU-only execution
    device = torch.device('cpu')
    if torch.cuda.is_available():
        print("⚠️ CUDA is available but will be ignored - forcing CPU for model.")
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    # Disable cuDNN to avoid any accidental GPU acceleration
    try:
        torch.backends.cudnn.enabled = False
    except Exception:
        pass

    model = torch.hub.load('ultralytics/yolov5', 'custom', YOLO_MODEL)
    model.conf = 0.7
    model.to(device)
    model.eval()
    model.float()
    torch.set_num_threads(4)
    
    print("Connecting to MAVLink...")
    master = mavutil.mavlink_connection(MAVLINK_CONNECTION, baud=115200, source_system=1, mavlink20=True)
    boot_time = time.time()
    master.wait_heartbeat()
    print(f"Connected to system (sys {master.target_system}, comp {master.target_component})")
    
    # Start continuous command sender thread
    command_thread = threading.Thread(target=command_sender_thread, daemon=True)
    command_thread.start()
    
    # Arm and takeoff
    arm_and_takeoff()
    time.sleep(10)
    
    print("Opening camera...")
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
            
            # Run inference on CPU without gradients
            with torch.no_grad():
                results = model(frame)
            detections = results.pandas().xyxy[0]
            
            target = detections[detections['name'] == TARGET_CLASS]
            
            if not target.empty:
                # Get largest detection
                target = target.assign(area=(target['xmax'] - target['xmin']) * (target['ymax'] - target['ymin']))
                target = target.iloc[target['area'].idxmax()]
                
                x_center = (target['xmin'] + target['xmax']) / 2
                y_center = (target['ymin'] + target['ymax']) / 2
                bbox_area = target['area']
                
                frame_h, frame_w, _ = frame.shape
                cx, cy = frame_w / 2, frame_h / 2
                
                offset_x = (x_center - cx) / cx
                offset_y = (y_center - cy) / cy
                
                # Check if helipad is centered
                if abs(offset_x) < CENTER_THRESHOLD_X and abs(offset_y) < CENTER_THRESHOLD_Y:
                    print("[CENTERED] Helipad at center of frame!")
                    return_to_launch()
                    break
                
                # Calculate velocity commands
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
                
                if abs(offset_y) < CENTER_THRESHOLD_Y:
                    vz = 0
                
                if abs(vx) < 0.1:
                    vx = 0
                
                print(f"[INFO] Target found - offset_x={offset_x:.3f}, offset_y={offset_y:.3f} - vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}, yaw_rate={yaw_rate:.2f}")
                
                set_velocity(vx, vy, vz, yaw_rate)
                
                # Draw detection
                label = f"{target['name']} {target['confidence']:.2f}"
                cv2.rectangle(frame, (int(target['xmin']), int(target['ymin'])),
                            (int(target['xmax']), int(target['ymax'])), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(target['xmin']), int(target['ymin']) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw center crosshair
                cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                cv2.line(frame, (int(cx) - 20, int(cy)), (int(cx) + 20, int(cy)), (0, 0, 255), 1)
                cv2.line(frame, (int(cx), int(cy) - 20), (int(cx), int(cy) + 20), (0, 0, 255), 1)
            
            else:
                print("[WARNING] Target not found - hovering")
                set_velocity(0, 0, 0, 0)
            
            # Try to display frame. Some OpenCV builds (headless) lack GUI support
            # and will raise cv2.error on imshow/destroyAllWindows. In that case
            # save the latest frame to disk so you can inspect it remotely.
            try:
                cv2.imshow('YOLOv5 Helipad Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except cv2.error as e:
                # Headless environment: fall back to writing the latest frame to disk
                try:
                    out_path = 'latest_frame.jpg'
                    cv2.imwrite(out_path, frame)
                    print(f"[HEADLESS] OpenCV has no GUI support. Saved latest frame to {out_path}.")
                except Exception as ex:
                    print("[HEADLESS] Failed to write frame to disk:", ex)
                # Avoid busy-looping too fast when no GUI is available
                time.sleep(0.05)
                continue
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        print("Landing...")
        set_velocity(0, 0, 0, 0)
        time.sleep(1)
        master.mav.command_long_send(
            master.target_system, master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH, 0,
            0, 0, 0, 0, 0, 0, 0
        )
        try:
            cap.release()
        except Exception:
            pass

        try:
            cv2.destroyAllWindows()
        except cv2.error:
            # Ignore destroyAllWindows errors in headless builds
            pass

if __name__ == "__main__":
    main()
