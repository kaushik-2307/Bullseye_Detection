from pymavlink import mavutil
import time

# Connection settings
MAVLINK_CONNECTION = 'tcp:127.0.0.1:5762'
TARGET_ALTITUDE = 10  # meters
FORWARD_DISTANCE = 2  # meters

boot_time = None

def wait_for_command_ack(master, timeout=5):
    """Wait for command acknowledgment"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        ack = master.recv_match(type='COMMAND_ACK', blocking=True, timeout=1)
        if ack:
            print(f"Command ACK: {ack.command} Result: {ack.result}")
            return ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED
    return False

def set_mode(master, mode_name):
    """Set flight mode"""
    print(f"Setting {mode_name} mode...")
    
    if mode_name not in master.mode_mapping():
        print(f"Unknown mode: {mode_name}")
        return False
    
    mode_id = master.mode_mapping()[mode_name]
    
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE,
        0,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id,
        0, 0, 0, 0, 0
    )
    
    return wait_for_command_ack(master)

def arm_throttle(master):
    """Arm the vehicle"""
    print("Arming throttle...")
    
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1,
        0, 0, 0, 0, 0, 0
    )
    
    return wait_for_command_ack(master)

def takeoff(master, altitude):
    """Takeoff to specified altitude"""
    print(f"Taking off to {altitude}m...")
    
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,
        0, 0, 0, 0, 0, 0,
        altitude
    )
    
    return wait_for_command_ack(master)

def send_velocity_command(master, vx, vy, vz, yaw_rate=0):
    """
    Send velocity command in body frame
    vx: forward velocity (m/s)
    vy: right velocity (m/s)
    vz: down velocity (m/s) - positive is DOWN
    yaw_rate: yaw rate (rad/s)
    """
    # Type mask to use velocity components only
    type_mask = (
        0b0000111111111000  # Position x,y,z + accel x,y,z ignored, velocity used
    )
    
    elapsed = (time.time() - boot_time) * 1000  # milliseconds
    
    master.mav.set_position_target_local_ned_send(
        int(elapsed),
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        type_mask,
        0, 0, 0,  # Position (ignored)
        vx, vy, vz,  # Velocity in m/s
        0, 0, 0,  # Acceleration (ignored)
        0, yaw_rate  # Yaw, yaw_rate
    )

def get_local_position(master):
    """Get current local position"""
    msg = master.recv_match(type='LOCAL_POSITION_NED', blocking=True, timeout=1)
    if msg:
        return msg.x, msg.y, msg.z
    return None, None, None

def wait_for_altitude(master, target_altitude, tolerance=0.5, timeout=30):
    """Wait until vehicle reaches target altitude"""
    print(f"Waiting to reach {target_altitude}m...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1)
        if msg:
            current_alt = msg.relative_alt / 1000.0
            print(f"Current altitude: {current_alt:.2f}m")
            
            if abs(current_alt - target_altitude) < tolerance:
                print(f"Reached target altitude: {current_alt:.2f}m")
                return True
        
        time.sleep(0.5)
    
    print("Timeout waiting for altitude")
    return False

def move_forward(master, distance, speed=0.5):
    """
    Move forward by specified distance at given speed
    Commands must be sent continuously at ~2-10Hz
    """
    print(f"Moving forward {distance}m at {speed}m/s...")
    
    # Get starting position
    x_start, y_start, z_start = get_local_position(master)
    if x_start is None:
        print("Could not get starting position")
        return False
    
    print(f"Starting position: x={x_start:.2f}, y={y_start:.2f}, z={z_start:.2f}")
    
    distance_covered = 0
    command_rate = 5  # Hz
    loop_delay = 1.0 / command_rate
    
    start_time = time.time()
    
    while distance_covered < distance:
        # Send velocity command continuously
        send_velocity_command(master, vx=speed, vy=0, vz=0, yaw_rate=0)
        
        # Get current position
        x_current, y_current, z_current = get_local_position(master)
        if x_current is not None:
            # Calculate distance traveled (forward is x in NED)
            distance_covered = abs(x_current - x_start)
            print(f"Distance covered: {distance_covered:.2f}m / {distance}m")
        
        time.sleep(loop_delay)
        
        # Safety timeout
        if time.time() - start_time > 30:
            print("Movement timeout")
            break
    
    # Stop
    print("Target distance reached, stopping...")
    for _ in range(10):  # Send stop commands for 2 seconds
        send_velocity_command(master, 0, 0, 0, 0)
        time.sleep(0.2)
    
    return True

def return_to_launch(master):
    """Return to launch position"""
    print("Returning to launch...")
    
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
        0,
        0, 0, 0, 0, 0, 0, 0
    )
    
    return wait_for_command_ack(master)

def main():
    global boot_time
    
    # Connect to vehicle
    print(f"Connecting to MAVLink on {MAVLINK_CONNECTION}...")
    master = mavutil.mavlink_connection(
        MAVLINK_CONNECTION,
        baud=115200,
        source_system=1,
        mavlink20=True
    )
    
    boot_time = time.time()
    
    # Wait for heartbeat
    print("Waiting for heartbeat...")
    master.wait_heartbeat()
    print(f"Connected to system {master.target_system}, component {master.target_component}")
    
    try:
        # Request position data at higher rate
        master.mav.request_data_stream_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_POSITION,
            10,  # 10 Hz
            1    # Start
        )
        
        # Set GUIDED mode
        if not set_mode(master, 'GUIDED'):
            print("Failed to set GUIDED mode")
            return
        
        time.sleep(2)
        
        # Arm
        if not arm_throttle(master):
            print("Failed to arm")
            return
        
        time.sleep(2)
        
        # Takeoff
        if not takeoff(master, TARGET_ALTITUDE):
            print("Failed to takeoff")
            return
        
        # Wait to reach altitude
        wait_for_altitude(master, TARGET_ALTITUDE, tolerance=1.0)
        
        # Stabilize for 3 seconds
        print("Stabilizing...")
        time.sleep(3)
        
        # Move forward 2 meters
        move_forward(master, FORWARD_DISTANCE, speed=0.5)
        
        # Hold position for 3 seconds
        print("Holding position for 3 seconds...")
        time.sleep(3)
        
        # Return to launch
        if not return_to_launch(master):
            print("Failed to RTL")
            return
        
        print("Mission complete!")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return_to_launch(master)
    except Exception as e:
        print(f"Error: {e}")
        return_to_launch(master)

if __name__ == "__main__":
    main()
