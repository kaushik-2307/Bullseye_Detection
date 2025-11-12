from pymavlink import mavutil
import time

# Connection settings
MAVLINK_CONNECTION = 'tcp:127.0.0.1:5762'  # Change to your connection string
TARGET_ALTITUDE = 10  # meters

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
    
    # Get mode ID
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
        1,  # 1 to arm, 0 to disarm
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

def land(master):
    """Land the vehicle"""
    print("Landing...")
    
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND,
        0,
        0, 0, 0, 0, 0, 0, 0
    )
    
    return wait_for_command_ack(master)

def wait_for_altitude(master, target_altitude, tolerance=0.5, timeout=30):
    """Wait until vehicle reaches target altitude"""
    print(f"Waiting to reach {target_altitude}m...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1)
        if msg:
            current_alt = msg.relative_alt / 1000.0  # Convert mm to meters
            print(f"Current altitude: {current_alt:.2f}m")
            
            if abs(current_alt - target_altitude) < tolerance:
                print(f"Reached target altitude: {current_alt:.2f}m")
                return True
        
        time.sleep(0.5)
    
    print("Timeout waiting for altitude")
    return False

def main():
    # Connect to vehicle
    print(f"Connecting to MAVLink on {MAVLINK_CONNECTION}...")
    master = mavutil.mavlink_connection(
        MAVLINK_CONNECTION,
        baud=115200,
        source_system=1,
        mavlink20=True
    )
    
    # Wait for heartbeat
    print("Waiting for heartbeat...")
    master.wait_heartbeat()
    print(f"Connected to system {master.target_system}, component {master.target_component}")
    
    try:
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
        
        # Hold for 10 seconds
        print("Holding position for 10 seconds...")
        time.sleep(10)
        
        # Land
        if not land(master):
            print("Failed to land")
            return
        
        print("Mission complete!")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        land(master)
    except Exception as e:
        print(f"Error: {e}")
        land(master)

if __name__ == "__main__":
    main()
