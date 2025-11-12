
"""
Simple drone takeoff and land script using pymavlink
"""

from pymavlink import mavutil
import time

# Configuration
MAVLINK_CONNECTION = 'tcp:127.0.0.1:5762'
TARGET_ALTITUDE = 10  # meters
HOVER_TIME = 10  # seconds

# Connect to drone
print("Connecting to MAVLink...")
master = mavutil.mavlink_connection(MAVLINK_CONNECTION, baud=115200, source_system=1, mavlink20=True)
master.wait_heartbeat()
print(f"Connected to system (sys {master.target_system}, comp {master.target_component})")

# Set GUIDED mode
print("Setting GUIDED mode...")
mode_id = master.mode_mapping()['GUIDED']
master.mav.command_long_send(
    master.target_system, master.target_component,
    mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
    mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
    mode_id, 0, 0, 0, 0, 0
)
master.recv_match(type='COMMAND_ACK', blocking=True, timeout=5)
time.sleep(1)

# Arm
print("Arming...")
master.mav.command_long_send(
    master.target_system, master.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
    1, 0, 0, 0, 0, 0, 0
)
master.recv_match(type='COMMAND_ACK', blocking=True, timeout=5)
time.sleep(2)

# Takeoff
print(f"Taking off to {TARGET_ALTITUDE}m...")
master.mav.command_long_send(
    master.target_system, master.target_component,
    mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0,
    0, 0, 0, 0, 0, 0, TARGET_ALTITUDE
)
master.recv_match(type='COMMAND_ACK', blocking=True, timeout=5)



# Hover
print(f"Hovering for {HOVER_TIME}s...")
time.sleep(HOVER_TIME)

# Land
print("Landing...")
master.mav.command_long_send(
    master.target_system, master.target_component,
    mavutil.mavlink.MAV_CMD_NAV_LAND, 0,
    0, 0, 0, 0, 0, 0, 0
)
master.recv_match(type='COMMAND_ACK', blocking=True, timeout=5)

print("Mission complete!")
