
"""
Drone takeoff, move forward 5m, and return to launch using velocity commands
"""

from pymavlink import mavutil
import time
import math

# Configuration
MAVLINK_CONNECTION = 'tcp:127.0.0.1:5762'
TARGET_ALTITUDE = 10  # meters
FORWARD_SPEED = 1.0  # m/s
DISTANCE_TO_MOVE = 5.0  # meters

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

# Wait for takeoff to complete
print("Waiting for takeoff to complete...")
time.sleep(10)

# Calculate time needed to move 5m at given speed
move_time = DISTANCE_TO_MOVE / FORWARD_SPEED
print(f"Moving forward 5m at {FORWARD_SPEED}m/s (will take ~{move_time:.1f}s)...")

# Send velocity command (forward/north direction: vx > 0)
# Velocity command in body frame: vx (forward), vy (right), vz (down)
master.mav.set_position_target_local_ned_send(
    0,  # time_boot_ms
    master.target_system, master.target_component,
    mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
    0b0000111111000111,  # type_mask (use velocity only)
    0, 0, 0,  # x, y, z (ignored)
    FORWARD_SPEED, 0, 0,  # vx, vy, vz (velocity in m/s)
    0, 0, 0,  # afx, afy, afz (acceleration - ignored)
    0, 0  # yaw, yaw_rate
)

# Keep sending velocity commands for the duration
start_time = time.time()
while time.time() - start_time < move_time:
    master.mav.set_position_target_local_ned_send(
        0,  # time_boot_ms
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        0b0000111111000111,  # type_mask
        0, 0, 0,  # x, y, z
        FORWARD_SPEED, 0, 0,  # vx, vy, vz
        0, 0, 0,  # afx, afy, afz
        0, 0  # yaw, yaw_rate
    )
    time.sleep(0.1)  # Send at 10Hz

# Stop forward movement
print("Stopping forward movement...")
master.mav.set_position_target_local_ned_send(
    0,  # time_boot_ms
    master.target_system, master.target_component,
    mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
    0b0000111111000111,  # type_mask
    0, 0, 0,  # x, y, z
    0, 0, 0,  # vx, vy, vz (stop)
    0, 0, 0,  # afx, afy, afz
    0, 0  # yaw, yaw_rate
)
time.sleep(1)

# Return to Launch
print("Returning to launch...")
master.mav.command_long_send(
    master.target_system, master.target_component,
    mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH, 0,
    0, 0, 0, 0, 0, 0, 0
)
master.recv_match(type='COMMAND_ACK', blocking=True, timeout=5)

print("Mission complete!")
