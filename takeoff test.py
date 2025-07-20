from pymavlink import mavutil
MAVLINK_CONNECTION = 'tcp:127.0.0.1:5762'

print("[INFO] Connecting to MAVLink...")
master = mavutil.mavlink_connection(MAVLINK_CONNECTION)
master.wait_heartbeat()
print(f"[INFO] Connected to system (system {master.target_system}, component {master.target_component})")

master.arducopter_arm()
master.mav.command_long_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    0,  # 0=disarm, 1=arm
    1, 0, 0, 0, 0, 0, 0, 0
)

print("[INFO] Drone armed, ready for takeoff")
# Wait for arming confirmation
master.recv_match(type='COMMAND_ACK', blocking=True)

master.mav.command_long_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
    0, 0, 0, 0, 0, 0, 0, 15)  # Takeoff to 15m altitude
print("[INFO] Takeoff command sent, drone ascending...")
# Wait for takeoff confirmation
master.recv_match(type='COMMAND_ACK', blocking=True)