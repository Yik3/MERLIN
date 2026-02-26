from Total_API import *
IP = "169.254.128.19" # Right Arm IP
    
try:
    # 1. Initialize
    print("Initializing Robot API...")
    robot = RobotControlAPI(IP)

    # 4. Set Hand Position (Using C-API)
    VALUE = MAX_VAL // 5
    print("\n--- Setting Hand Position ---")
    target = [0, 0, 0, 0, 0, 0]
    success = robot.set_hand_position(target)
    print(f"Command Sent: {success}")

except Exception as e:
    print(f"Main Error: {e}")
finally:
    if 'robot' in locals():
        robot.disconnect()
        RoboticArm.rm_destroy()