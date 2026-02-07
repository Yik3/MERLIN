from Total_API import *
import numpy as np
Z_min = 0.065556
X_min = 0.145337
X_max = 0.422297
Z_max = 0.356552
Y_min = -0.49027
Y_max = -0.365178

if __name__ == "__main__":
    IP = "169.254.128.19" # Left Arm IP
    
    try:
        # 1. Initialize
        print("Initializing Robot API...")
        robot = RobotControlAPI(IP)

        # 2. Read Arm Pose (Using updated method)
        print("\n--- Reading Arm Pose ---")
        arm_data = robot.get_current_arm_pose()
        if arm_data:
            joints, pose = arm_data
            print(f"Joints: {joints}")
            print(f"Pose:   {pose}")

    except Exception as e:
        print(f"Main Error: {e}")
    finally:
        if 'robot' in locals():
            robot.disconnect()
            RoboticArm.rm_destroy()