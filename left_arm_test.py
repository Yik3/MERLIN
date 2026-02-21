from Total_API import *

if __name__ == "__main__":
    IP = "169.254.128.18" # Left Arm IP
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

        # 4. Set Hand Position (Using C-API)

        print("\n--- Setting Hand Position ---")
        target = [0, 0, 0, 0, 0, 0]
        success = robot.set_hand_position(target)
        print(f"Command Sent: {success}")

        BASE_POSE = [0.195654, -0.494819, 0.131595, -2.14, -1.196, -2.337]
        # for i in range(6):
        #     if i != 1:
        #         continue
        #     BASE_POSE[i] += 0.15
        #     robot.move_arm_to_pose(BASE_POSE, speed=20, block=True)
        #     print(f"Moved to pose: {BASE_POSE}")
        #     time.sleep(1) # Pause to observe
        #     BASE_POSE[i] -= 0.2
        #     robot.move_arm_to_pose(BASE_POSE, speed=20, block=True)
        #     print(f"Moved to pose: {BASE_POSE}")
        #     time.sleep(1) # Pause to observe
        #     BASE_POSE[i] += 0.05 # Reset to original
        #     robot.move_arm_to_pose(BASE_POSE, speed=20, block=True)
        #     print(f"Moved to pose: {BASE_POSE}")
        #     time.sleep(1) # Pause to observe
        #     print(f"Completed movement for joint {i}")

        time.sleep(1) # Pause to observe
        BASE_POSE[1] += 0.15
        robot.move_arm_to_pose(BASE_POSE, speed=20, block=True)
        print(f"Moved to pose: {BASE_POSE}")
        time.sleep(1) # Pause to observe
        BASE_POSE[1] -= 0.2
        robot.move_arm_to_pose(BASE_POSE, speed=20, block=True)
        print(f"Moved to pose: {BASE_POSE}")
        time.sleep(1) # Pause to observe
        BASE_POSE[1] += 0.05 # Reset to original
        robot.move_arm_to_pose(BASE_POSE, speed=20, block=True)
        print(f"Moved to pose: {BASE_POSE}")
        time.sleep(1) # Pause to observe
        
        for i in range(10):
            BASE_POSE[1] += 0.015
            time.sleep(0.05)
            robot.move_arm_to_pose(BASE_POSE, speed=20, block=True)
        
        for i in range(10):
            BASE_POSE[1] -= 0.02
            time.sleep(0.05)
            robot.move_arm_to_pose(BASE_POSE, speed=20, block=True)
        
        for i in range(10):
            BASE_POSE[1] += 0.005
            time.sleep(0.05)
            robot.move_arm_to_pose(BASE_POSE, speed=20, block=True)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'robot' in locals():
            robot.disconnect()
            RoboticArm.rm_destroy()

