'''
API Primitives for Robotic Arm Control
Adapted from Megumi's codebase

Author: Yike Shi
'''


import time
import numpy as np
import signal
import sys

from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

class RealManRobot:
    """Provides the motion primitive API to control the dual-hand robot pianist using Python API."""

    # Connection Constants
    LEFT_ARM_IP = "169.254.128.18"
    RIGHT_ARM_IP = "169.254.128.19"
    PORT = 8080

    # Home position joint angles (in degrees) - modify these based on your setup
    LEFT_HOME_JOINTS = [-98.23, -77.51, -107.24, -19.66, 84.58, -279.46]
    RIGHT_HOME_JOINTS = [91.45, 78.77, 109.33, -3.53, -95.01, 159.59]

    def __init__(self):
        """
        Initialize robot hand.
        """
        self.left_arm = None
        self.right_arm = None

    def connect(self):
        print("--- Connecting to Arms ---")
        try:
            # self.left_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
            self.right_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
            # if self.left_arm.rm_create_robot_arm(ip=self.LEFT_ARM_IP, port=self.PORT).id == -1:
            #     print("ERROR: Left arm connection failed.")
            #     return False
            # print("Left arm connected.")
            if self.right_arm.rm_create_robot_arm(ip=self.RIGHT_ARM_IP, port=self.PORT).id == -1:
                print("ERROR: Right arm connection failed.")
                return False
            print("Right arm connected.")
            return True
        except Exception as e:
            print(f"ERROR: Failed to connect to arms. Is the library imported? Error: {e}")
            return False
    
    def move_Arm(self, arm_side, joint_angles, speed=50):
        """
        Move the specified arm to the given joint angles at the specified speed.

        Args:
            arm_side (str): 'left' or 'right' to specify which arm to move.
            joint_angles (list): List of 6 joint angles in degrees.
            speed (int): Speed percentage (0-100).
        """
        if arm_side.lower() == 'left':
            arm = self.left_arm
        else:
            arm = self.right_arm
        '''
        arm.rm_move_joint(joint_angles, speed)
        print(f"{arm_side.capitalize()} arm moving to {joint_angles} at speed {speed}%.")
        '''
        arm.rm_movej(joint_angles, 5, 0, 0, 0)
        print(f"{arm_side.capitalize()} arm moving to {joint_angles} at speed {speed}%.")

    def debug_joint_positions(self, hand='both'):
        """Print current and target joint positions for debugging."""
        hands_to_check = ['left', 'right'] if hand == 'both' else [hand]
        
        for hand_name in hands_to_check:
            arm_instance = self.left_arm if hand_name == 'left' else self.right_arm
            if arm_instance is None:
                print(f"ERROR: {hand_name} arm not connected.")
                continue
                
            home_joints = self.LEFT_HOME_JOINTS if hand_name == 'left' else self.RIGHT_HOME_JOINTS
            
            ret, state = arm_instance.rm_get_current_arm_state()
            if ret == 0 and 'joint' in state:
                current = state['joint']
                print(f"\n{hand_name.upper()} ARM:")
                print(f"  Current joints: {current}")
                print(f"  Target joints:  {home_joints}")
                
                diffs = [abs(home_joints[i] - current[i]) for i in range(len(current))]
                print(f"  Differences:    {[f'{d:.2f}' for d in diffs]}")
                print(f"  Total movement: {sum(diffs):.2f}°")
            else:
                print(f"ERROR: Could not read {hand_name} arm state")

    def _emergency_stop_handler(self, signum, frame):
        print("\n\n>>> Ctrl+C detected - Emergency stopping all arms! <<<")
        self.emergency_stop()
        print(">>> Emergency stop complete. Exiting program. <<<")
        sys.exit(0)

    def move_wrist(self, arm_side, target_pose):
        """
        Move the wrist of the specified arm to the given pose.

        Args:
            arm_side (str): 'left' or 'right' to specify which arm to move.
            target_pose (list): List of 6 pose values [x, y, z, rx, ry, rz].
        """
        arm_instance = self.left_arm if arm_side.lower() == 'left' else self.right_arm
        result = arm_instance.rm_movel(target_pose, 5, 0, 0, 1) # 5 is wrist speed

    def _get_arm_state(self, hand_name: str):
        arm_instance = self.left_arm if hand_name == 'left' else self.right_arm
        if arm_instance is None:
            return None
            
        ret_state, state_dict = arm_instance.rm_get_current_arm_state()
        if ret_state == 0 and state_dict.get('pose'):
            pose = state_dict['pose']
            # self.hands[hand_name]['current_pose'] = {'position': np.array(pose[:3]), 'orientation': np.array(pose[3:])}
            return pose
        return None

if __name__ == "__main__":
    robot = RealManRobot()
    if robot.connect():
        robot.debug_joint_positions('right')
        state = robot._get_arm_state('right')
        print(f"Right arm current pose: {state}")
        robot.move_wrist('right', [-0.197827, -0.479751, 0.461914, -0.103, -0.77, 1.711])
        time.sleep(2)
        robot.debug_joint_positions('right')
        state = robot._get_arm_state('right')
        print(f"Right arm current pose: {state}")