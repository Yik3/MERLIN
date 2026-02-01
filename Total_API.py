import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from ctypes import byref, c_int

# Import necessary classes from your existing environment
# Ensure these files are in the python path
from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
from Robotic_Arm.rm_ctypes_wrap import (
    rm_inverse_kinematics_params_t,
    rm_current_arm_state_t,
    rm_peripheral_read_write_params_t
)

# --- Constants from roh_registers_v1.py ---
# We define them here to avoid direct dependency on the file execution
ROH_FINGER_POS0_ADDR = 1145    # Start address for finger positions (Read)
ROH_FINGER_ANGLE0_ADDR = 1165  # Start address for finger angles (Read)
ROH_NODE_ID = 2                # Default ROHand Node ID
LEFT_ARM_IP = "169.254.128.18"
RIGHT_ARM_IP = "169.254.128.19"
PORT = 8080
class RobotControlAPI:
    def __init__(self, arm_ip, port=8080):
        """
        Initialize the Robot Control API.
        
        Args:
            arm_ip (str): IP address of the Robotic Arm.
            port (int): Port number (default 8080).
        """
        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        handle = self.arm.rm_create_robot_arm(ip=arm_ip, port=port)
        
        if handle.id == -1:
            raise ConnectionError(f"Failed to connect to Robot Arm at {arm_ip}:{port}")
        
        print(f"Connected to Robot Arm: {handle.id}")

        # Ensure Tool Port (Port 1) is configured for ROHand communication if necessary
        # Note: rm_set_hand_follow_pos usually handles configuration, 
        # but reading registers might require specific baudrate settings (defaults to 115200 for ROHand).

    def disconnect(self):
        """
        Disconnect the robot arm.
        """
        self.arm.rm_delete_robot_arm()

    # ------------------------------------------------------------------
    # 1. Move to specified pose (Arm)
    # ------------------------------------------------------------------
    def move_arm_to_pose(self, target_pose, speed=10, block=True):
        """
        Move the robotic arm to a specified 6D pose.
        It uses Inverse Kinematics to calculate joint angles and moves the arm.

        Args:
            target_pose (list[float]): Target pose [x, y, z, qw, qx, qy, qz].
                                       Position in meters (or mm depending on your config), 
                                       Rotation in Quaternion (w, x, y, z).
            speed (int): Movement speed percentage (1-100). Default is 20.
            block (bool): If True, wait until movement is complete.

        Returns:
            bool: True if movement command was sent successfully, False otherwise.
        """
        # Get current joint angles for IK reference
        ret, current_state = self.arm.rm_get_current_arm_state()
        if ret != 0:
            print("Error: Failed to get current arm state for IK.")
            return False
        
        current_joints = current_state['joint']

        # Construct IK parameters
        # Note: Ensure target_pose format matches what rm_inverse_kinematics expects.
        # Usually [x, y, z, rx, ry, rz] (Euler) or specific quaternion handling.
        # Based on hand_kinematics_monitor, we rely on the algo interface.
        
        params = rm_inverse_kinematics_params_t()
        params.q_in = (c_float * 7)(*current_joints) # Assuming 7 DOF or 6 DOF handles automatically
        params.q_pose = (c_float * 7)(*target_pose)  # [x, y, z, w, x, y, z]
        params.flag = 0 # 0 usually implies Quaternion input for this struct in some wrappers

        # Call Inverse Kinematics
        ret_ik, target_joints = self.arm.rm_algo_inverse_kinematics(params)
        
        if ret_ik != 0:
            print(f"Error: Inverse Kinematics failed with error code {ret_ik}")
            return False

        # Execute MoveJ (Joint Move)
        # block=1 means blocking, block=0 means non-blocking
        block_flag = 1 if block else 0
        ret_move = self.arm.rm_movej(target_joints, speed, 0, 0, block_flag)

        return ret_move == 0

    # ------------------------------------------------------------------
    # 2. Read current Arm Position
    # ------------------------------------------------------------------
    def get_current_arm_pose(self):
        """
        Read the current end-effector pose of the robotic arm.

        Args:
            None

        Returns:
            list[float]: Current pose [x, y, z, qw, qx, qy, qz].
                         Returns None if reading fails.
        """
        state_struct = rm_current_arm_state_t()
        ret = self.arm.handle.contents.rm_get_current_arm_state(self.arm.handle, byref(state_struct))

        if ret == 0:
            pos = state_struct.pose.position
            quat = state_struct.pose.quaternion
            # Return format: [x, y, z, w, x, y, z]
            return [pos.x, pos.y, pos.z, quat.w, quat.x, quat.y, quat.z]
        else:
            print(f"Error: Failed to get arm state. Code: {ret}")
            return None

    # ------------------------------------------------------------------
    # 3. Set current Hand Position
    # ------------------------------------------------------------------
    def set_hand_position(self, finger_positions, block=False):
        """
        Control the ROHand fingers to specific positions.

        Args:
            finger_positions (list[int]): List of 6 integers [Thumb, Index, Middle, Ring, Pinky, ThumbRot].
                                          Range: 0 (Open) to 65535 (Closed).
            block (bool): If True, wait for the hand to reach the position. 
                          For high-frequency control (teleop), set to False.

        Returns:
            bool: True if successful, False otherwise.
        """
        if len(finger_positions) != 6:
            print("Error: finger_positions must contain 6 integers.")
            return False

        # Ensure inputs are integers
        positions = [int(p) for p in finger_positions]

        # Use the built-in interface for hand control
        # This writes to ROH_FINGER_POS_TARGETx registers internally
        ret = self.arm.rm_set_hand_follow_pos(positions, block)

        return ret == 0

    # ------------------------------------------------------------------
    # 4. Read current Hand States
    # ------------------------------------------------------------------
    def get_hand_state(self, read_type='pos'):
        """
        Read the current state (Position or Angle) of the ROHand fingers via Modbus.

        Args:
            read_type (str): 'pos' to read raw positions (0-65535).
                             'angle' to read angles (scaled by 100).

        Returns:
            list[int]: List of 6 values corresponding to [Thumb, Index, Middle, Ring, Pinky, ThumbRot].
                       Returns None if Modbus read fails.
        """
        # Determine Register Address based on roh_registers_v1.py
        if read_type == 'pos':
            start_address = ROH_FINGER_POS0_ADDR  # 1145
        elif read_type == 'angle':
            start_address = ROH_FINGER_ANGLE0_ADDR # 1165
        else:
            print("Error: read_type must be 'pos' or 'angle'")
            return None

        # Configure Modbus Read Parameters
        params = rm_peripheral_read_write_params_t()
        params.port = 1              # Port 1 is the End-Effector/Tool Port
        params.address = start_address
        params.num = 6               # We want to read 6 fingers
        params.device_address = ROH_NODE_ID # ID 2

        # Execute Read
        # rm_read_multiple_holding_registers returns (ret_code, list_of_ints)
        ret, data = self.arm.rm_read_multiple_holding_registers(params)

        if ret == 0 and len(data) == 6:
            return data
        else:
            print(f"Error: Failed to read hand registers. Code: {ret}")
            return None


# --- Example Usage ---
if __name__ == "__main__":
    # Configuration
    IP = "192.168.1.18" # Change to your Robot IP
    
    try:
        # 1. Initialize
        robot = RobotControlAPI(IP)

        # 2. Read Arm Pose
        current_pose = robot.get_current_arm_pose()
        print(f"Current Arm Pose: {current_pose}")

        # 3. Read Hand State (Raw Position)
        hand_pos = robot.get_hand_state('pos')
        print(f"Current Hand Positions: {hand_pos}")

        # 4. Read Hand State (Angle)
        hand_angle = robot.get_hand_state('angle')
        if hand_angle:
            # Convert scaled int to float degrees: value / 100.0
            # Handle negative values for angles > 32768 (as per README logic)
            real_angles = []
            for a in hand_angle:
                val = a - 65536 if a > 32768 else a
                real_angles.append(val / 100.0)
            print(f"Current Hand Angles (deg): {real_angles}")

        # 5. Set Hand Position (Close Hand slightly)
        # 30000 is roughly half closed (Max 65535)
        target_hand = [30000, 30000, 30000, 30000, 30000, 0] 
        robot.set_hand_position(target_hand)
        time.sleep(1)

        # 6. Move Arm (Small Offset Example - BE CAREFUL)
        if current_pose:
            # Move Z up by 1cm (0.01m)
            target_pose = list(current_pose)
            target_pose[2] += 0.01 
            print("Moving arm up...")
            # robot.move_arm_to_pose(target_pose) # Uncomment to execute

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Cleanup
        if 'robot' in locals():
            robot.disconnect()
            RoboticArm.rm_destroy()