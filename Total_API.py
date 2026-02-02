import time
import json
import socket
import threading
from ctypes import byref, c_int, c_float

# Import RealMan SDK
from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
from Robotic_Arm.rm_ctypes_wrap import (
    rm_inverse_kinematics_params_t,
    rm_current_arm_state_t,
    rm_peripheral_read_write_params_t,
    rm_pose_t
)
from scipy.spatial.transform import Rotation as R

def convert_6d_to_7d(pose_6d):
    """
    6D Pose to 7D Pose Conversion for RealMan Inverse Kinematics.
    Input:  [x, y, z, rx, ry, rz] (Euler angles in radians)
    Output: [x, y, z, qw, qx, qy, qz] (RealMan inverse kinematics format)
    """
    # 1. Extract position and Euler angles
    pos = pose_6d[:3]   # [x, y, z]
    euler = pose_6d[3:] # [rx, ry, rz]

    # 2. Convert Euler angles to quaternion
    # Note: RealMan typically uses 'xyz' order for Euler angles
    r = R.from_euler('xyz', euler, degrees=False)
    quat = r.as_quat() 
    pose_7d = [pos[0], pos[1], pos[2], quat[3], quat[0], quat[1], quat[2]]
    
    return pose_7d

# --- Constants based on roh_registers_v1.py ---
ROH_NODE_ID = 2
BASE_ADDR_POS_CURRENT = 1145 # ROH_FINGER_POS0 (Read)
BASE_ADDR_ANGLE = 1165       # ROH_FINGER_ANGLE0 (Read)

class RobotControlAPI:
    def __init__(self, arm_ip, port=8080):
        """
        Initialize the Robot Control API.
        
        Args:
            arm_ip (str): IP address of the Robotic Arm.
            port (int): Port number (default 8080).
        """
        self.ip = arm_ip
        self.port = port
        
        # 1. Initialize C-API for Arm Control & Hand Write
        # Using TRIPLE_MODE as per your original codebase
        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.handle = self.arm.rm_create_robot_arm(ip=arm_ip, port=port)
        
        if self.handle.id == -1:
            raise ConnectionError(f"Failed to connect to Robot Arm (C-API) at {arm_ip}:{port}")
        
        print(f"Connected to Robot Arm (C-API): {self.handle.id}")

        # 2. Initialize Raw Socket for Hand Reading (Robust method)
        self.sock = None
        self._connect_socket()

        # 3. Ensure Port 1 is configured for ModbusRTU (115200 baud)
        # We use the socket to set this, as it proved successful in your test script
        self._init_modbus_mode_socket()

    def _connect_socket(self):
        """Establish a raw socket connection for robust reading."""
        try:
            if self.sock:
                self.sock.close()
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(2.0)
            self.sock.connect((self.ip, self.port))
            print("Connected to Robot Arm (Raw Socket)")
        except Exception as e:
            print(f"Socket connection failed: {e}")

    def _init_modbus_mode_socket(self):
        """
        Configure End-Effector Port (Port 1) to ModbusRTU (115200 baud).
        Command: {"command":"set_modbus_mode","port":1,"baudrate":115200,"timeout":2}
        """
        cmd = {
            "command": "set_modbus_mode",
            "port": 1,
            "baudrate": 115200,
            "timeout": 2
        }
        resp = self._send_socket_cmd(cmd)
        # Simple log to confirm configuration
        if resp:
             print(">>> Port 1 Configured to ModbusRTU via Socket")

    def _send_socket_cmd(self, cmd_dict):
        """Helper to send JSON command via raw socket."""
        if not self.sock:
            self._connect_socket()
        
        try:
            cmd_str = json.dumps(cmd_dict) + "\r\n"
            self.sock.send(cmd_str.encode('utf-8'))
            data = self.sock.recv(4096).decode('utf-8')
            return data
        except (socket.timeout, BrokenPipeError, ConnectionResetError) as e:
            print(f"Socket Error ({e}), reconnecting...")
            self._connect_socket()
            return None
        except Exception as e:
            print(f"Unknown Socket Error: {e}")
            return None

    def disconnect(self):
        """Clean up connections."""
        if self.sock:
            self.sock.close()
        self.arm.rm_delete_robot_arm()

    # ------------------------------------------------------------------
    # 1. Move to Specified Pose (Arm) - [Original Logic]
    # ------------------------------------------------------------------
    def move_arm_to_pose(self, target_pose, speed=10, block=True):
        """
        Move the robotic arm to a specified 6D pose using Inverse Kinematics.
        Fixed to use rm_pose_t struct instead of float array.
        """
        # 1. 统一转换为 7D 数据 [x, y, z, w, x, y, z]
        if len(target_pose) == 6:
            target_pose_7d = convert_6d_to_7d(target_pose)
        elif len(target_pose) == 7:
            target_pose_7d = target_pose
        else:
            print("Error: Target pose must be 6D or 7D")
            return False

        # 2. 获取当前关节角作为逆解初值
        ret, current_state = self.arm.rm_get_current_arm_state()
        if ret != 0:
            print("Error: Failed to get current arm state for IK.")
            return False
        
        current_joints = current_state['joint']
        
        # 3. 构造逆解参数 (Struct Construction)
        params = rm_inverse_kinematics_params_t()
        params.q_in = (c_float * 7)(*current_joints)
        
        # [FIX START] --------------------------------------------
        # 不能直接赋值数组，必须构造 rm_pose_t 结构体
        pose_struct = rm_pose_t()
        
        # 填充 Position (x, y, z)
        pose_struct.position.x = target_pose_7d[0]
        pose_struct.position.y = target_pose_7d[1]
        pose_struct.position.z = target_pose_7d[2]
        
        # 填充 Quaternion (w, x, y, z)
        # 注意: target_pose_7d 格式为 [x, y, z, w, x, y, z]
        pose_struct.quaternion.w = target_pose_7d[3]
        pose_struct.quaternion.x = target_pose_7d[4]
        pose_struct.quaternion.y = target_pose_7d[5]
        pose_struct.quaternion.z = target_pose_7d[6]
        
        # 将构造好的结构体赋值给参数
        params.q_pose = pose_struct
        # [FIX END] ----------------------------------------------

        params.flag = 0 # 0 代表输入的是四元数 (Quaternion)
       
        # 4. 调用逆解
        ret_ik, target_joints = self.arm.rm_algo_inverse_kinematics(params)
       
        if ret_ik != 0:
            print(f"Error: Inverse Kinematics failed with error code {ret_ik}")
            return False
            
        print(f"IK Result Joints: {list(target_joints)}")

        # 5. 执行运动
        block_flag = 1 if block else 0
        ret_move = self.arm.rm_movej(target_joints, speed, 0, 0, block_flag)

        return ret_move == 0

    def move_arm_to_joints(self, target_joints, speed=10, block=True):
        """
        Move the robotic arm to specified joint angles.

        Args:
            target_joints (list[float]): List of 6 joint angles.
            speed (int): Movement speed percentage (1-100).
            block (bool): If True, wait until movement is complete.

        Returns:
            bool: True if successful, False otherwise.
        """
        if len(target_joints) != 6:
            print("Error: target_joints must contain 6 joint angles.")
            return False
        block_flag = 1 if block else 0
        ret_move = self.arm.rm_movej(target_joints, speed, 0, 0, block_flag)

        return ret_move == 0
    # ------------------------------------------------------------------
    # 2. Read current Arm Position
    # ------------------------------------------------------------------
    def get_current_arm_pose(self):
        """
        Read the current end-effector pose of the robotic arm.

        Returns:
            tuple: (current_joints, current_pose)
                   current_joints: list[float]
                   current_pose: list[float] [x, y, z, qx, qy, qz]
            Returns None if reading fails.
        """
        ret, state = self.arm.rm_get_current_arm_state()
        
        if ret == 0 and 'joint' in state:
            current_joints = state['joint']
            current_pose = state['pose']  # Format: [x, y, z, qx, qy, qz]
            
            # Debug prints (Optional, can be removed)
            # print(f"  Current joints: {current_joints}")
            # print(f"  Current pose:   {current_pose}")
            
            return current_joints, current_pose
        else:
            print(f"ERROR: Could not read arm state, ret={ret}")
            return None

    # ------------------------------------------------------------------
    # 3. Set current Hand Position - [Original C-API Logic]
    # ------------------------------------------------------------------
    def set_hand_position(self, finger_positions, block=False):
        """
        Control the ROHand fingers.

        Args:
            finger_positions (list[int]): 6 integers [Thumb, Index, Middle, Ring, Pinky, ThumbRot].
                                          Range: 0 (Open) to 65535 (Closed).
            block (bool): Blocking call if True.

        Returns:
            bool: True if successful.
        """
        if len(finger_positions) != 6:
            print("Error: finger_positions must contain 6 integers.")
            return False

        positions = [int(p) for p in finger_positions]

        # Use C-API for writing (this worked in your original code)
        ret = self.arm.rm_set_hand_follow_pos(positions, block)

        return ret == 0

    # ------------------------------------------------------------------
    # 4. Read current Hand States - [New Socket Logic]
    # ------------------------------------------------------------------
    def get_hand_state(self, read_type='pos'):
        """
        Read ROHand state using Raw Socket (Robust Method).
        Iterates 6 times to read each register individually, mimicking try_socket_read.py.

        Args:
            read_type (str): 'pos' (0-65535) or 'angle' (float deg).

        Returns:
            list: 6 values corresponding to fingers. Returns None on failure.
        """
        results = []
        
        if read_type == 'pos':
            base_addr = BASE_ADDR_POS_CURRENT # 1145
        elif read_type == 'angle':
            base_addr = BASE_ADDR_ANGLE       # 1165
        else:
            print("Error: read_type must be 'pos' or 'angle'")
            return None

        # Loop 6 times to read each finger individually
        for i in range(6):
            addr = base_addr + i
            
            # Construct JSON command exactly like try_socket_read.py
            cmd = {
                "command": "read_holding_registers",
                "port": 1,
                "address": addr,
                "num": 1,        # Important: Read 1 by 1
                "device": ROH_NODE_ID
            }
            
            resp_str = self._send_socket_cmd(cmd)
            
            if not resp_str:
                return None
                
            try:
                # Parse JSON response
                # Example response: {"command":"read_holding_registers","data":[123],"ret":0}
                data_json = json.loads(resp_str)
                
                # Validation
                if "data" not in data_json:
                    # print(f"[Socket] No data in response: {resp_str}")
                    return None
                    
                val_list = data_json["data"]
                # Handle cases where data might be a list or a single int
                val = val_list[0] if isinstance(val_list, list) else int(val_list)
                
                if read_type == 'angle':
                    # Handle int16 complement for angles
                    if val > 32767: 
                        val -= 65536
                    results.append(val / 100.0)
                else:
                    results.append(val)
                    
            except Exception as e:
                print(f"[Socket] JSON Parsing Exception: {e} | Raw: {resp_str}")
                return None
                
            # Tiny sleep to prevents overloading the gateway (from your script)
            # time.sleep(0.01) 
            
        return results


# --- Example Usage ---
if __name__ == "__main__":
    # Configuration
    IP = "169.254.128.19" # Right Arm IP
    
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

        # 3. Read Hand State (Using Socket)
        print("\n--- Reading Hand State ---")
        hand_pos = robot.get_hand_state('pos')
        print(f"Hand Positions: {hand_pos}")

        hand_angle = robot.get_hand_state('angle')
        print(f"Hand Angles:    {hand_angle}")

        # 4. Set Hand Position (Using C-API)

        print("\n--- Setting Hand Position ---")
        target = [0, 0, 0, 0, 0, 0]
        success = robot.set_hand_position(target)
        print(f"Command Sent: {success}")

        # 5. Move Arm to Pose
        print("\n--- Moving Arm to Pose ---")
        target_pose = [-0.080309, -0.4453, 0.172321, -1.564, -1.163, -3.129]
        success = robot.move_arm_to_pose(target_pose)
        print(f"Move Command Sent: {success}")

    except Exception as e:
        print(f"Main Error: {e}")
    finally:
        if 'robot' in locals():
            robot.disconnect()
            RoboticArm.rm_destroy()

