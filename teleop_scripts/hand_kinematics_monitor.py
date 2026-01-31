import numpy as np
import time
from scipy.spatial.transform import Rotation as R
from ctypes import byref
from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
from Robotic_Arm.rm_ctypes_wrap import rm_inverse_kinematics_params_t, rm_current_arm_state_t, rm_get_current_arm_state

LEFT_ARM_IP = "169.254.128.18"
RIGHT_ARM_IP = "169.254.128.19"
PORT = 8080

def get_pose(arm):
    state_struct = rm_current_arm_state_t()
    ret = rm_get_current_arm_state(arm.handle, byref(state_struct))
    
    if ret == 0:
        pos = state_struct.pose.position
        quat = state_struct.pose.quaternion
        
        rotation = R.from_quat([quat.x, quat.y, quat.z, quat.w])
        rot_matrix = rotation.as_matrix()
        
        transform = np.eye(4)
        transform[:3, :3] = rot_matrix
        transform[:3, 3] = [pos.x, pos.y, pos.z]
        
        return transform
    
    return None

def get_joints(arm):
    ret, state = arm.rm_get_current_arm_state()
    return state.get('joint') if ret == 0 else None

def forward_kinematics(arm, joints):
    return arm.rm_algo_forward_kinematics(joints, flag=0)

def inverse_kinematics(arm, target_pose, current_joints):
    params = rm_inverse_kinematics_params_t(
        q_in=current_joints,
        q_pose=target_pose,
        flag=0
    )
    
    ret, joints = arm.rm_algo_inverse_kinematics(params)
    return (ret == 0, joints)

def move_to_pose(arm, target_pose, speed=20, blocking=True):
    current_joints = get_joints(arm)
    if not current_joints:
        return False
    
    success, target_joints = inverse_kinematics(arm, target_pose, current_joints)
    if not success:
        return False
    
    block_flag = 1 if blocking else 0
    return arm.rm_movej(target_joints, speed, 0, 0, block_flag) == 0

def move_to_joints_follow(arm, target_joints):
    return arm.rm_movej_follow(target_joints) == 0

def move_to_pose_follow(arm, target_pose):
    current_joints = get_joints(arm)
    if not current_joints:
        return False
    
    success, target_joints = inverse_kinematics(arm, target_pose, current_joints)
    if not success:
        return False
    
    return move_to_joints_follow(arm, target_joints)

def monitor_pose(hand='right'):
    arm_ip = RIGHT_ARM_IP if hand == 'right' else LEFT_ARM_IP
    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    if arm.rm_create_robot_arm(ip=arm_ip, port=PORT).id == -1:
        return
    
    try:
        while True:
            transform = get_pose(arm)
            joints = get_joints(arm)
            if transform is not None and joints:
                pos = transform[:3, 3]
                rot = R.from_matrix(transform[:3, :3])
                quat = rot.as_quat()
                
                print(f"Pose: x={pos[0]:.4f} y={pos[1]:.4f} z={pos[2]:.4f} qx={quat[0]:.4f} qy={quat[1]:.4f} qz={quat[2]:.4f} qw={quat[3]:.4f}")
                print(f"Joints: {[f'{j:.2f}' for j in joints]}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        arm.rm_delete_robot_arm()
        RoboticArm.rm_destroy()

if __name__ == '__main__':
    monitor_pose('right')
