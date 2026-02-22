import numpy as np
import time
import os
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R  # [新增] 引入 scipy 用于空间变换
#from Total_API import *
#from Safety_Constraint import * # ================= CONFIGURATION =================
# 请替换为你的实际文件路径
ENC_NPY = '/home/classysh/MERLIN/MERLIN/data/211data/encoder/processed_encoder_t/adc_data_20260212013551.npy'
ACT_NPY = '/home/classysh/MERLIN/MERLIN/data/211data/action/processed_action_t/iphone_data_20260210_234313.npy'
SYNC_NPZ = '/home/classysh/MERLIN/MERLIN/data/211data/syncs/video_recording_realsense#20260210234343_sync.npz'

range_map = {
    'CH0': (1286, 2400),
    'CH1': (1413, 1840),
    'CH2': (1900, 2883),
    'CH3': (1902, 2742),
    'CH4': (1750, 2700),
    'CH5': (1970, 2667) 
}
IP = '169.254.128.19'
ENABLE_ROBOT = False  # Set to True to actually move the robot
FPS = 30
BASE_POSE = [-0.298979, -0.303811, 0.308773, 1.823, -0.94, 0.221]

# [新增] Transformation 配置 (来自 Script B)
# Offset: 法兰中心 -> Sensor 中心 (在法兰/Sensor坐标系下)
OFFSET_VECTOR = [0.0017, 0.03088, 0.19273] 
# Y轴的额外偏移修正 (来自 Script B 的 target_pose[1] -= 0.46)
Y_AXIS_EXTRA_OFFSET = -0.46 
# =================================================

def smooth_data(data, window=8):
    """ 对数据进行前后 window 帧的移动平均 """
    padded = np.pad(data, ((window, window), (0, 0)), mode='edge')
    kernel_size = window * 2 + 1
    smoothed = np.zeros_like(data)
    for dim in range(data.shape[1]):
        col = padded[:, dim]
        ret = np.cumsum(col, dtype=float)
        ret[kernel_size:] = ret[kernel_size:] - ret[:-kernel_size]
        moving_sum = ret[kernel_size - 1 : kernel_size - 1 + len(data)]
        smoothed[:, dim] = moving_sum / kernel_size
    return smoothed

def map_encoder_to_motor(encoder_vals, gain = 0.83):
    motor_vals = []
    MAX_VAL = 1000 # 假设电机最大值，请根据 Total_API 调整
    for i in range(6):
        ch_name = f'CH{i}'
        if ch_name in range_map:
            min_val, max_val = range_map[ch_name]
            # 线性映射
            mapped_val = gain * (encoder_vals[i] - min_val) * MAX_VAL/ (max_val - min_val)
            motor_vals.append(int(mapped_val))
        else:
            motor_vals.append(0)  
    ret_motor_vals = [motor_vals[1], motor_vals[2], motor_vals[3], motor_vals[4], motor_vals[5], motor_vals[0]]
    return ret_motor_vals

def recover_trajectory(pos_B_list, quat_B_list, offset_vector, base_pose):
    """
    批量处理还原函数
    pos_B_list: 传感器测量的相对位移列表
    quat_B_list: 传感器测量的姿态列表 (Delta Rotation)
    offset_vector: 杠杆臂向量
    base_pose: 机器人初始位姿
    """
    xyz_check = [[], [], []]
    euler_check = [[], [], []]
    
    # 准备 Base Rotation
    r_base = R.from_euler('xyz', base_pose[3:6], degrees=False)
    
    # 准备 Offset
    v_offset = np.array(offset_vector)
    
    for i in range(len(pos_B_list)):
        # 当前传感器数据
        pos_meas = pos_B_list[i]
        quat_meas = quat_B_list[i]
        
        # 1. 计算当前的 Delta 旋转 (相对于起始时刻)
        r_delta = R.from_quat(quat_meas)
        
        # 2. 计算因为旋转产生的“杠杆位移” (Lever Arm Shift)
        # 逻辑：当前杠杆状态 - 初始杠杆状态
        # 这就是“绕圈”产生的额外位移
        lever_arm_shift = r_delta.apply(v_offset) - v_offset
        
        # 3. 还原法兰的 Delta 位移
        # 法兰位移 = 测量位移 - 杠杆位移
        flange_delta_pos = pos_meas - lever_arm_shift
        
        # 4. 叠加到 Base 初始位置 (得到绝对坐标)
        final_pos = flange_delta_pos + np.array(base_pose[0:3])
        
        # 5. 计算绝对姿态 (用于发送给机器人)
        # 姿态 = Sensor变动 * 初始姿态 (或者 初始 * 变动，取决于控制逻辑)
        # 这里假设 Sensor 变动是基于当前工具坐标系的
        r_total = r_delta * r_base
        final_euler = r_total.as_euler('xyz', degrees=False)
        
        # 存入列表用于绘图
        for j in range(3):
            xyz_check[j].append(final_pos[j])
            euler_check[j].append(final_euler[j])
            
    return xyz_check, euler_check

def main():
    # 1. Load Data
    print("Loading data files...")
    try:
        raw_enc = np.load(ENC_NPY)  # [T_enc, 6]
        raw_act = np.load(ACT_NPY)  # [T_act, 6] (Deltas)
        sync_data = np.load(SYNC_NPZ)
        
        frame_idxs = sync_data['frame_idx'].astype(int)
        enc_idxs = sync_data['encoder_idx'].astype(int)
        pose_idxs = sync_data['pose_idx'].astype(int)
        
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # 2. Pre-process Data
    print("Processing: Smoothing Encoder...")
    final_enc_data = smooth_data(raw_enc, window=8)

    print("Processing: Integrating Action Deltas -> Relative Trajectory...")
    # Action[t] = Action[t-1] + Delta[t]
    # 这一步还原出了相对于起点(0,0,0)的轨迹 (Relative Pose in Sensor Frame)
    relative_trajectory = np.cumsum(raw_act, axis=0)

    # 3. Initialize Robot
    if ENABLE_ROBOT:
        print(f"Connecting to robot at {IP}...")
        robot = RobotControlAPI(IP)
        print("Moving to BASE_POSE...")
        robot.move_arm_to_pose(BASE_POSE)
        robot.set_hand_position([0]*6)
        time.sleep(2.0)
    else:
        print("Robot Disabled (Simulation Mode)")

    rec_xyz = []
    rec_euler = []
    rec_hand = []

    print(f"Starting Replay ({len(frame_idxs)} frames)...")
    
    # 准备 Offset 向量 (用于循环内计算)
    v_offset = np.array(OFFSET_VECTOR)

    # 4. Main Loop
    for i in range(len(frame_idxs)):
        p_idx = pose_idxs[i]
        e_idx = enc_idxs[i]
        
        if p_idx >= len(relative_trajectory) or e_idx >= len(final_enc_data):
            continue

        # --- B. Compute Target Arm Pose (TRANSFORMED) ---
        
        # 1. 获取当前帧的“相对”传感器数据 (Sensor Frame)
        # 假设 raw_act 顺序是 [x, y, z, rx, ry, rz]
        sens_rel_pos = relative_trajectory[p_idx][0:3]   # relative position
        sens_rel_euler = relative_trajectory[p_idx][3:6] # relative rotation (euler)

        # 2. 计算 Lever Arm Compensation (杠杆臂补偿)
        # 逻辑：因为传感器不在旋转中心，旋转会产生额外的位移，需要减去这个位移
        r_delta = R.from_euler('xyz', sens_rel_euler) # 创建旋转对象
        
        # 计算偏移量：(Rotated_Offset - Original_Offset)
        lever_arm_shift = r_delta.apply(v_offset) - v_offset
        
        # 修正位移：Sensor位移 - 杠杆产生的虚假位移 = 法兰真实位移
        flange_rel_pos = sens_rel_pos - lever_arm_shift

        mapped_pos = [
            flange_rel_pos[0],     # X_rob = Y_sens
            -flange_rel_pos[1],    # Y_rob = -X_sens
            flange_rel_pos[2]      # Z_rob = Z_sens
        ]
        
        mapped_euler = [
            sens_rel_euler[0],     # Rx_rob = Ry_sens
            -sens_rel_euler[1],    # Ry_rob = -Rx_sens
            sens_rel_euler[2]      # Rz_rob = Rz_sens
        ]

        # 4. 叠加到 BASE_POSE (Absolute Target)
        target_pose = [0.0] * 6
        
        # 位置叠加
        target_pose[0] = BASE_POSE[0] + mapped_pos[0]
        target_pose[1] = BASE_POSE[1] + mapped_pos[1]
        target_pose[2] = BASE_POSE[2] + mapped_pos[2]
        
        # [Script B 特有] 额外的 Y 轴偏移
        target_pose[1] += Y_AXIS_EXTRA_OFFSET

        # 姿态叠加
        target_pose[3] = BASE_POSE[3] + mapped_euler[0]
        target_pose[4] = BASE_POSE[4] + mapped_euler[1]
        target_pose[5] = BASE_POSE[5] + mapped_euler[2]

        # Wrap angles (安全检查)
        for j in range(3, 6):
            while target_pose[j] > np.pi: target_pose[j] -= 2 * np.pi
            while target_pose[j] < -np.pi: target_pose[j] += 2 * np.pi

        # --- C. Compute Target Hand Gesture ---
        raw_hand_vals = final_enc_data[e_idx]
        target_hand = map_encoder_to_motor(raw_hand_vals)

        # --- D. Send Command ---
        if ENABLE_ROBOT and i > 30: # Skip first few frames for safety
            robot.move_arm_to_pose(target_pose)
            robot.set_hand_position(target_hand)

        # --- E. Record Data ---
        rec_xyz.append(target_pose[:3])
        rec_euler.append(target_pose[3:])
        rec_hand.append(target_hand)

        if i % 10 == 0:
            # 打印少量信息用于调试
            print(f"Frame {i} | Tgt: {np.round(target_pose, 3)}")
        
        # 控制循环频率，如果非实时系统需要 sleep
        # time.sleep(0.005) 

    print("Replay Finished.")

    # 5. Visualization
    rec_xyz = np.array(rec_xyz)
    rec_euler = np.array(rec_euler)
    rec_hand = np.array(rec_hand)
    t_steps = np.arange(len(rec_xyz))
    np.savez("replay_results.npz", xyz=rec_xyz, euler=rec_euler, hand=rec_hand)
    # Plot XYZ
    plt.figure(figsize=(10, 6))
    plt.plot(t_steps, rec_xyz[:, 0], label='X', color='r')
    plt.plot(t_steps, rec_xyz[:, 1], label='Y', color='g')
    plt.plot(t_steps, rec_xyz[:, 2], label='Z', color='b')
    plt.title("Transformed Robot Trajectory (XYZ)")
    plt.legend()
    plt.grid(True)
    plt.savefig("replay_xyz.png")

if __name__ == "__main__":
    main()