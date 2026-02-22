import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R

# ==========================================
# 0. Configuration & Constants
# ==========================================

# Offset: 法兰中心 -> Sensor 中心 (在 Sensor/法兰坐标系下)
# 对应之前的 OFFSET_VECTOR
OFFSET_VECTOR = [0.0017, 0.03088, 0.19273] 

# ==========================================
# 1. 通用工具函数 (Filter & Helpers)
# ==========================================

def butter_lowpass_filter(data, cutoff, fs, order=2):
    """
    Apply Butterworth Low Pass Filter
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # 避免由数据太短导致的 padlen 错误
    if len(data) < 10:
        return data
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data, axis=0) # axis=0 for column-wise filtering
    return y

def get_euler_from_quat(w, x, y, z):
    """
    Convert Quaternion (w, x, y, z) to Euler (x, y, z)
    """
    # Scipy expects [x, y, z, w]
    r = R.from_quat([x, y, z, w]) 
    return r.as_euler('xyz', degrees=False)

def apply_lever_arm_correction(pos, w, x, y, z, offset):
    """
    Transformation Step 1: Lever Arm Compensation
    只改变 XYZ，不改变姿态。
    公式: Flange_Pos = Sensor_Pos - R * Offset
    """
    # 构造旋转对象 (Scipy order: x, y, z, w)
    r = R.from_quat(np.stack([x, y, z, w], axis=-1))
    
    # 计算 Offset 在当前姿态下的旋转向量
    rotated_offset = r.apply(offset)
    
    # 修正位置
    corrected_pos = pos - rotated_offset
    
    return corrected_pos

def map_axes_z90(pos, euler):
    """
    Transformation Step 2: Axis Mapping (Flip Axis)
    将修正后的 XYZ 和 原始 Euler 进行坐标系转换
    Mapping:
      X_new = Y_old
      Y_new = -X_old
      Z_new = Z_old
      Rx_new = Ry_old
      Ry_new = -Rx_old
      Rz_new = Rz_old
    """
    # pos: [N, 3], euler: [N, 3]
    new_pos = np.zeros_like(pos)
    new_euler = np.zeros_like(euler)
    
    # Position Mapping
    new_pos[:, 0] = pos[:, 1]      # X = Y
    new_pos[:, 1] = -pos[:, 0]     # Y = -X
    new_pos[:, 2] = pos[:, 2]      # Z = Z
    
    # Rotation Mapping
    new_euler[:, 0] = euler[:, 1]  # Rx = Ry
    new_euler[:, 1] = -euler[:, 0] # Ry = -Rx
    new_euler[:, 2] = euler[:, 2]  # Rz = Rz
    
    return np.hstack([new_pos, new_euler])

# ==========================================
# 2. Encoder 处理逻辑 (保持不变)
# ==========================================

def process_single_encoder(csv_path, save_path=None, cutoff=2.0, fs=30):
    # Mapping definitions
    range_map_keys = ['CH0-ThumbLow', 'CH1-ThumbUp', 'CH2-Pointer', 
                      'CH3-Middle', 'CH4-Ring', 'CH5-Pinky']
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None

    extracted_data = []
    
    # 1. Extract raw data
    for col_name in range_map_keys:
        if col_name in df.columns:
            raw_vals = df[col_name].values
            # Fill NaN if any
            if np.isnan(raw_vals).any():
                raw_vals = pd.Series(raw_vals).fillna(method='ffill').fillna(0).values
            extracted_data.append(raw_vals)
        else:
            print(f"Warning: Column {col_name} not found in {csv_path}")
            return None

    # Shape: (N_frames, 6)
    data_array = np.column_stack(extracted_data)

    # 2. Apply LPF
    filtered_data = butter_lowpass_filter(data_array, cutoff, fs)

    # 3. Save
    if save_path:
        np.save(save_path, filtered_data)
        print(f"[Encoder] Saved {filtered_data.shape} to {os.path.basename(save_path)}")
    
    return filtered_data

# ==========================================
# 3. Action 处理逻辑 (核心修改)
# ==========================================

def process_single_action(txt_path, save_path=None, cutoff=2.0, fs=30):
    try:
        # Skip header, read comma separated
        # Format assumed: index, x, y, z, qw, qx, qy, qz
        raw_data = np.loadtxt(txt_path, delimiter=',', skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7))
    except Exception as e:
        print(f"Error reading {txt_path}: {e}")
        return None

    if raw_data.ndim == 1:
        raw_data = raw_data.reshape(1, -1)

    # Extract columns
    pos_raw = raw_data[:, 1:4]     # x, y, z
    quat_raw = raw_data[:, 4:8]    # qw, qx, qy, qz (N, 4)

    # --- Step 1: Transformation (Lever Arm Compensation) ---
    # 这一步修正 XYZ，消除杠杆效应，但保留原始姿态
    # pos_corrected shape: (N, 3)
    pos_corrected = apply_lever_arm_correction(
        pos_raw, 
        quat_raw[:, 0], quat_raw[:, 1], quat_raw[:, 2], quat_raw[:, 3], 
        OFFSET_VECTOR
    )

    # --- Step 2: Convert Quat to Euler ---
    # 我们需要 Euler 角度来进行 Axis Mapping
    # 注意：scipy from_quat 需要 (N, 4) [x,y,z,w]
    scipy_quats = np.column_stack([quat_raw[:, 1], quat_raw[:, 2], quat_raw[:, 3], quat_raw[:, 0]])
    r = R.from_quat(scipy_quats)
    euler_raw = r.as_euler('xyz', degrees=False) # (N, 3)

    # --- Step 3: Axis Mapping (Flip/Swap) ---
    # 输入: Corrected XYZ, Raw Euler
    # 输出: Transformed 6D Pose (Robot Frame)
    transformed_6d = map_axes_z90(pos_corrected, euler_raw)

    # --- Step 4: Unwrap Euler Angles ---
    # 防止角度跳变 (-pi -> pi) 影响滤波
    transformed_6d[:, 3:] = np.unwrap(transformed_6d[:, 3:], axis=0)

    # --- Step 5: Apply LPF (Smoothing) ---
    smoothed_poses = butter_lowpass_filter(transformed_6d, cutoff, fs)

    # --- Step 6: Calculate Delta Action ---
    # Action[t] = Pose[t] - Pose[t-1]
    delta_actions = np.zeros_like(smoothed_poses)
    if len(smoothed_poses) > 1:
        delta_actions[1:] = smoothed_poses[1:] - smoothed_poses[:-1]
    
    # --- Step 7: Save ---
    if save_path:
        np.save(save_path, delta_actions)
        print(f"[Action] Saved {delta_actions.shape} to {os.path.basename(save_path)}")

    return delta_actions

# ==========================================
# 4. Batch Processing 逻辑
# ==========================================

def batch_process_encoder(input_dir):
    output_dir = os.path.join(input_dir, "processed_encoder_t")
    os.makedirs(output_dir, exist_ok=True)
    
    files = [f for f in os.listdir(input_dir) if f.lower().endswith('.csv')]
    files.sort()
    
    print(f"Found {len(files)} CSV files for Encoder processing.")
    
    for f in files:
        in_path = os.path.join(input_dir, f)
        save_name = os.path.splitext(f)[0] + ".npy"
        out_path = os.path.join(output_dir, save_name)
        
        process_single_encoder(in_path, out_path, cutoff=0.4, fs=30)

def batch_process_action(input_dir):
    output_dir = os.path.join(input_dir, "processed_action_t")
    os.makedirs(output_dir, exist_ok=True)
    
    files = [f for f in os.listdir(input_dir) if f.lower().endswith('.txt')]
    files.sort()
    
    print(f"Found {len(files)} TXT files for Action processing.")
    
    for f in files:
        in_path = os.path.join(input_dir, f)
        save_name = os.path.splitext(f)[0] + ".npy"
        out_path = os.path.join(output_dir, save_name)
        
        process_single_action(in_path, out_path, cutoff=2.0, fs=30)

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    # 请修改这里的路径为你的实际路径
    #ENCODER_DIR = '/home/rm/Documents/MERLIN/training_data_check/encoder' 
    #ACTION_DIR = '/home/rm/Documents/MERLIN/training_data_check/action'

    ENCODER_DIR = '/home/classysh/MERLIN/MERLIN/data/211data/encoder'
    ACTION_DIR = '/home/classysh/MERLIN/MERLIN/data/211data/action'
    
    print("--- Starting Encoder Batch Processing ---")
    if os.path.exists(ENCODER_DIR):
        batch_process_encoder(ENCODER_DIR)
    else:
        print(f"Directory not found: {ENCODER_DIR}")
        
    print("\n--- Starting Action Batch Processing ---")
    if os.path.exists(ACTION_DIR):
        batch_process_action(ACTION_DIR)
    else:
        print(f"Directory not found: {ACTION_DIR}")