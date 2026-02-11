import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R

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

def transform_frame_z90(x, y, z, rx, ry, rz):
    """
    User defined coordinate transformation
    """
    ret_x = y
    ret_y = -x
    ret_z = z
    ret_rx = ry
    ret_ry = -rx
    ret_rz = rz
    return np.array([ret_x, ret_y, ret_z, ret_rx, ret_ry, ret_rz])

# ==========================================
# 2. Encoder 处理逻辑
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

    # 2. Apply LPF (No Normalization as requested)
    filtered_data = butter_lowpass_filter(data_array, cutoff, fs)

    # 3. Save
    if save_path:
        np.save(save_path, filtered_data)
        print(f"[Encoder] Saved {filtered_data.shape} to {os.path.basename(save_path)}")
    
    return filtered_data

# ==========================================
# 3. Action 处理逻辑
# ==========================================

def process_single_action(txt_path, save_path=None, cutoff=2.0, fs=30):
    try:
        # Skip header, read comma separated
        # Format assumed: index, x, y, z, qw, qx, qy, qz
        raw_data = np.loadtxt(txt_path, delimiter=',', skiprows=1,usecols=(0, 1, 2, 3, 4, 5, 6, 7))
    except Exception as e:
        print(f"Error reading {txt_path}: {e}")
        return None

    if raw_data.ndim == 1:
        raw_data = raw_data.reshape(1, -1)

    # Extract columns
    # index = raw_data[:, 0]
    pos = raw_data[:, 1:4]     # x, y, z
    quat = raw_data[:, 4:8]    # qw, qx, qy, qz

    processed_poses = []

    # 1. Convert & Transform
    for i in range(len(pos)):
        curr_pos = pos[i]
        curr_quat = quat[i] # qw, qx, qy, qz
        
        # Quat -> Euler
        curr_euler = get_euler_from_quat(curr_quat[0], curr_quat[1], curr_quat[2], curr_quat[3])
        
        # Coordinate Transform
        # Input: x, y, z, rx, ry, rz
        transformed_6d = transform_frame_z90(
            curr_pos[0], curr_pos[1], curr_pos[2],
            curr_euler[0], curr_euler[1], curr_euler[2]
        )
        processed_poses.append(transformed_6d)
    
    processed_poses = np.array(processed_poses) # Shape (T, 6)

    # 2. Unwrap Euler Angles (Critical before LPF!)
    # 这一步是为了防止 -3.14 跳变到 3.14 被 LPF 平滑成 0
    processed_poses[:, 3:] = np.unwrap(processed_poses[:, 3:], axis=0)

    # 3. Apply LPF
    smoothed_poses = butter_lowpass_filter(processed_poses, cutoff, fs)

    # 4. Calculate Delta Action
    # Action[t] = Pose[t] - Pose[t-1]
    # Action[0] = 0
    delta_actions = np.zeros_like(smoothed_poses)
    if len(smoothed_poses) > 1:
        delta_actions[1:] = smoothed_poses[1:] - smoothed_poses[:-1]
    
    # 5. Save
    if save_path:
        np.save(save_path, delta_actions)
        print(f"[Action] Saved {delta_actions.shape} to {os.path.basename(save_path)}")

    return delta_actions

# ==========================================
# 4. Batch Processing 逻辑
# ==========================================

def batch_process_encoder(input_dir):
    output_dir = os.path.join(input_dir, "processed_encoder")
    os.makedirs(output_dir, exist_ok=True)
    
    files = [f for f in os.listdir(input_dir) if f.lower().endswith('.csv')]
    files.sort()
    
    print(f"Found {len(files)} CSV files for Encoder processing.")
    
    for f in files:
        in_path = os.path.join(input_dir, f)
        # 保持文件名，只是后缀改为 .npy
        save_name = os.path.splitext(f)[0] + ".npy"
        out_path = os.path.join(output_dir, save_name)
        
        process_single_encoder(in_path, out_path, cutoff=0.4, fs=30)

def batch_process_action(input_dir):
    output_dir = os.path.join(input_dir, "processed_action")
    os.makedirs(output_dir, exist_ok=True)
    
    files = [f for f in os.listdir(input_dir) if f.lower().endswith('.txt')]
    files.sort()
    
    print(f"Found {len(files)} TXT files for Action processing.")
    
    for f in files:
        in_path = os.path.join(input_dir, f)
        save_name = os.path.splitext(f)[0] + ".npy"
        out_path = os.path.join(output_dir, save_name)
        
        process_single_action(in_path, out_path, cutoff=2.0, fs=50)

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    # 请修改这里的路径为你的实际路径
    ENCODER_DIR = '/home/classysh/MERLIN/MERLIN/data/210data/encoder'
    ACTION_DIR = '/home/classysh/MERLIN/MERLIN/data/210data/action'
    
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

        
    