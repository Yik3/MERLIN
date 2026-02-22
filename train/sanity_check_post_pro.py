import os
import numpy as np
import pandas as pd
import random
import matplotlib
# --- 关键修改 1: 设置后端为 Agg，必须在 import pyplot 之前 ---
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# ==========================================
# 配置路径 (请修改)
# ==========================================
RAW_ENCODER_DIR = '/home/classysh/MERLIN/MERLIN/data/211data/encoder'
RAW_ACTION_DIR = '/home/classysh/MERLIN/MERLIN/data/211data/action'
PROCESSED_ENCODER_DIR = '/home/classysh/MERLIN/MERLIN/data/211data/encoder/processed_encoder_t'
PROCESSED_ACTION_DIR = '/home/classysh/MERLIN/MERLIN/data/211data/action/processed_action_t'
OUTPUT_DIR = './sanity_results' # 图片保存位置

# ==========================================
# 辅助函数 
# ==========================================
def get_euler_from_quat(w, x, y, z):
    r = R.from_quat([x, y, z, w]) 
    return r.as_euler('xyz', degrees=False)

def transform_frame_z90(x, y, z, rx, ry, rz):
    return np.array([y, -x, z, ry, -rx, rz])

def load_raw_action_transformed(txt_path):
    try:
        raw_data = np.loadtxt(txt_path, delimiter=',', skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7))
    except Exception as e:
        print(f"Error reading {txt_path}: {e}")
        return None

    if raw_data.ndim == 1:
        raw_data = raw_data.reshape(1, -1)

    pos = raw_data[:, 1:4]     
    quat = raw_data[:, 4:8]    

    processed_poses = []
    for i in range(len(pos)):
        curr_euler = get_euler_from_quat(quat[i,0], quat[i,1], quat[i,2], quat[i,3])
        transformed_6d = transform_frame_z90(
            pos[i,0], pos[i,1], pos[i,2],
            curr_euler[0], curr_euler[1], curr_euler[2]
        )
        processed_poses.append(transformed_6d)
    
    processed_poses = np.array(processed_poses)
    processed_poses[:, 3:] = np.unwrap(processed_poses[:, 3:], axis=0)
    return processed_poses

def load_raw_encoder(csv_path):
    range_map_keys = ['CH0-ThumbLow', 'CH1-ThumbUp', 'CH2-Pointer', 
                      'CH3-Middle', 'CH4-Ring', 'CH5-Pinky']
    df = pd.read_csv(csv_path)
    extracted_data = []
    for col_name in range_map_keys:
        if col_name in df.columns:
            raw_vals = df[col_name].values
            if np.isnan(raw_vals).any():
                raw_vals = pd.Series(raw_vals).fillna(method='ffill').fillna(0).values
            extracted_data.append(raw_vals)
    return np.column_stack(extracted_data)

# ==========================================
# 主逻辑
# ==========================================
def run_sanity_check():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 获取文件列表
    raw_enc_files = sorted([f for f in os.listdir(RAW_ENCODER_DIR) if f.endswith('.csv')])
    raw_act_files = sorted([f for f in os.listdir(RAW_ACTION_DIR) if f.endswith('.txt')])
    proc_enc_files = sorted([f for f in os.listdir(PROCESSED_ENCODER_DIR) if f.endswith('.npy')])
    proc_act_files = sorted([f for f in os.listdir(PROCESSED_ACTION_DIR) if f.endswith('.npy')])

    num_samples = min(len(raw_enc_files), len(raw_act_files), len(proc_enc_files), len(proc_act_files))
    if num_samples == 0:
        print("No matched files found!")
        return

    # 2. 随机采样
    # idx = random.randint(0, num_samples - 1)
    idx = 19
    print(f"--- Sanity Check on Sample Index: {idx} ---")
    print(f"Raw Encoder: {raw_enc_files[idx]}")
    print(f"Processed Action: {proc_act_files[idx]}")

    # ==========================================
    # Part 1: Plot Encoder
    # ==========================================
    raw_enc_data = load_raw_encoder(os.path.join(RAW_ENCODER_DIR, raw_enc_files[idx]))
    proc_enc_data = np.load(os.path.join(PROCESSED_ENCODER_DIR, proc_enc_files[idx]))

    finger_names = ['ThumbLow', 'ThumbUp', 'Pointer', 'Middle', 'Ring', 'Pinky']
    
    plt.figure(figsize=(25, 10))
    # don't plot values above 2350 for better visualization
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.plot(raw_enc_data[:, i], label='Raw', alpha=0.5, color='gray')
        plt.plot(proc_enc_data[:, i], label='Processed (LPF)', color='blue', linewidth=1.5)
        #plt.ylim(1900, 2500)
        plt.title(f"{finger_names[i]}")
        plt.legend()
    plt.suptitle(f"Encoder Check (Sample {idx})", fontsize=16)
    plt.tight_layout()
    
    # --- 关键修改 2: 保存图片 ---
    save_path_enc = os.path.join(OUTPUT_DIR, f'sample_{idx}_encoder.png')
    plt.savefig(save_path_enc)
    plt.close() # 释放内存
    print(f"Saved Encoder plot to: {save_path_enc}")

    # ==========================================
    # Part 2: Plot Action
    # ==========================================
    delta_actions = np.load(os.path.join(PROCESSED_ACTION_DIR, proc_act_files[idx]))
    gt_poses = load_raw_action_transformed(os.path.join(RAW_ACTION_DIR, raw_act_files[idx]))

    # 积分重建
    reconstructed_poses = np.zeros_like(delta_actions)
    if len(gt_poses) > 0:
        reconstructed_poses[0] = gt_poses[0] 
        for t in range(1, len(delta_actions)):
            reconstructed_poses[t] = reconstructed_poses[t-1] + delta_actions[t]

    dims = ['X', 'Y', 'Z', 'Rx', 'Ry', 'Rz']
    units = ['m', 'm', 'm', 'rad', 'rad', 'rad']
    
    plt.figure(figsize=(15, 10))
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.plot(gt_poses[:, i], label='Ground Truth', alpha=0.5, color='gray', linewidth=3)
        plt.plot(reconstructed_poses[:, i], label='Reconstructed', color='red', linestyle='--', linewidth=1.5)
        plt.title(f"{dims[i]} ({units[i]})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    plt.suptitle(f"Action Check (Sample {idx})", fontsize=16)
    plt.tight_layout()
    
    # --- 关键修改 3: 保存图片 ---
    save_path_act = os.path.join(OUTPUT_DIR, f'sample_{idx}_action.png')
    plt.savefig(save_path_act)
    plt.close()
    print(f"Saved Action plot to: {save_path_act}")

if __name__ == "__main__":
    run_sanity_check()