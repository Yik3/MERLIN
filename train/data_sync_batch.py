import os
import numpy as np
import pandas as pd
# 引入你原本的 sync_data 函数 (保持不变)
from data_sync import sync_data 

def save_sync_results(save_path, sync_results):
    """
    将 sync_data 的结果保存为 .npz 文件
    sync_results: list of tuples (frame_idx, encoder_avg, pose_val)
    """
    if not sync_results:
        print(f"Warning: No data to save for {save_path}")
        return

    frame_indices = np.array([item[0] for item in sync_results], dtype=np.int32)
    qpos_idx = np.array([item[1] for item in sync_results], dtype=np.float32)
    actions = np.array([item[2] for item in sync_results], dtype=np.float32)

    # 保存为压缩的 numpy 文件
    np.savez_compressed(
        save_path, 
        frame_idx=frame_indices, 
        encoder_idx=qpos_idx, 
        pose_idx=actions
    )
    print(f"Saved sync data to {save_path} | Frames: {len(frame_indices)}")

def batch_process_data(vid_dir, encoder_dir, action_dir, sync_dir, npy_dir):
    """
    遍历目录，匹配文件，执行同步并保存。
    假设文件名中有时间戳可以用于匹配 (这里简化逻辑，假设文件名排序后是一一对应的，你需要根据实际文件名匹配逻辑修改)
    """
    os.makedirs(sync_dir, exist_ok=True)
    
    # 获取排序后的文件列表
    vids = sorted([f for f in os.listdir(vid_dir) if f.endswith('.mp4')])
    npys = sorted([f for f in os.listdir(npy_dir) if f.endswith('.npy')]) # 视频时间戳
    csvs = sorted([f for f in os.listdir(encoder_dir) if f.endswith('.csv')])
    txts = sorted([f for f in os.listdir(action_dir) if f.endswith('.txt')])
    
    # 简单的索引对应检查 (实际应用中最好用文件名解析时间戳来这就)
    min_len = min(len(vids), len(npys), len(csvs), len(txts))
    
    for i in range(min_len):
        vid_name = vids[i]
        # 构造完整路径
        npy_path = os.path.join(npy_dir, npys[i])
        csv_path = os.path.join(encoder_dir, csvs[i])
        txt_path = os.path.join(action_dir, txts[i])
        
        # 定义输出路径
        save_name = os.path.splitext(vid_name)[0] + "_sync.npz"
        save_path = os.path.join(sync_dir, save_name)
        
        print(f"Processing set {i}: {vid_name}...")
        
        # 调用你的同步逻辑
        # 注意：你需要把 sync_data 函数放到这里或 import 进来
        results = sync_data(npy_path, csv_path, txt_path) 
        
        # 保存
        save_sync_results(save_path, results)

if __name__ == "__main__":
    batch_process_data(
        vid_dir="/home/classysh/MERLIN/MERLIN/data/210data/camera/mp4_files",
        encoder_dir="/home/classysh/MERLIN/MERLIN/data/210data/encoder",
        action_dir="/home/classysh/MERLIN/MERLIN/data/210data/action",
        sync_dir="/home/classysh/MERLIN/MERLIN/data/210data/syncs",
        npy_dir="/home/classysh/MERLIN/MERLIN/data/210data/camera/npy_files"
    )