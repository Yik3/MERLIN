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
    """
    os.makedirs(sync_dir, exist_ok=True)
    
    # 获取排序后的文件列表
    vids = sorted([f for f in os.listdir(vid_dir) if f.endswith('.mp4')])
    npys = sorted([f for f in os.listdir(npy_dir) if f.endswith('.npy')]) # 视频时间戳
    csvs = sorted([f for f in os.listdir(encoder_dir) if f.endswith('.csv')])
    txts = sorted([f for f in os.listdir(action_dir) if f.endswith('.txt')])
    
    min_len = min(len(vids), len(npys), len(csvs), len(txts))
    print(f"Processing {min_len} sets of data based on sorted file lists.")
    
    for i in range(min_len):
        vid_name = vids[i]
        npy_path = os.path.join(npy_dir, npys[i])
        csv_path = os.path.join(encoder_dir, csvs[i])
        txt_path = os.path.join(action_dir, txts[i])
        
        save_name = os.path.splitext(vid_name)[0] + "_sync.npz"
        save_path = os.path.join(sync_dir, save_name)
        
        print(f"Processing set {i}: {vid_name}...")
        
        # 调用你的同步逻辑
        results = sync_data(npy_path, csv_path, txt_path) 
        
        # ==========================================
        # NEW LOGIC: Drop the first 30 frames
        # ==========================================
        DROP_COUNT = 30
        if results:
            if len(results) > DROP_COUNT:
                results = results[DROP_COUNT:]
                print(f"  -> Dropped first {DROP_COUNT} frames. Remaining: {len(results)}")
            else:
                print(f"  -> Warning: Data length ({len(results)}) is less than drop count ({DROP_COUNT}). Emptying results.")
                results = []
        # ==========================================
        
        # 保存
        save_sync_results(save_path, results)
        
    print(f"Found {len(vids)} videos, {len(npys)} npy files, {len(csvs)} encoder files, {len(txts)} action files.")
if __name__ == "__main__":
    batch_process_data(
        vid_dir="/home/classysh/MERLIN/MERLIN/data/bottle/camera/mp4_files",
        encoder_dir="/home/classysh/MERLIN/MERLIN/data/bottle/encoder",
        action_dir="/home/classysh/MERLIN/MERLIN/data/bottle/action",
        sync_dir="/home/classysh/MERLIN/MERLIN/data/bottle/syncs",
        npy_dir="/home/classysh/MERLIN/MERLIN/data/bottle/camera/npy_files"
    )