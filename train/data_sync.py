import numpy as np
import pandas as pd

def find_nearest_index(array, value):
    """
    辅助函数：在有序数组 array 中找到最接近 value 的索引。
    返回 (index, difference)
    """
    idx = np.searchsorted(array, value, side="left")
    
    # 边界情况处理
    if idx == 0:
        return 0, abs(array[0] - value)
    if idx == len(array):
        return len(array) - 1, abs(array[-1] - value)
    
    # 比较 idx 和 idx-1 哪个更近
    before = idx - 1
    after = idx
    
    diff_before = abs(array[before] - value)
    diff_after = abs(array[after] - value)
    
    if diff_before < diff_after:
        return before, diff_before
    else:
        return after, diff_after

def sync_data(npy_path, csv_path, txt_path):
    """
    以 NPY (Video) 为主轴，寻找 CSV 和 TXT 中最近的数据。
    只有当时间差超过 MAX_DIFF 时才丢弃，否则全部保留。
    """
    
    # --- 1. 参数设置 ---
    MAX_DIFF_SEC = 0.07
    MAX_DIFF_NS = MAX_DIFF_SEC * 1e9  # 1000 ms = 1 s = 1e9 ns
    WINDOW_SIZE = 30
    HALF_WINDOW = WINDOW_SIZE // 2

    # --- 2. 读取数据 ---
    print(f"Loading Video Timestamps (NPY) from {npy_path}...")
    try:
        master_times = np.load(npy_path)
    except Exception as e:
        print(f"Error loading npy file: {e}")
        return []

    print(f"Loading Encoder Data (CSV) from {csv_path}...")
    df_csv = pd.read_csv(csv_path)
    # 确保按时间排序
    df_csv = df_csv.sort_values('Time_ns').reset_index(drop=True)
    csv_times = df_csv['Time_ns'].values
    # 提取 Sensor 数据列 (CH0 - CH5)
    # 假设列名包含 'CH'
    sensor_cols = [c for c in df_csv.columns if 'CH' in c]
    csv_data = df_csv[sensor_cols].values
    
    print(f"Loading Pose Data (TXT) from {txt_path}...")
    df_txt = pd.read_csv(txt_path)
    df_txt = df_txt.sort_values('timestamp').reset_index(drop=True)
    txt_times = df_txt['timestamp'].values
    # 提取 7D Pose 数据
    pose_cols = ['pos_x', 'pos_y', 'pos_z', 'quat_w', 'quat_x', 'quat_y', 'quat_z']
    txt_data = df_txt[pose_cols].values

    results = []
    dropped_count = 0
    
    print(f"Starting Synchronization for {len(master_times)} video frames...")

    # --- 3. 遍历每一个 Video Frame ---
    for i_cam, t_vid in enumerate(master_times):
        
        # === A. 找 CSV (Encoder) 最近点 ===
        idx_csv, diff_csv = find_nearest_index(csv_times, t_vid)
        
        # === B. 找 TXT (Pose) 最近点 ===
        idx_txt, diff_txt = find_nearest_index(txt_times, t_vid)
        
        # === C. 检查时间差是否太大 ===
        # 只要都在允许范围内，就收录
        if diff_csv <= MAX_DIFF_NS and diff_txt <= MAX_DIFF_NS:
            
            # 1. 处理 Encoder：取窗口平均
            # 窗口范围：[idx - 15, idx + 15]
            start = max(0, idx_csv - HALF_WINDOW)
            end = min(len(csv_data), idx_csv + HALF_WINDOW)
            
            if start < end:
                encoder_avg = np.mean(csv_data[start:end], axis=0)
            else:
                encoder_avg = csv_data[idx_csv] # 极其罕见的边界fallback
            
            # 2. 处理 Pose：直接取最近点
            pose_val = txt_data[idx_txt]
            
            # 3. 存入结果
            results.append((i_cam, idx_csv, idx_txt))
            
        else:
            dropped_count += 1
            # 可选：打印为什么丢弃 (方便调试)
        
            # print(f"Frame {i_cam} dropped. Diff CSV: {diff_csv/1e9:.4f}s, Diff TXT: {diff_txt/1e9:.4f}s")

    print(f"Sync Complete.")
    print(f"Total Video Frames: {len(master_times)}")
    print(f"Matched Frames:     {len(results)}")
    print(f"Dropped Frames:     {dropped_count}")
    
    return results

# --- 测试代码 ---
if __name__ == "__main__":
    # 请替换为你实际的路径
    npy_path = '/home/classysh/MERLIN/MERLIN/data/210data/camera/video_recording_realsense#20260210234343.npy' 
    csv_path = '/home/classysh/MERLIN/MERLIN/data/210data/encoder/adc_data_20260210234340.csv'
    txt_path = '/home/classysh/MERLIN/MERLIN/data/210data/action/iphone_data_20260210_234313.txt'
    
    data = sync_data(npy_path, csv_path, txt_path)
    # 打印前5条结果示例
    for i, (frame_idx, encoder_avg, pose_val) in enumerate(data[:5]):
        print(f"Frame {frame_idx}: Encoder Avg: {encoder_avg}, Pose: {pose_val}")