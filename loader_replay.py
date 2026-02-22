import numpy as np
import time
import os
import matplotlib.pyplot as plt
from Total_API import *
from Safety_Constraint import * 
import cv2
# ================= CONFIGURATION =================
# 请替换为你的实际文件路径
ENC_NPY = '/home/rm/Documents/MERLIN/training_data_check/adc_data_20260212013551.npy'
ACT_NPY = '/home/rm/Documents/MERLIN/training_data_check/iphone_data_20260212_013544.npy'
SYNC_NPZ = '/home/rm/Documents/MERLIN/training_data_check/video_recording_realsense#20260212013552_sync.npz'
SAVE_DIR = '/home/rm/Documents/MERLIN/replay_results'
range_map = {
    'CH0': (1286, 2400),
    'CH1': (1413, 1840),
    'CH2': (1900, 2883),
    'CH3': (1902, 2742),
    'CH4': (1750, 2700),
    'CH5': (1970, 2667) 
}
IP = '169.254.128.19'
ENABLE_ROBOT = True  # Set to True to actually move the robot
RECORD_OBSERVATION = True  # Whether to record the camera observation during replay (for visualization)
CAMERA_ID = 20
FPS = 30
BASE_POSE = [-0.298979, -0.303811, 0.263773, 2.023, -0.94, -0.021]
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

def map_encoder_to_motor(encoder_vals, gain = 0.78):
    motor_vals = []
    for i in range(6):
        ch_name = f'CH{i}'
        if ch_name in range_map:
            min_val, max_val = range_map[ch_name]
            # 线性映射
            mapped_val = gain * (encoder_vals[i] - min_val) * MAX_VAL/ (max_val - min_val)
            motor_vals.append(int(mapped_val))
        else:
            motor_vals.append(0)  # Default to 0 if no mapping defined
    ret_motor_vals = [motor_vals[1], motor_vals[2], motor_vals[3], motor_vals[4], motor_vals[5], motor_vals[0]]
    return ret_motor_vals

def main():
    global ENABLE_ROBOT, RECORD_OBSERVATION
    # 1. Load Data
    print("Loading data files...")
    try:
        raw_enc = np.load(ENC_NPY, allow_pickle=True)  # [T_enc, 6]
        raw_act = np.load(ACT_NPY, allow_pickle=True)  # [T_act, 6] (Deltas)
        sync_data = np.load(SYNC_NPZ, allow_pickle=True)

        # 确保 Index 是整数
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
    # 这一步还原出了相对于起点的轨迹 (Relative Pose)
    relative_trajectory = np.cumsum(raw_act, axis=0)

    # 3. Initialize Robot
    if ENABLE_ROBOT:
        print(f"Connecting to robot at {IP}...")
        robot = RobotControlAPI(IP)
        # Move to Home Position first
        print("Moving to BASE_POSE...")
        robot.move_arm_to_pose(BASE_POSE)
        # Reset Hand
        robot.set_hand_position([0]*6)
        time.sleep(2.0)
    else:
        print("Robot Disabled (Simulation Mode)")
    # initialize camera if recording observation
    if RECORD_OBSERVATION:
        print(f"Initializing camera (ID: {CAMERA_ID}) for observation recording...")
        cap = cv2.VideoCapture(CAMERA_ID)
        if not cap.isOpened():
            print(f"Camera ID {CAMERA_ID} is not available. Observation recording disabled.")
            RECORD_OBSERVATION = False
        else:
            # Set resolution and FPS if needed
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, FPS)
    # Recording buffers for visualization
    rec_xyz = []
    rec_euler = []
    rec_hand = []

    print(f"Starting Replay ({len(frame_idxs)} frames)...")
    if RECORD_OBSERVATION:
        # if save dir is not exist, create it
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        out = cv2.VideoWriter(os.path.join(SAVE_DIR, "replay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (640, 480))
    # 4. Main Loop
    for i in range(len(frame_idxs)):
        # --- A. Get Indices from Sync File ---
        # If record, get frame from camera
        
        p_idx = pose_idxs[i]
        e_idx = enc_idxs[i]
        
        # Boundary check (just in case)
        if p_idx >= len(relative_trajectory) or e_idx >= len(final_enc_data):
            print(f"Frame {i}: Index out of bounds, skipping.")
            continue

        # --- B. Compute Target Arm Pose ---
        # Target = Base + Relative_Trajectory[idx]
        rel_pose = relative_trajectory[p_idx]
        rel_pose[0] *= -1
        rel_pose[4] *= -1
        target_pose = [0.0] * 6
        gain = 1.0
        for j in range(6):
            target_pose[j] = BASE_POSE[j] + rel_pose[j] * gain
            
            # Wrap angles to [-pi, pi] if necessary (standard robotics practice)
            if j >= 3:
                while target_pose[j] > np.pi: target_pose[j] -= 2 * np.pi
                while target_pose[j] < -np.pi: target_pose[j] += 2 * np.pi

        # --- C. Compute Target Hand Gesture ---
        # Map encoder values to robot hand command
        # Mapping: Encoder (0-max) -> Robot (0-1000 or similar)
        # 假设你的 Encoder 已经是处理过的数值，这里直接转换成 int
        # 注意：这里需要确认你的 range_map 和 motor_monotonicity 逻辑是否已经包含在 Encoder 处理中
        # 如果 processed_encoder 已经是归一化好的 (0-MAX)，则直接用。
        # 如果是原始值，你需要在这里做线性映射。
        # 假设这里读到的已经是你想发送给电机的值：
        
        raw_hand_vals = final_enc_data[e_idx]
        target_hand = map_encoder_to_motor(raw_hand_vals)

        # --- D. Send Command ---
        if ENABLE_ROBOT and i > 30:
            if RECORD_OBSERVATION:
                ret, frame = cap.read()
                if ret:
                    # Save as mp4 video
                    out.write(frame)
                    # show in real-time
                    cv2.imshow("Replay Observation", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Replay interrupted by user.")
                        break
                else:
                    continue
            # Send Arm
            robot.move_arm_to_pose(target_pose, speed=20)  # Adjust speed as needed
            # Send Hand
            target_hand[0] -= 10000
            robot.set_hand_position(target_hand) # 取消注释以启用手部


        # --- E. Record Data ---
        rec_xyz.append(target_pose[:3])
        rec_euler.append(target_pose[3:])
        rec_hand.append(target_hand)

        if i % 10 == 0:
            print(f"Frame {i}/{len(frame_idxs)} | Pose: {np.round(target_pose, 3)}")
        
        # if ENABLE_ROBOT:
        #     time.sleep(0.01)

    print("Replay Finished.")
    if RECORD_OBSERVATION:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    # 5. Visualization
    rec_xyz = np.array(rec_xyz)
    rec_euler = np.array(rec_euler)
    rec_hand = np.array(rec_hand)
    np.savez("replay_results.npz", xyz=rec_xyz, euler=rec_euler, hand=rec_hand)
    t_steps = np.arange(len(rec_xyz))

    # Plot 1: XYZ
    plt.figure(figsize=(10, 6))
    plt.plot(t_steps, rec_xyz[:, 0], label='X', color='r')
    plt.plot(t_steps, rec_xyz[:, 1], label='Y', color='g')
    plt.plot(t_steps, rec_xyz[:, 2], label='Z', color='b')
    plt.title("Robot End-Effector Trajectory (XYZ)")
    plt.ylabel("Position (m)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot 2: Euler
    plt.figure(figsize=(10, 6))
    plt.plot(t_steps, rec_euler[:, 0], label='Rx', color='c')
    plt.plot(t_steps, rec_euler[:, 1], label='Ry', color='m')
    plt.plot(t_steps, rec_euler[:, 2], label='Rz', color='y')
    plt.title("Robot End-Effector Orientation (Euler)")
    plt.ylabel("Rotation (rad)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot 3: Hand
    plt.figure(figsize=(10, 6))
    finger_names = ['ThumbLow', 'ThumbUp', 'Pointer', 'Middle', 'Ring', 'Pinky']
    for k in range(6):
        plt.plot(t_steps, rec_hand[:, k], label=finger_names[k])
    plt.title("Robot Hand Commands")
    plt.ylabel("Motor Value")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()