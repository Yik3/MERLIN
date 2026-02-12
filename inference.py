import torch
import numpy as np
import os
import time
import cv2
import argparse
import collections
from torchvision import transforms

# 引入 Robot API 和 Core
from Total_API import RobotControlAPI, BASE_POSE
from train.core import build 

# ===========================================================================
# CONFIGURATION
# ===========================================================================
# 模型与统计数据
CKPT_PATH = "weights/policy_last.pth" 
NORM_STATS_PATH = "normalization_stats_12d.npz"

# 硬件参数
ROBOT_IP = "169.254.128.19"
CAMERA_ID = 0  # 根据实际情况修改
INFERENCE_FPS = 15 

# ACT 参数
NUM_QUERIES = 70 
STATE_DIM = 12 # 6 (Arm Delta) + 6 (Hand Abs)
CAMERA_NAMES = ["cam_high"]
TEMPORAL_AGGREGATION = True
K_AGGREGATION = 70 # 聚合视野，通常等于 NUM_QUERIES

# ===========================================================================
# MODEL ARGS (必须与训练一致)
# ===========================================================================
class ModelArgs:
    def __init__(self):
        self.num_queries = NUM_QUERIES
        self.camera_names = CAMERA_NAMES
        self.state_dim = STATE_DIM
        self.hidden_dim = 512
        self.dropout = 0.1
        self.nheads = 8
        self.dim_feedforward = 3200
        self.enc_layers = 4
        self.dec_layers = 7
        self.pre_norm = False
        self.position_embedding = 'sine'
        self.backbone = 'resnet18'
        self.lr_backbone = 1e-5
        self.masks = False
        self.dilation = False

# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================
def get_image(cap, transform, device):
    ret, frame = cap.read()
    if not ret:
        print("Warning: Failed to read camera")
        return torch.zeros(1, 1, 3, 480, 640).to(device) # Return black if failed
    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img) # [3, H, W]
    return img_tensor.unsqueeze(0).unsqueeze(0).to(device) # [1, 1, 3, H, W]

def main():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Normalization Stats
    print(f"Loading stats from {NORM_STATS_PATH}...")
    stats = np.load(NORM_STATS_PATH)
    qpos_mean = torch.from_numpy(stats['qpos_mean']).float().to(device)
    qpos_std = torch.from_numpy(stats['qpos_std']).float().to(device)
    
    # 3. Load Model
    print(f"Loading model from {CKPT_PATH}...")
    args = ModelArgs()
    model = build(args)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.to(device)
    model.eval()

    # 4. Initialize Hardware
    print("Initializing Robot...")
    robot = RobotControlAPI(ROBOT_IP)
    
    print("Moving to BASE POSE...")
    robot.move_arm_to_pose(BASE_POSE, speed=20, block=True)
    
    init_hand = [0, 0, 0, 0, 0, 0]
    robot.set_hand_position(init_hand)
    
    print("Opening Camera...")
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {CAMERA_ID}")

    # Image Transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    # 5. Initialize State & Buffers
    # qpos_norm_input: [1, 12] Tensor
    # 初始状态: Arm Delta = 0, Hand = init_hand
    raw_arm_delta_init = torch.zeros(6).to(device)
    raw_hand_abs_init = torch.tensor(init_hand).float().to(device)
    raw_qpos_init = torch.cat([raw_arm_delta_init, raw_hand_abs_init])
    
    # 当前归一化的状态输入 (Auto-regressive input)
    curr_qpos_norm = (raw_qpos_init - qpos_mean) / qpos_std
    
    # 维护当前的 Arm Target Pose (用于积分)
    curr_target_pose = np.array(BASE_POSE, dtype=np.float32)

    # Temporal Aggregation Buffer
    # 存储过去 K 次推理的完整预测结果
    past_predictions_buffer = collections.deque(maxlen=K_AGGREGATION)
    
    # Temporal Aggregation 参数 (k=0.01 from ACT paper)
    exp_weight_k = 0.01 

    print("\n=== STARTING INFERENCE LOOP (Press Ctrl+C to stop) ===")
    
    try:
        t_step = 0
        while True:
            loop_start = time.time()

            # --- A. Get Observation ---
            image_input = get_image(cap, transform, device)
            qpos_input = curr_qpos_norm.unsqueeze(0) # [1, 12]

            # --- B. Model Inference ---
            with torch.no_grad():
                # all_actions: [1, NUM_QUERIES, 12]
                all_actions = model(qpos_input, image_input, None) 
            
            # 转为 Numpy 存入 Buffer: [NUM_QUERIES, 12]
            pred_cpu = all_actions[0].cpu().numpy()
            past_predictions_buffer.append(pred_cpu)

            # --- C. Temporal Aggregation ---
            if TEMPORAL_AGGREGATION:
                # 初始化加权和
                action_weighted_sum = np.zeros(STATE_DIM)
                weight_sum = 0.0
                
                # 遍历 buffer 中存储的过去预测
                # i=0: 最新的预测 (time=t)
                # i=1: 上一步的预测 (time=t-1)
                for i in range(len(past_predictions_buffer)):
                    # 取出第 i 个历史预测序列
                    # buffer[-1] 是最新, buffer[-(i+1)] 是往前推 i 步
                    past_pred = past_predictions_buffer[-(i+1)]
                    
                    # 我们需要的是当前时间步的动作
                    # 第 i 个历史预测是在 t-i 时刻生成的
                    # 在那个序列中，第 i 个元素对应的是时间 t
                    if i < past_pred.shape[0]:
                        weight = np.exp(-exp_weight_k * i)
                        action_weighted_sum += past_pred[i] * weight
                        weight_sum += weight
                
                # 计算加权平均作为最终动作 (Normalized)
                curr_action_norm_np = action_weighted_sum / weight_sum
                
            else:
                # 不使用聚合，直接取第一个预测
                curr_action_norm_np = pred_cpu[0]

            # --- D. Post-Process Action ---
            # 1. 转回 Tensor 进行反归一化方便计算
            curr_action_norm_tensor = torch.from_numpy(curr_action_norm_np).float().to(device)
            raw_action = (curr_action_norm_tensor * qpos_std) + qpos_mean
            
            # 2. 提取指令
            raw_action_np = raw_action.cpu().numpy()
            pred_arm_delta = raw_action_np[:6] # [x, y, z, rx, ry, rz]
            pred_hand_abs = raw_action_np[6:]  # [6 motors]

            # 3. 计算 Arm Target (Integration)
            next_target_pose = curr_target_pose + pred_arm_delta
            
            # 4. 计算 Hand Target (Clipping)
            MAX_HAND_VAL = 3000 # 根据实际情况调整
            cmd_hand = np.clip(pred_hand_abs, 0, MAX_HAND_VAL).astype(int)

            # --- E. Execute Command ---
            # 发送 Arm 指令
            robot.move_arm_to_pose(next_target_pose, speed=100, block=False)
            
            # 发送 Hand 指令
            robot.set_hand_position(cmd_hand)

            # --- F. Update State for Next Step ---
            # ACT 的自回归特性：将当前执行的动作（归一化后）作为下一步的输入状态
            curr_qpos_norm = curr_action_norm_tensor
            
            # 更新积分器基准
            curr_target_pose = next_target_pose

            # --- G. Timing Control ---
            t_step += 1
            elapsed = time.time() - loop_start
            wait_time = (1.0 / INFERENCE_FPS) - elapsed
            if wait_time > 0:
                time.sleep(wait_time)
            
            # Print status every 10 steps
            if t_step % 10 == 0:
                print(f"Step {t_step} | Delta: {np.round(pred_arm_delta[:3], 4)} | Hand: {cmd_hand[0]}")

    except KeyboardInterrupt:
        print("\nStopping inference...")
    
    finally:
        cap.release()
        robot.disconnect()
        print("Hardware disconnected.")

if __name__ == "__main__":
    main()