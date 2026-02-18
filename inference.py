import torch
import numpy as np
import os
import time
import cv2
import collections
import zmq
from torchvision import transforms
import sys

# 路径设置
sys.path.append(os.path.join(os.path.dirname(__file__), "train"))
from train.core import build 

# ================= CONFIGURATION =================
CKPT_PATH = "weights/policy_last_218.pth" 
NORM_STATS_PATH = "weights/normalization_stats_12d_218.npz"
CAMERA_ID = 16
NUM_QUERIES = 70 
STATE_DIM = 12 
TEMPORAL_AGGREGATION = True
K_AGGREGATION = 70
ZMQ_PORT = 5555

# ================= MODEL ARGS =================
class ModelArgs:
    def __init__(self):
        self.num_queries = NUM_QUERIES
        self.camera_names = ["cam_high"]
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

# ================= HELPER FUNCTIONS =================
def get_image(cap, transform, device):
    ret, frame = cap.read()
    if not ret:
        print("Warning: Failed to read camera")
        return torch.zeros(1, 1, 3, 480, 640).to(device), np.zeros((480, 640, 3), dtype=np.uint8)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img)
    return img_tensor.unsqueeze(0).unsqueeze(0).to(device), frame

def main():
    # 1. Setup ZMQ (PAIR Pattern for synchronous lock-step)
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.bind(f"tcp://*:{ZMQ_PORT}")
    print(f"[Inference] ZMQ PAIR bound to port {ZMQ_PORT}")

    # 2. Setup Device & Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"Loading stats from {NORM_STATS_PATH}...")
    stats = np.load(NORM_STATS_PATH)
    qpos_mean = torch.from_numpy(stats['qpos_mean']).float().to(device)
    qpos_std = torch.from_numpy(stats['qpos_std']).float().to(device)
    
    print(f"Loading model from {CKPT_PATH}...")
    args = ModelArgs()
    model = build(args)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.to(device)
    model.eval()

    # 3. Initialize Camera
    cap = cv2.VideoCapture(CAMERA_ID)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    # 4. Initialize State
    # 初始状态设为0，符合 ACT 训练时的相对坐标逻辑
    raw_arm_delta_init = torch.zeros(6).to(device)
    raw_hand_abs_init = torch.zeros(6).float().to(device) # 假设初始手部也是0或者特定值
    raw_qpos_init = torch.cat([raw_arm_delta_init, raw_hand_abs_init])
    
    curr_qpos_norm = (raw_qpos_init - qpos_mean) / qpos_std
    
    past_predictions_buffer = collections.deque(maxlen=K_AGGREGATION)
    exp_weight_k = 0.05

    print("\n=== STARTING SYNCHRONOUS INFERENCE LOOP ===")
    
    try:
        t_step = 0
        while True:
            # --- A. Get Observation ---
            # 注意：在 Stop-and-Wait 模式下，这里的图像是机器人动作完成后的新图像
            start_time = time.time()
            image_input, frame = get_image(cap, transform, device)
            qpos_input = curr_qpos_norm.unsqueeze(0)
            
            # --- B. Model Inference ---
            with torch.no_grad():
                all_actions = model(qpos_input, image_input, None) 
            
            pred_cpu = all_actions[0][0].cpu().numpy()
            past_predictions_buffer.append(pred_cpu)

            # --- C. Temporal Aggregation ---
            if TEMPORAL_AGGREGATION:
                action_weighted_sum = np.zeros(STATE_DIM)
                weight_sum = 0.0
                for i in range(len(past_predictions_buffer)):
                    past_pred = past_predictions_buffer[-(i+1)]
                    if i < past_pred.shape[0]:
                        weight = np.exp(-exp_weight_k * i)
                        action_weighted_sum += past_pred[i] * weight
                        weight_sum += weight
                curr_action_norm_np = action_weighted_sum / weight_sum
            else:
                curr_action_norm_np = pred_cpu[0]

            # --- D. Post-Process (Denormalize Only) ---
            curr_action_norm_tensor = torch.from_numpy(curr_action_norm_np).float().to(device)
            raw_action = (curr_action_norm_tensor * qpos_std) + qpos_mean
            raw_action_np = raw_action.cpu().numpy()
            
            # 提取原始数据 (Delta 和 Hand Abs)
            pred_arm_delta = raw_action_np[:6]
            pred_hand_abs = raw_action_np[6:]
            end_time = time.time()
            print(f"Step {t_step}: Inference Time = {end_time - start_time:.3f} seconds")
            # --- E. Send Command & WAIT ---
            data_packet = {
                'step': t_step,
                'delta': pred_arm_delta.tolist(),     # 原始 Delta
                'hand': pred_hand_abs.tolist()        # 原始 Hand
            }
            
            # 1. 发送数据
            socket.send_pyobj(data_packet)
            
            # 2. 【关键】阻塞等待机器人完成 (Stop Inference)
            print(f"Step {t_step}: Waiting for robot execution...")
            ack = socket.recv_string() # 此时程序会卡在这里，直到收到 "DONE"
            print(f"Step {t_step}: Robot finished. Continuing.")

            # --- F. Update Model State ---
            curr_qpos_norm = curr_action_norm_tensor
            
            # Visualization
            cv2.putText(frame, f"Step: {t_step} (Synced)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Robot Camera", frame) 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            t_step += 1

    except KeyboardInterrupt:
        print("\nStopping inference...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        socket.close()
        context.term()

if __name__ == "__main__":
    main()