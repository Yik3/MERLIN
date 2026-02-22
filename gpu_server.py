import torch
import numpy as np
import os
import time
import cv2
import collections
import zmq
from torchvision import transforms
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "train"))
from train.core import build 

# ================= CONFIGURATION =================
# 这里的 IP 设置为 0.0.0.0 表示监听所有网口（包括 Ethernet）
ZMQ_BIND_ADDR = "tcp://0.0.0.0:5555" 

CKPT_PATH = "weights/policy_last_218.pth" 
NORM_STATS_PATH = "weights/normalization_stats_12d_218.npz"
NUM_QUERIES = 70 
STATE_DIM = 12 
TEMPORAL_AGGREGATION = True
K_AGGREGATION = 70

# ================= MODEL ARGS (保持原样) =================
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
def process_received_image(img_bytes, transform, device):
    """
    将接收到的字节流解码为 Tensor
    """
    # 1. 解码图片
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        print("Warning: Failed to decode image from robot")
        return torch.zeros(1, 1, 3, 480, 640).to(device), np.zeros((480, 640, 3), dtype=np.uint8)

    # 2. 转换格式 (BGR -> RGB)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 3. Transform & Unsqueeze
    img_tensor = transform(img)
    return img_tensor.unsqueeze(0).unsqueeze(0).to(device), frame

def main():
    # 1. Setup ZMQ (Server Side)
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.bind(ZMQ_BIND_ADDR)
    print(f"[GPU Server] ZMQ PAIR bound to {ZMQ_BIND_ADDR}")
    print("[GPU Server] Waiting for Robot connection...")

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

    # 3. Setup Transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    # 4. Initialize State
    raw_arm_delta_init = torch.zeros(6).to(device)
    raw_hand_abs_init = torch.zeros(6).float().to(device)
    raw_qpos_init = torch.cat([raw_arm_delta_init, raw_hand_abs_init])
    curr_qpos_norm = (raw_qpos_init - qpos_mean) / qpos_std
    
    past_predictions_buffer = collections.deque(maxlen=K_AGGREGATION)
    exp_weight_k = 0.05

    print("\n=== GPU READY, WAITING FOR IMAGE STREAM ===")
    
    try:
        t_step = 0
        while True:
            # --- A. Receive Image from Robot (Blocking) ---
            # 机器人每执行完一次动作，会采集新图片发过来
            # 这相当于原逻辑中的 "Wait" + "Get Observation"
            data_packet = socket.recv_pyobj() 
            
            img_bytes = data_packet['image']
            robot_step = data_packet['step']
            
            start_time = time.time()
            
            # 处理图像
            image_input, frame = process_received_image(img_bytes, transform, device)
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

            # --- D. Post-Process ---
            curr_action_norm_tensor = torch.from_numpy(curr_action_norm_np).float().to(device)
            raw_action = (curr_action_norm_tensor * qpos_std) + qpos_mean
            raw_action_np = raw_action.cpu().numpy()
            
            pred_arm_delta = raw_action_np[:6]
            pred_hand_abs = raw_action_np[6:]
            
            end_time = time.time()
            print(f"Step {t_step} (Robot Step {robot_step}): Inference Time = {end_time - start_time:.3f}s")

            # --- E. Send Command back to Robot ---
            response_packet = {
                'step': t_step,
                'delta': pred_arm_delta.tolist(),
                'hand': pred_hand_abs.tolist()
            }
            socket.send_pyobj(response_packet)

            # --- F. Update Model State ---
            # 更新状态用于下一次推理
            curr_qpos_norm = curr_action_norm_tensor
            
            # Visualization (On GPU Server)
            cv2.putText(frame, f"Server Step: {t_step}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("GPU Server View", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            t_step += 1

    except KeyboardInterrupt:
        print("\nStopping GPU server...")
    finally:
        cv2.destroyAllWindows()
        socket.close()
        context.term()

if __name__ == "__main__":
    main()