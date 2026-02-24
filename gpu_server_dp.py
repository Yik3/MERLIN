import numpy as np
import os
import time
import cv2
import zmq
import sys

# === 添加模块搜索路径 ===
# Diffusion Policy 路径
sys.path.append(os.path.join(os.path.dirname(__file__), "diffusion-policies/shared/env/merlin"))
try:
    from merlin_inference import MerlinPolicyInference
except ImportError:
    print("Warning: Could not import MerlinPolicyInference. Check your paths.")

# ================= CONFIGURATION =================
ZMQ_BIND_ADDR = "tcp://0.0.0.0:5555" 
DP_CKPT_PATH = "weights/diff_policy.ckpt" # 替换为真实的 DP 权重路径

# ================= HELPER FUNCTIONS =================
def process_received_image(img_bytes):
    """解码图片，返回 RGB uint8 数组用于 DP，以及 BGR 用于 OpenCV 渲染"""
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame_bgr is None:
        print("Warning: Failed to decode image from robot")
        frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)

    # DP 需要 RGB
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb, frame_bgr

def main():
    # 1. Setup ZMQ
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.bind(ZMQ_BIND_ADDR)
    print(f"[DP Server] ZMQ PAIR bound to {ZMQ_BIND_ADDR}")

    # 2. Setup Diffusion Policy Model
    print("=== Loading Diffusion Policy ===")
    try:
        dp_policy = MerlinPolicyInference(
            checkpoint_path=DP_CKPT_PATH,
            device="cuda", # 如果没有GPU可以改为 "cuda" if torch.cuda.is_available() else "cpu"
            action_mode="all" 
        )
        print("✅ DP Model Loaded Successfully!")
    except Exception as e:
        print(f"❌ Failed to load DP model: {e}")
        return

    # 初始化状态 (12D: 6 arm + 6 hand)
    dp_policy.reset()
    curr_qpos_raw_np = np.zeros(12, dtype=np.float32)

    print("\n=== DP GPU READY, WAITING FOR IMAGE STREAM ===")
    
    try:
        t_step = 0
        while True:
            data_packet = socket.recv_pyobj() 
            img_bytes = data_packet['image']
            robot_step = data_packet['step']
            
            start_time = time.time()
            
            # --- 处理图像 ---
            dp_img_input, frame_bgr = process_received_image(img_bytes)
            
            # --- Model Inference ---
            # 传入 RGB uint8 image 和 12D numpy state
            dp_output = dp_policy.predict(image=dp_img_input, robot_state=curr_qpos_raw_np)
            
            # 假设输出完整的 12 维 action
            dp_action = dp_output.get("action") 
            if dp_action is None:
                print("Warning: DP returned None action. Using previous state.")
                dp_action = curr_qpos_raw_np

            pred_arm_delta = dp_action[:6]
            pred_hand_abs = dp_action[6:]

            end_time = time.time()
            print(f"Step {t_step}: Inference Time = {end_time - start_time:.3f}s (Pure DP)")

            # --- Send Command ---
            response_packet = {
                'step': t_step,
                'delta': pred_arm_delta.tolist(),
                'hand': pred_hand_abs.tolist()
            }
            socket.send_pyobj(response_packet)

            # --- Update State ---
            # 用刚刚预测出的 action 作为下一步的状态输入
            curr_qpos_raw_np = dp_action.astype(np.float32)
            
            # 可视化
            cv2.putText(frame_bgr, f"DP Step: {t_step}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 2)
            cv2.imshow("GPU Server View", frame_bgr)
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