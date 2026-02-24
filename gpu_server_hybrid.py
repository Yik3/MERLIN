# RUN WITH ACT TOGETHER!!!

import numpy as np
import os
import time
import cv2
import zmq
import threading
from collections import deque
import sys

# 导入你的 DP inference
sys.path.append(os.path.join(os.path.dirname(__file__), "diffusion-policies/shared/env/merlin"))
from merlin_inference import MerlinPolicyInference

# ================= CONFIGURATION =================
ZMQ_BIND_ADDR = "tcp://0.0.0.0:5556"  # 注意：这里改成了 5556，与 ACT 的 5555 错开
DP_CKPT_PATH = "weights/diff_policy.ckpt" 

# ================= BUFFER CLASS =================
class LatestImageBuffer:
    def __init__(self):
        # 长度为 2 的双端队列，满时自动从左侧扔掉旧数据
        self.buffer = deque(maxlen=2)
        self.lock = threading.Lock()
        self.stall = False

    def push(self, data):
        """高频接收端调用：推入最新图片"""
        with self.lock:
            # 如果推理端正在 stall 提取图片，跳过此次更新
            if not self.stall:
                self.buffer.append(data)

    def pop_latest(self):
        """低频推理端调用：stall 缓冲区并取出最新图片"""
        with self.lock:
            self.stall = True  # 锁定，停止更新
            
            if len(self.buffer) == 0:
                self.stall = False
                return None
                
            # 取出最新的数据 (最右侧)
            latest_data = self.buffer[-1]
            self.stall = False
            return latest_data

# ================= SERVER THREADS =================
def zmq_receive_thread(socket, img_buffer):
    """后台线程：疯狂接收机器人发来的图片，塞入 Buffer"""
    while True:
        data_packet = socket.recv_pyobj()
        img_buffer.push(data_packet)

def main():
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.bind(ZMQ_BIND_ADDR)
    print(f"[DP Server] ZMQ PAIR bound to {ZMQ_BIND_ADDR}")

    dp_policy = MerlinPolicyInference(checkpoint_path=DP_CKPT_PATH, device="cuda", action_mode="all")
    dp_policy.reset()

    # 初始化 Buffer 并启动接收线程
    img_buffer = LatestImageBuffer()
    recv_thread = threading.Thread(target=zmq_receive_thread, args=(socket, img_buffer), daemon=True)
    recv_thread.start()

    curr_qpos_raw_np = np.zeros(12, dtype=np.float32)
    t_step = 0
    
    print("\n=== DP GPU READY (BUFFER MODE), WAITING FOR IMAGES ===")
    
    try:
        while True:
            # 1. Stall update 并获取最新图片
            latest_packet = img_buffer.pop_latest()
            if latest_packet is None:
                time.sleep(0.01)
                continue
                
            start_time = time.time()
            img_bytes = latest_packet['image']
            robot_step = latest_packet['step']
            
            # 2. 解码图片
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # 3. DP Inference (耗时 ~1s)
            dp_output = dp_policy.predict(image=img_rgb, robot_state=curr_qpos_raw_np)
            dp_action = dp_output.get("action") 
            
            pred_arm_delta = dp_action[:6]
            pred_hand_abs = dp_action[6:]
            curr_qpos_raw_np = dp_action.astype(np.float32)

            end_time = time.time()
            print(f"DP Inference Time: {end_time - start_time:.3f}s | Robot Sent Step: {robot_step}")

            # 4. 发送结果回机器人
            response_packet = {
                'step': t_step,
                'delta': pred_arm_delta.tolist(),
                'hand': pred_hand_abs.tolist() # 虽然机器人端可能不用这个 hand 数据，但传回去保持结构一致
            }
            # 注意：在 PAIR 模式下，接收线程一直在 recv，发送线程在这里 send 是安全的
            socket.send_pyobj(response_packet)
            
            t_step += 1

    except KeyboardInterrupt:
        print("\nStopping DP server...")

if __name__ == "__main__":
    main()