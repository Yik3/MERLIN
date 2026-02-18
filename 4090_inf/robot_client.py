import zmq
import numpy as np
import time
import cv2
from Total_API import RobotControlAPI 
# from Safety_Constraint import * # ================= CONFIGURATION =================
# 【重要】请修改为 GPU 电脑的 Ethernet IP 地址
GPU_IP = "192.168.1.100" 
ZMQ_PORT = 5555

ROBOT_IP = "169.254.128.19"
CAMERA_ID = 16

# BASE_POSE (保持原样)
# BASE_POSE = [-0.259968, -0.253127, 0.265704, 1.89, -0.996, -0.185]
BASE_POSE = [-0.298979, -0.303811, 0.263773, 2.023, -0.94, -0.021]

# 手部映射参数
MAX_HAND_VAL = 65535
range_map = {
    'CH0': (1286, 2400),
    'CH1': (1413, 1840),
    'CH2': (1900, 2883),
    'CH3': (1902, 2742),
    'CH4': (1750, 2700),
    'CH5': (1970, 2667) 
}

# ================= HELPER FUNCTIONS =================
def map_encoder_to_motor(encoder_vals, gain=1.0):
    motor_vals = []
    for i in range(6):
        ch_name = f'CH{i}'
        if ch_name in range_map:
            min_val, max_val = range_map[ch_name]
            mapped_val = gain * (encoder_vals[i] - min_val) * MAX_HAND_VAL / (max_val - min_val)
            motor_vals.append(int(mapped_val))
        else:
            motor_vals.append(0) 
    ret_motor_vals = [motor_vals[1], motor_vals[2], motor_vals[3], motor_vals[4], motor_vals[5], motor_vals[0]]
    ret_motor_vals = np.clip(ret_motor_vals, 0, MAX_HAND_VAL).astype(int).tolist()
    return ret_motor_vals

def get_compressed_image(cap):
    """读取相机并进行 JPEG 压缩以便网络传输"""
    ret, frame = cap.read()
    if not ret:
        print("Warning: Failed to read camera")
        return None
    # 编码为 JPG 以减少 Ethernet 延迟 (质量 90 足够)
    ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return buffer.tobytes()

def main():
    # 1. Initialize Robot
    print(f"Connecting to Robot at {ROBOT_IP}...")
    try:
        robot = RobotControlAPI(ROBOT_IP)
        print("Moving to BASE POSE...")
        robot.move_arm_to_pose(BASE_POSE, speed=20, block=True)
        robot.set_hand_position([0]*6)
        print("Robot Ready.")
    except Exception as e:
        print(f"Robot connection failed: {e}")
        return

    # 2. Initialize Camera (Robot Side)
    print(f"Opening Camera {CAMERA_ID}...")
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {CAMERA_ID}")
        return

    # 3. Setup ZMQ (Connect to GPU)
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    print(f"Connecting to GPU Server at {GPU_IP}:{ZMQ_PORT}...")
    socket.connect(f"tcp://{GPU_IP}:{ZMQ_PORT}")
    print("ZMQ Connected.")

    # 4. State Maintenance
    current_pose = np.array(BASE_POSE, dtype=np.float32)
    t_step = 0

    print("\n=== STARTING ETHERNET CONTROL LOOP ===")
    try:
        while True:
            # --- A. Capture & Send Observation ---
            # 1. 获取图片
            img_bytes = get_compressed_image(cap)
            if img_bytes is None:
                break
            
            # 2. 发送图片给 GPU
            # 这里的发送相当于原逻辑中 Inference 循环的开始
            send_packet = {
                'step': t_step,
                'image': img_bytes
            }
            socket.send_pyobj(send_packet)
            
            # --- B. Receive Command (Blocking) ---
            # 等待 GPU 计算完成并传回数据
            # 这相当于原逻辑中的 "Stop and Wait"
            data = socket.recv_pyobj()
            
            raw_delta = np.array(data['delta']) 
            raw_hand_enc = np.array(data['hand'])
            
            # --- C. Process Data ---
            delta_to_apply = raw_delta.copy()
            delta_to_apply[0] *= -1 # X Axis Flip
            delta_to_apply[4] *= -1 # Pitch/Ry Flip
            
            next_target_pose = current_pose + delta_to_apply
            cmd_hand = map_encoder_to_motor(raw_hand_enc)
            
            # --- D. Execute Command (Blocking) ---
            cmd_hand[0] -= 5000 
            if cmd_hand[0] < 0: cmd_hand[0] = 0
            
            robot.set_hand_position(cmd_hand)
            robot.move_arm_to_pose(next_target_pose, speed=20, block=True)
            time.sleep(0.01) # Ensure completion
            
            # --- E. Update Internal State ---
            current_pose = next_target_pose
            
            if t_step % 10 == 0:
                print(f"Step {t_step} Executed | Sent to GPU -> Recv -> Moved")

            t_step += 1

    except KeyboardInterrupt:
        print("\nStopping robot control...")
    finally:
        cap.release()
        socket.close()
        context.term()
        # robot.disconnect() 
        print("Cleaned up.")

if __name__ == "__main__":
    main()