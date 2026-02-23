import zmq
import numpy as np
import time
import cv2
import threading
from Total_API import RobotControlAPI 

# ================= CONFIGURATION =================
GPU_IP = "192.168.1.100" 
ACT_PORT = 5555  # ACT 端口
DP_PORT = 5556   # DP 端口

ROBOT_IP = "169.254.128.19"
CAMERA_ID = 20

BASE_POSE = [-0.142258,-0.287446,0.250924,2.78,-1.105,-1.107]
MAX_HAND_VAL = 65535
range_map = {
    'CH0': (1740, 2750), 
    'CH1': (2050, 2500),
    'CH2': (1100, 2037),
    'CH3': (1235, 2060),
    'CH4': (1235, 2200),
    'CH5': (1240, 2130) 
}
monotonicity_map = {
    'CH0': 0,
    'CH1': 1,
    'CH2': 0,
    'CH3': 0,
    'CH4': 0,
    'CH5': 0
}
# 全局变量与锁
global_img_bytes = None
camera_lock = threading.Lock()
robot_lock = threading.Lock()  # 防止同时发指令给机器人导致崩溃
is_running = True

# ================= HELPER FUNCTIONS =================
def map_encoder_to_motor(encoder_vals, gain = 1.0):
    motor_vals = []
    for i in range(6):
        ch_name = f'CH{i}'
        if ch_name in range_map:
            min_val, max_val = range_map[ch_name]
            # Linear Mapping with Monotonicity Consideration
            if monotonivity_map[ch_name] == 0:  # Monotonically Decreasing
                encoder_vals[i] = max(min(encoder_vals[i], max_val), min_val)  # Clip to range
                mapped_val = gain * (max_val - encoder_vals[i]) * MAX_VAL / (max_val - min_val)
            else:  # Monotonically Increasing
                encoder_vals[i] = max(min(encoder_vals[i], max_val), min_val)  # Clip to range
                mapped_val = gain * (encoder_vals[i] - min_val) * MAX_VAL / (max_val - min_val)
            motor_vals.append(int(mapped_val))
        else:
            motor_vals.append(0)  # Default to 0 if no mapping defined
    ret_motor_vals = [motor_vals[1], motor_vals[2], motor_vals[4], motor_vals[3], motor_vals[5], motor_vals[0]]
    return ret_motor_vals

def get_compressed_image(frame):
    ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return buffer.tobytes()

global_12d_state = np.zeros(12, dtype=np.float32)  # [x, y, z, roll, pitch, yaw, hand1, hand2, hand3, hand4, hand5, hand6]
state_lock = threading.Lock()
# ================= THREADS =================
def act_hand_thread(robot):
    """高频线程 (~50Hz): 负责与 ACT 通讯并控制手部"""
    global is_running, global_img_bytes, global_12d_state

    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.connect(f"tcp://{GPU_IP}:{ACT_PORT}")
    
    t_step = 0
    while is_running:
        with camera_lock:
            img = global_img_bytes
            
        if img is None:
            time.sleep(0.01)
            continue
        
        with state_lock:
            current_state = global_12d_state.copy()
        socket.send_pyobj({'step': t_step, 'image': img, 'state': current_state})
        print(f"[ACT Hand] Step {t_step} Sent Image and State, waiting for hand encoders...")
        data = socket.recv_pyobj() # 阻塞等待 0.02s
        print(f"[ACT Hand] Step {t_step} Received Hand Encoders")
        raw_hand_enc = np.array(data['hand'])
        with state_lock:
            global_12d_state[6:12] = raw_hand_enc
        cmd_hand = map_encoder_to_motor(raw_hand_enc)
        # 加锁执行
        with robot_lock:
            robot.set_hand_position(cmd_hand)
            
        t_step += 1

def dp_arm_thread(robot):
    """低频线程 (~1Hz): 负责与 DP 通讯并控制机械臂"""
    global is_running, global_img_bytes, global_12d_state
    
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.connect(f"tcp://{GPU_IP}:{DP_PORT}")
    
    current_pose = np.array(BASE_POSE, dtype=np.float32)
    t_step = 0
    
    while is_running:
        with camera_lock:
            img = global_img_bytes
            
        if img is None:
            time.sleep(0.01)
            continue
        
        with state_lock:
            current_state = global_12d_state.copy()
        # 发送图片，服务端的高频 Buffer 会处理它
        socket.send_pyobj({'step': t_step, 'image': img, 'state': current_state})
        print(f"[DP Arm] Step {t_step} Sent Image, waiting for delta...")
        data = socket.recv_pyobj() # 阻塞等待 1s
        print(f"[DP Arm] Step {t_step} Received Delta")
        raw_delta = np.array(data['delta']) 
        with state_lock:
            global_12d_state[:6] = raw_delta
        delta_to_apply = raw_delta.copy()
        # delta_to_apply[0] *= -1 
        # delta_to_apply[4] *= -1 
        
        next_target_pose = current_pose + delta_to_apply
        
        # 加锁执行
        with robot_lock:
            robot.move_arm_to_pose(next_target_pose, speed=20, block=True)
            
        current_pose = next_target_pose
        print(f"[DP Arm] Step {t_step} Moved")
        t_step += 1

# ================= MAIN =================
def main():
    global is_running, global_img_bytes
    
    robot = RobotControlAPI(ROBOT_IP)
    robot.move_arm_to_pose(BASE_POSE, speed=20, block=True)
    start_hand = [0, 0, 0, 0, 0, MAX_HAND_VAL-10000]
    robot.set_hand_position(start_hand)

    cap = cv2.VideoCapture(CAMERA_ID)
    
    # 启动双线程
    t_act = threading.Thread(target=act_hand_thread, args=(robot,))
    t_dp = threading.Thread(target=dp_arm_thread, args=(robot,))
    t_act.start()
    t_dp.start()

    print("\n=== STARTING THREADED CONTROL LOOP ===")
    try:
        # 主线程只负责疯狂读摄像头，更新全局变量
        while True:
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1) # 水平翻转
                frame = cv2.flip(frame, 0) # 垂直翻转
                compressed = get_compressed_image(frame)
                with camera_lock:
                    global_img_bytes = compressed
            time.sleep(0.01) # 限制在 100fps 左右，避免吃满 CPU
            
    except KeyboardInterrupt:
        print("\nStopping...")
        is_running = False
        t_act.join()
        t_dp.join()
        cap.release()

if __name__ == "__main__":
    main()