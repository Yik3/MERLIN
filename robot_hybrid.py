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
CAMERA_ID = 16

BASE_POSE = [-0.298979, -0.303811, 0.263773, 2.023, -0.94, -0.021]
MAX_HAND_VAL = 65535
range_map = {
    'CH0': (1286, 2400), 'CH1': (1413, 1840), 'CH2': (1900, 2883),
    'CH3': (1902, 2742), 'CH4': (1750, 2700), 'CH5': (1970, 2667) 
}

# 全局变量与锁
global_img_bytes = None
camera_lock = threading.Lock()
robot_lock = threading.Lock()  # 防止同时发指令给机器人导致崩溃
is_running = True

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
    return np.clip(ret_motor_vals, 0, MAX_HAND_VAL).astype(int).tolist()

def get_compressed_image(frame):
    ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return buffer.tobytes()

# ================= THREADS =================
def act_hand_thread(robot):
    """高频线程 (~50Hz): 负责与 ACT 通讯并控制手部"""
    global is_running, global_img_bytes
    
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
            
        socket.send_pyobj({'step': t_step, 'image': img})
        data = socket.recv_pyobj() # 阻塞等待 0.02s
        
        raw_hand_enc = np.array(data['hand'])
        cmd_hand = map_encoder_to_motor(raw_hand_enc)
        cmd_hand[0] -= 5000 
        if cmd_hand[0] < 0: cmd_hand[0] = 0
        
        # 加锁执行
        with robot_lock:
            robot.set_hand_position(cmd_hand)
            
        t_step += 1

def dp_arm_thread(robot):
    """低频线程 (~1Hz): 负责与 DP 通讯并控制机械臂"""
    global is_running, global_img_bytes
    
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
            
        # 发送图片，服务端的高频 Buffer 会处理它
        socket.send_pyobj({'step': t_step, 'image': img})
        data = socket.recv_pyobj() # 阻塞等待 1s
        
        raw_delta = np.array(data['delta']) 
        delta_to_apply = raw_delta.copy()
        delta_to_apply[0] *= -1 
        delta_to_apply[4] *= -1 
        
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
    robot.set_hand_position([0]*6)
    
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