import zmq
import numpy as np
import time
import cv2
import threading
from collections import deque
import matplotlib.pyplot as plt
from finger_mapping import *
from Total_API import RobotControlAPI 

# ================= CONFIGURATION =================
GPU_IP = "192.168.1.100" 
DP_PORT = 5555   # Pure DP Channel

ROBOT_IP = "169.254.128.19"
CAMERA_ID = 20
RECED = True

# --- 新增的 Hand 逻辑配置 ---
ABSOLUTE = False  # True: DP 发来绝对 Encoder 位置；False: DP 发来相对 Delta
# 如果 ABSOLUTE=False，需要定义手指的初始 Encoder Base Value
BASE_HAND_ENCODER = [1800, 2000, 2000, 2200, 2000, 2000] 
ENABLE_LIVE_DELTA_PLOT = True
LIVE_PLOT_WINDOW = 200

BASE_POSE = [-0.142258,-0.287446,0.250924,2.78,-1.105,-1.107]
MAX_HAND_VAL = 65535

# ================= 全局变量与锁 =================
global_img_bytes = None
camera_lock = threading.Lock()
robot_lock = threading.Lock()  # 防止同时发指令给机器人导致崩溃
is_running = True
MAX_VAL = 65535

global_12d_state = np.array(BASE_POSE + BASE_HAND_ENCODER)
current_base = global_12d_state.copy()
state_lock = threading.Lock()

# Buffer 结构变更为存储 Tuple: (arm_action, hand_action)
global_chunk_buffer = []
chunk_lock = threading.Lock()

policy_hand_encoder_buffer = deque(maxlen=LIVE_PLOT_WINDOW)
plot_lock = threading.Lock()

# ================= HELPER FUNCTIONS =================
# def map_encoder_to_motor(encoder_vals, gain = 1.0):
#     motor_vals = []
#     for i in range(6):
#         ch_name = f'CH{i}'
#         if ch_name in range_map:
#             min_val, max_val = range_map[ch_name]
#             # Linear Mapping with Monotonicity Consideration
#             if monotonicity_map[ch_name] == 0:  # Monotonically Decreasing
#                 encoder_vals[i] = max(min(encoder_vals[i], max_val), min_val)  # Clip to range
#                 mapped_val = gain * (max_val - encoder_vals[i]) * MAX_VAL / (max_val - min_val)
#             else:  # Monotonically Increasing
#                 encoder_vals[i] = max(min(encoder_vals[i], max_val), min_val)  # Clip to range
#                 mapped_val = gain * (encoder_vals[i] - min_val) * MAX_VAL / (max_val - min_val)
#             motor_vals.append(int(mapped_val))
#         else:
#             motor_vals.append(0)  # Default to 0 if no mapping defined
#     ret_motor_vals = [motor_vals[1], motor_vals[2], motor_vals[4], motor_vals[3], motor_vals[5], motor_vals[0]]
#     return ret_motor_vals

def get_compressed_image(frame):
    ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return buffer.tobytes()

def live_delta_plot_thread():
    """实时绘制累计后的手指 encoder 位置 (6 维)."""
    global is_running

    if not ENABLE_LIVE_DELTA_PLOT:
        return

    plt.ion()
    fig, ax = plt.subplots(figsize=(11, 5))
    line_labels = [f"CH{i}" for i in range(6)]
    lines = [ax.plot([], [], label=label)[0] for label in line_labels]
    ax.set_title("Live Accumulated Hand Encoder Positions")
    ax.set_xlabel("Recent Step")
    ax.set_ylabel("Encoder Position")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    while is_running:
        with plot_lock:
            if len(policy_hand_encoder_buffer) > 0:
                deltas = np.array(policy_hand_encoder_buffer, dtype=np.float32)
            else:
                deltas = None

        if deltas is not None:
            x_axis = np.arange(deltas.shape[0])
            for i, line in enumerate(lines):
                line.set_data(x_axis, deltas[:, i])
            ax.set_xlim(0, max(1, len(x_axis) - 1))
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)
            fig.canvas.draw_idle()

        plt.pause(0.05)

    plt.ioff()
    plt.close(fig)

# ================= THREADS =================
def dp_communication_thread():
    """低频线程 (~1Hz): 负责与 DP 通讯并获取 Arm & Hand 数据"""
    global is_running, global_img_bytes, global_12d_state, global_chunk_buffer, current_base
    
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.connect(f"tcp://{GPU_IP}:{DP_PORT}")
    
    t_step = 0
    
    while is_running:
        with camera_lock:
            img = global_img_bytes
            
        if img is None:
            time.sleep(0.01)
            continue
        
        with state_lock:
            current_state = global_12d_state.copy()
            
        socket.send_pyobj({'step': t_step, 'image': img, 'state': current_state.tolist()})
        print(f"[DP Comm] Step {t_step} Sent Image & State, waiting for DP response...")
        
        data = socket.recv_pyobj() 
        print(f"[DP Comm] Step {t_step} Received Data")
        
        with chunk_lock:
            if RECED:
                # 假设服务端在 RECED=True 时返回序列: data['delta'] (Nx6) 和 data['hand'] (Nx6)
                arm_chunks = data['delta']
                hand_chunks = data['hand']
                current_base = data['base']
                # 将对应的 arm 和 hand 打包存入 buffer
                print(f"[DP Comm] Step {t_step} Received {len(arm_chunks)} Arm Deltas and {len(hand_chunks)} Hand Deltas")
                print(f"[DP Comm] Sample Arm Delta: {arm_chunks[0]}, Sample Hand Delta: {hand_chunks[0]}")
                global_chunk_buffer = list(zip(arm_chunks, hand_chunks))
                print(f"[DP Comm] Step {t_step} Buffer Updated. Buffer length: {len(global_chunk_buffer)}")
                print(f"[DP Comm] Step {t_step} First Buffer Item: Arm Delta: {global_chunk_buffer[0][0]}, Hand Delta: {global_chunk_buffer[0][1]}")
            else:
                # 假设服务端在 RECED=False 时返回单步数据
                # 注意：请根据你服务端的实际 key 进行修改
                single_arm = data['single_delta'] 
                single_hand = data['single_hand']
                global_chunk_buffer = [(single_arm, single_hand)]
                
            print(f"[DP Comm] Step {t_step} Buffer Updated. Buffer length: {len(global_chunk_buffer)}")
        
        t_step += 1

def execution_thread(robot):
    """独立线程: 统一处理 Buffer 提取并控制机械臂和手部"""
    global is_running, global_chunk_buffer, global_12d_state, current_base
    
    while is_running:
        with chunk_lock:
            if len(global_chunk_buffer) > 0:
                if RECED:
                    arm_action, hand_action = global_chunk_buffer.pop(0)
                else:
                    arm_action, hand_action = global_chunk_buffer[0]
                    global_chunk_buffer = []  # 单步执行后清空
                
                # 修复 3: 在锁内安全地获取关联的 base，防止计算时被网络线程覆盖
                local_base = np.array(current_base, dtype=np.float32)
            else:
                time.sleep(0.01)
                continue
                
        # 修复 1: 切片分离 arm base 和 hand base，防止 12D + 6D 导致崩溃
        base_arm = local_base[:6]
        # 防御性编程: 确保服务端确实传回了 12 维数据
        base_hand = local_base[6:12] if len(local_base) >= 12 else np.array(BASE_HAND_ENCODER)
        
        # --- 计算新的位姿和手部 Encoder ---
        arm_delta = np.array(arm_action, dtype=np.float32)
        hand_delta = np.array(hand_action, dtype=np.float32)

        # 核心逻辑: 永远在服务端传来的 base_arm 上累加
        next_target_pose = base_arm + arm_delta
        
        if ABSOLUTE:
            raw_hand_enc = hand_delta
        else:
            gain = 10
            # 修复 2: 永远在服务端传来的 base_hand 上累加，而不是使用死循环外的旧变量
            raw_hand_enc = base_hand + (hand_delta * gain)

        with plot_lock:
            policy_hand_encoder_buffer.append(raw_hand_enc.copy())

        # 更新全局状态（此数据会被通讯线程读取并发送给 Server 作为 current pose）
        with state_lock:
            global_12d_state[:6] = next_target_pose 
            global_12d_state[6:12] = raw_hand_enc 

        # 映射并处理手部边界和增益
        # 注意: 确保 map_encoder_to_motor 函数已从 finger_mapping 成功导入
        cmd_hand = map_encoder_to_motor(raw_hand_enc.copy())
        
        with robot_lock:
            # 发送硬件指令
            sent_cmd = [int(x) for x in cmd_hand]
            robot.set_hand_position(sent_cmd)
            robot.move_arm_to_pose(next_target_pose.tolist(), speed=20, block=True)
            print(f"[Execution] Executed Arm Pose: {next_target_pose} | Hand Encoders: {raw_hand_enc}")
        
        time.sleep(0.01)

# ================= MAIN =================
def main():
    global is_running, global_img_bytes
    
    robot = RobotControlAPI(ROBOT_IP)
    robot.move_arm_to_pose(BASE_POSE, speed=20, block=True)
    start_hand = [0, 0, 0, 0, 0, MAX_HAND_VAL]
    robot.set_hand_position(start_hand)

    cap = cv2.VideoCapture(CAMERA_ID)
    
    # 启动双线程
    t_comm = threading.Thread(target=dp_communication_thread)
    t_exec = threading.Thread(target=execution_thread, args=(robot,))
    t_plot = threading.Thread(target=live_delta_plot_thread)

    t_comm.start()
    t_exec.start()
    t_plot.start()

    print("\n=== STARTING THREADED CONTROL LOOP ===")
    try:
        # 主线程负责高频读取摄像头
        while True:
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1) # 水平翻转
                frame = cv2.flip(frame, 0) # 垂直翻转
                compressed = get_compressed_image(frame)
                with camera_lock:
                    global_img_bytes = compressed
            time.sleep(0.01) 
            
    except KeyboardInterrupt:
        print("\nStopping...")
        is_running = False
        t_comm.join()
        t_exec.join()
        t_plot.join()
        cap.release()

if __name__ == "__main__":
    main()
