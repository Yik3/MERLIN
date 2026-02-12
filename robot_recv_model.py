import zmq
import numpy as np
import time
from Total_API import RobotControlAPI 
# from Safety_Constraint import * # 如果需要安全约束请自行打开

# ================= CONFIGURATION =================
ROBOT_IP = "169.254.128.19"
ZMQ_PORT = 5555
# BASE_POSE = [-0.259968, -0.253127, 0.265704, 1.89, -0.996, -0.185]
BASE_POSE = [-0.259968, -0.253127, 0.255704, 2.05, -0.896, -0.185]
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
    """
    将模型预测的 Encoder 数值映射为电机控制信号
    包含特定的通道重排逻辑
    """
    motor_vals = []
    for i in range(6):
        ch_name = f'CH{i}'
        if ch_name in range_map:
            min_val, max_val = range_map[ch_name]
            # 线性映射
            mapped_val = gain * (encoder_vals[i] - min_val) * MAX_HAND_VAL / (max_val - min_val)
            motor_vals.append(int(mapped_val))
        else:
            motor_vals.append(0) 
            
    # [关键] Replay 脚本中的通道重排: [1, 2, 3, 4, 5, 0]
    ret_motor_vals = [motor_vals[1], motor_vals[2], motor_vals[3], motor_vals[4], motor_vals[5], motor_vals[0]]
    
    # 简单的边界限制，防止电机过载
    ret_motor_vals = np.clip(ret_motor_vals, 0, MAX_HAND_VAL).astype(int).tolist()
    return ret_motor_vals

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

    # 2. Setup ZMQ (Connect to the PAIR)
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.connect(f"tcp://localhost:{ZMQ_PORT}")
    print(f"[Robot] Connected to Inference Server on port {ZMQ_PORT}")

    # 3. State Maintenance
    current_pose = np.array(BASE_POSE, dtype=np.float32)

    try:
        while True:
            # --- A. Receive Command (Blocking) ---
            # 等待推理端发来数据
            data = socket.recv_pyobj()
            
            step = data['step']
            raw_delta = np.array(data['delta']) # 原始模型输出 Delta
            raw_hand_enc = np.array(data['hand']) # 原始模型输出 Hand (Encoder Domain)
            
            # --- B. Process Data ---
            
            # 1. Arm Processing (Apply Replay Script Logic)
            # Replay逻辑: rel_pose[0] *= -1, rel_pose[4] *= -1
            # 我们对 Delta 做同样的处理
            delta_to_apply = raw_delta.copy()
            delta_to_apply[0] *= -1 # X Axis Flip
            delta_to_apply[4] *= -1 # Pitch/Ry Flip
            
            # 积分计算新的 Target
            next_target_pose = current_pose + delta_to_apply
            
            # 2. Hand Processing (Mapping)
            # 将模型预测的 Encoder 值转换为 Motor 值
            cmd_hand = map_encoder_to_motor(raw_hand_enc)
            
            # --- C. Execute Command (Blocking) ---
            # 移动机械手
            cmd_hand[0] -= 5000 
            if cmd_hand[0] < 0:
                cmd_hand[0] = 0
            robot.set_hand_position(cmd_hand)
            # 移动机械臂 (Block直到完成)
            robot.move_arm_to_pose(next_target_pose, speed=20, block=True)
            time.sleep(0.01) # 小延时确保动作完成
            
            # --- D. Update Internal State ---
            current_pose = next_target_pose

            # --- E. Send Completion Signal ---
            socket.send_string("DONE")
            
            if step % 10 == 0:
                print(f"Exec Step {step} | Delta Z: {raw_delta[2]:.4f} | Hand[0]: {cmd_hand[0]}")

    except KeyboardInterrupt:
        print("\nStopping robot control...")
    except zmq.ZMQError as e:
        print(f"ZMQ Error: {e}")
    finally:
        # robot.disconnect() 
        print("Cleaned up.")

if __name__ == "__main__":
    main()