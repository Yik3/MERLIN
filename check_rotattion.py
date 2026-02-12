import numpy as np
from scipy.spatial.transform import Rotation as R
from Total_API import *
# --- 你的基础配置 ---
BASE_POSE = [-0.196845, -0.328856, 0.042091, 1.517, -1.19, 0.07]

# Offset: 法兰中心 -> Sensor 中心 (在法兰坐标系下)
# 基于数据分析：
# 当 Roll (X) 增加时，Y 坐标大幅减小 (向负方向)。
# 根据右手定则，+Z 的向量绕 +X 旋转 90度会变成 -Y。
# 因此，为了抵消这个 -Y 运动，Offset 的 Z 分量应该是正的 (+0.19)。
# 如果你的物理安装确实是负的，请改回负数，但那样可能无法消除 Z/Y 的耦合运动。
OFFSET_VECTOR = [0.0017, 0.03088, 0.19273] 

def parse_input_txt(file_path):
    pose = []
    with open(file_path, 'r') as f:
        next(f) # skip header
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split(',')
            # parts[1:8] -> pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z
            pose.append(tuple(map(float, parts[1:8])))
    return pose

def get_euler_scipy(w, x, y, z):
    r = R.from_quat([x, y, z, w]) # scipy order x, y, z, w
    return r.as_euler('xyz')

def plot_3d_trajectory(xyz, M_type):
    import matplotlib.pyplot as plt
    title = "3D Trajectory" if M_type == '3D' else "XYZ over Time (Robot Frame)"
    if M_type == '3D':
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(xyz[0], xyz[1], xyz[2], marker='.', linestyle='-', alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel('X (Forward)')
        ax.set_ylabel('Y (Left)')
        ax.set_zlabel('Z (Up)')
        # 设置相同的比例尺以便观察真实轨迹
        try:
            ax.set_aspect('equal')
        except:
            pass
        plt.show()
    else:
        plt.figure(figsize=(10, 6))
        plt.plot(xyz[0], label='X (Forward)')
        plt.plot(xyz[1], label='Y (Left)')
        plt.plot(xyz[2], label='Z (Up)')
        plt.title(title)
        plt.xlabel('Frame')
        plt.ylabel('Position (m)')
        plt.grid(True)
        plt.legend()
        plt.show()

def plot_euler_angles(euler, title="Euler Angles"):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(euler[0], label='Roll (X)')
    plt.ylabel('Rad')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(euler[1], label='Pitch (Y)')
    plt.ylabel('Rad')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(euler[2], label='Yaw (Z)')
    plt.xlabel('Frame')
    plt.ylabel('Rad')
    plt.legend()
    plt.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def align_sensor_to_robot_identity(pos_sensor, quat_sensor):
    """
    修正后的对齐函数：直接透传 (Identity Mapping)。
    数据追踪显示 iPhone X 轴即为前进方向，与 Robot X 轴一致。
    """
    # 1. 位置直接透传
    pos_robot = np.array(pos_sensor)
    
    # 2. 姿态直接透传
    # 注意：这里只负责坐标系对齐，不负责 quaternion 顺序 (x,y,z,w) 的转换
    quat_robot = np.array(quat_sensor)
    
    return pos_robot, quat_robot

def recover_trajectory(pos_B_list, quat_B_list, offset_vector, base_pose):
    """
    批量处理还原函数
    pos_B_list: 传感器测量的相对位移列表
    quat_B_list: 传感器测量的姿态列表 (Delta Rotation)
    offset_vector: 杠杆臂向量
    base_pose: 机器人初始位姿
    """
    xyz_check = [[], [], []]
    euler_check = [[], [], []]
    
    # 准备 Base Rotation
    r_base = R.from_euler('xyz', base_pose[3:6], degrees=False)
    
    # 准备 Offset
    v_offset = np.array(offset_vector)
    
    for i in range(len(pos_B_list)):
        # 当前传感器数据
        pos_meas = pos_B_list[i]
        quat_meas = quat_B_list[i]
        
        # 1. 计算当前的 Delta 旋转 (相对于起始时刻)
        r_delta = R.from_quat(quat_meas)
        
        # 2. 计算因为旋转产生的“杠杆位移” (Lever Arm Shift)
        # 逻辑：当前杠杆状态 - 初始杠杆状态
        # 这就是“绕圈”产生的额外位移
        lever_arm_shift = r_delta.apply(v_offset) - v_offset
        
        # 3. 还原法兰的 Delta 位移
        # 法兰位移 = 测量位移 - 杠杆位移
        flange_delta_pos = pos_meas - lever_arm_shift
        
        # 4. 叠加到 Base 初始位置 (得到绝对坐标)
        final_pos = flange_delta_pos + np.array(base_pose[0:3])
        
        # 5. 计算绝对姿态 (用于发送给机器人)
        # 姿态 = Sensor变动 * 初始姿态 (或者 初始 * 变动，取决于控制逻辑)
        # 这里假设 Sensor 变动是基于当前工具坐标系的
        r_total = r_delta * r_base
        final_euler = r_total.as_euler('xyz', degrees=False)
        
        # 存入列表用于绘图
        for j in range(3):
            xyz_check[j].append(final_pos[j])
            euler_check[j].append(final_euler[j])
            
    return xyz_check, euler_check

# --- Main Logic ---

if __name__ == "__main__":
    # 1. 读取数据
    poses = parse_input_txt('210data/iphone_data_20260210_195831.txt')
    IP = "169.254.128.19"
    robot = RobotControlAPI(IP)
    pos_list = []
    quat_list = []
    PLOT = True
    ENABLE_ROBOT = False
    #robot.move_arm_to_pose(BASE_POSE)
    #time.sleep(1.5)
    for pose in poses:
        # 提取位置 (x, y, z)
        raw_pos = list(pose[0:3])
        
        # 提取四元数，TXT格式: [w, x, y, z] -> Scipy格式: [x, y, z, w]
        raw_quat = [pose[4], pose[5], pose[6], pose[3]]
        
        # [Step 1] 坐标系对齐 (Identity)
        rob_pos, rob_quat = align_sensor_to_robot_identity(raw_pos, raw_quat)
        
        pos_list.append(rob_pos)
        quat_list.append(rob_quat)
        
    # [Step 2] 核心还原算法
    xyz_res, euler_res = recover_trajectory(
        pos_list, 
        quat_list, 
        offset_vector=OFFSET_VECTOR, 
        base_pose=BASE_POSE
    )
    
    # [Step 3] 绘图验证
    

    # [Step 4] 发送给机器人 (如果启用)

    xyz_res_1 = [[], [], []]
    euler_res_1 = [[], [], []]
    for i in range(len(xyz_res[0])):
        angle = get_euler_scipy(poses[i][3], poses[i][4], poses[i][5], poses[i][6])

        target_pose = [
            xyz_res[1][i], 
            -xyz_res[0][i], 
            xyz_res[2][i], 
            angle[1] + BASE_POSE[3],
            -angle[0] + BASE_POSE[4],
            angle[2] + BASE_POSE[5],
        ]
        target_pose[1] -= 0.46
        xyz_res_1[0].append(target_pose[0])
        xyz_res_1[1].append(target_pose[1])
        xyz_res_1[2].append(target_pose[2])
        euler_res_1[0].append(target_pose[3])
        euler_res_1[1].append(target_pose[4])
        euler_res_1[2].append(target_pose[5])
        if ENABLE_ROBOT:
            print(f"Sending Pose {i}: {target_pose}")
            robot.move_arm_to_pose(target_pose)
            time.sleep(0.01) 
    if PLOT:
        print("Close the plot windows to finish...")
        plot_3d_trajectory(xyz_res_1, M_type='Time')
        plot_3d_trajectory(xyz_res_1, M_type='3D')
        plot_euler_angles(euler_res_1, title="Recovered Robot Flange Pose")