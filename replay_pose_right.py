from Total_API import *
from Safety_Constraint import *
import numpy as np
BASE_POSE = [-0.196845, -0.328856, 0.042091, 1.517, -1.19, 0.07]
BASE_POSE_7D = convert_6d_to_7d(BASE_POSE)
from scipy.spatial.transform import Rotation as R

transformation_matrix = np.array([[0,1,0,0.0017],
                                [-1,0,0,0.03088],
                                [0,0,1,0.19273],
                                [0,0,0,1]]) 

def transform_frame_z90(x,y,z, rx, ry, rz):
    ret_x = y
    ret_y = -x
    ret_z = z
    ret_rx = ry
    ret_ry = -rx
    ret_rz = rz
    return [ret_x, ret_y, ret_z, ret_rx, ret_ry, ret_rz]

def recover(Pos,qunt,offset,base_euler):
    r_base = R.from_euler('xyz', base_euler, degrees=False)
    r_delta = R.from_quat(qunt)
    r_new = r_delta * r_base
    rotated_offset = r_new.apply(offset)
    new_pos = Pos - rotated_offset
    return new_pos

def get_euler_scipy(w, x, y, z):
    r = R.from_quat([x, y, z, w]) # scipy order x, y, z, w
    return r.as_euler('xyz')

def is_pose_safe(pose):
    if len(pose) == 7:
        x, y, z, qw, qx, qy, qz = pose
    elif len(pose) == 6:
        x, y, z, roll, pitch, yaw = pose
    else:
        raise ValueError("Pose must be either 6D (x, y, z, roll, pitch, yaw) or 7D (x, y, z, qw, qx, qy, qz)")
    return (X_min <= x <= X_max) and (Y_min <= y <= Y_max) and (Z_min <= z <= Z_max)

def parse_input_txt(file_path):
    pose = []
    with open(file_path, 'r') as f:
        #skip the first line (header)
        next(f)
        lines = f.readlines()
        # the value stored in txt is a string. Separate that by ,
        for i in range(len(lines)):
            lines[i] = lines[i].strip().split(',')
            # put the first seven as a tuple to pose
            pose.append(tuple(map(float, lines[i][1:8])))
    return pose


if __name__ == "__main__":
    poses = parse_input_txt('/home/rm/Documents/MERLIN/iphone_data_20260209_185943.txt')
    IP = "169.254.128.19"
    robot = RobotControlAPI(IP)
    euler_check = [[],[],[]]
    xyz_check = [[],[],[]]
    EEF_ONLY = True
    robot.move_arm_to_pose(BASE_POSE)
    time.sleep(1.5)
    for i, pose in enumerate(poses):
        actual_pose = [0] * 6
        angle = get_euler_scipy(pose[3], pose[4], pose[5], pose[6])
        
        # r = R.from_euler('xyz', angle,degrees=False)
        # M_current = np.eye(4)
        # M_current[:3, :3] = r.as_matrix()
        # M_current[:3, 3] = np.array(pose[:3])
        # M_transformed = transformation_matrix @ M_current
        # res = np.zeros(6)
        # res[:3] = M_transformed[:3, 3]
        # r_transformed = R.from_matrix(M_transformed[:3, :3])
        # res[3:] = r_transformed.as_euler('xyz', degrees=False)
        # temp = res[3]
        # res[3] = res[4]
        # res[4] = temp
        # res[5] = angle[-1]

        res = transform_frame_z90(pose[0], pose[1], pose[2], angle[0], angle[1], angle[2])
        recovered_pos = recover(res[:3], pose[3:7], offset=[0.03, 0.0017, 0.19273], base_euler=BASE_POSE[3:6])
        # recovered_pos = res
        recovered_pos[1] += 0.2
        for j in range(6):
            if j < 3:
                actual_pose[j] = float(recovered_pos[j] + BASE_POSE[j])
                xyz_check[j].append(actual_pose[j])
            else:
                actual_pose[j] = float(res[j] + BASE_POSE[j])
                if actual_pose[j] > np.pi:
                    actual_pose[j] -= 2 * np.pi
                elif actual_pose[j] < -np.pi:
                    actual_pose[j] += 2 * np.pi
                euler_check[j - 3].append(actual_pose[j]) 
        print(f"Actual Pose {i}: {actual_pose}")
        if i > 10:
            if is_pose_safe(actual_pose):
                print(f"Pose {i} is safe.")
                if EEF_ONLY:
                    send_pose = []
                    send_pose.extend(actual_pose[:3]) # x, y, z
                    send_pose.extend(BASE_POSE[3:6]) # roll, pitch, yaw
                    send_pose[4] = actual_pose[4]
                    send_pose[3] = actual_pose[3]
                    send_pose[5] = actual_pose[5]
                    # print(f"Sending EEF-only pose: {send_pose}")
                    robot.move_arm_to_pose(send_pose)
                    time.sleep(0.03)  # Adjust sleep time as needed
            else:
                print(f"Pose {i} is NOT safe.")
                if EEF_ONLY:
                    send_pose = []
                    send_pose.extend(actual_pose[:3]) # x, y, z
                    send_pose.extend(BASE_POSE[3:6]) # roll, pitch, yaw
                    # print(f"Sending EEF-only pose: {send_pose}")
                    send_pose[4] = actual_pose[4]
                    send_pose[3] = actual_pose[3]
                    send_pose[5] = actual_pose[5]
                    # print(f"Sending EEF-only pose: {send_pose}")
                    robot.move_arm_to_pose(send_pose)
                    time.sleep(0.03)  # Adjust sleep time as needed
    time.sleep(1.5)
    robot.move_arm_to_pose(BASE_POSE)
    time.sleep(1.5)
    PLOT = True
    if PLOT:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.subplot(3, 1, 1)
        plt.plot(euler_check[0], label='Roll')
        plt.ylabel('Roll (radians)')
        plt.legend()
        plt.subplot(3, 1, 2)
        plt.plot(euler_check[1], label='Pitch')
        plt.ylabel('Pitch (radians)')
        plt.legend()
        plt.subplot(3, 1, 3)
        plt.plot(euler_check[2], label='Yaw')
        plt.xlabel('Frame')
        plt.ylabel('Yaw (radians)')
        plt.legend()
        plt.suptitle('Euler Angles from Quaternions')
        plt.tight_layout()
        plt.show()
        plt.figure(figsize=(12, 6))
        plt.subplot(3, 1, 1)
        plt.plot(xyz_check[0], label='X')
        plt.ylabel('X (meters)')
        plt.legend()
        plt.subplot(3, 1, 2)
        plt.plot(xyz_check[1], label='Y')
        plt.ylabel('Y (meters)')
        plt.legend()
        plt.subplot(3, 1, 3)
        plt.plot(xyz_check[2], label='Z')
        plt.xlabel('Frame')
        plt.ylabel('Z (meters)')
        plt.legend()
        plt.suptitle('XYZ Positions from Quaternions')
        plt.tight_layout()
        plt.show()

    