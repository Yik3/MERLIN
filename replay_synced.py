from data_sync import *
from Total_API import *
from merlin_replay import *
from replay_pose_right import *

encoder_path = '/home/rm/Documents/MERLIN/adc_data_20260205193055.csv'
pose_path = '/home/rm/Documents/MERLIN/iphone_data_20260205_193047.txt'
camera_path = '/home/rm/Documents/MERLIN/video_recording_realsense#20260205193056.npy'

fist_gesture = [0, 0, 0, 0, 0, MAX_VAL-5000]
IP = '169.254.128.19' 
FPS = 30
gain = 0.95

data = sync_data(camera_path, encoder_path, pose_path)
df = read_and_test_csv(encoder_path)
norm_data = normalize_and_visualize(df, if_plot=False,downsample=False)
raw_data_size = len(norm_data['CH0-ThumbLow'])
robot = RobotControlAPI(IP)
robot.set_hand_position(fist_gesture)
robot.move_arm_to_pose(BASE_POSE)
time.sleep(0.5)
FINGER_names = ['CH0-ThumbLow', 'CH1-ThumbUp', 'CH2-Pointer', 'CH3-Middle', 'CH4-Ring', 'CH5-Pinky']
for i, (frame_idx, encoder_idx, pose) in enumerate(data):
    encoder_data = None
    encoder_idx = int(encoder_idx)
    cur_gest = [0,0,0,0,0,0]
    for k in range(6):
        cur_finger = FINGER_names[k]
        if encoder_idx > raw_data_size - 15 or encoder_idx < 15:
            encoder_data = norm_data[cur_finger][encoder_idx] * gain
        else:
            encoder_data = np.mean(norm_data[cur_finger][encoder_idx-10:encoder_idx+10], axis=0) * gain
        cur_gest[k] = int(encoder_data)
    actual_pose = [0] * 6
    angle = get_euler_scipy(pose[3], pose[4], pose[5], pose[6])
    res = transform_frame_z90(pose[0], pose[1], pose[2], angle[0], angle[1], angle[2])
    for j in range(6):
        if j < 3:
            actual_pose[j] = res[j] + BASE_POSE[j]
        else:
            actual_pose[j] = float(res[j] + BASE_POSE[j])
            if actual_pose[j] > np.pi:
                actual_pose[j] -= 2 * np.pi
            elif actual_pose[j] < -np.pi:
                actual_pose[j] += 2 * np.pi
    robot.move_arm_to_pose(actual_pose)
    robot.set_hand_position(cur_gest)
    print(f"Frame {i}: Moved to Pose {actual_pose} with Encoder {cur_gest}")
    time.sleep(1/FPS)