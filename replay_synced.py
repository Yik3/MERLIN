from data_sync import *
from Total_API import *
from merlin_replay import *
from replay_pose_right import *

encoder_path = '/home/rm/Documents/MERLIN/adc_data_20260205193055.csv'
pose_path = '/home/rm/Documents/MERLIN/iphone_data_20260205_193047.txt'
camera_path = '/home/rm/Documents/MERLIN/video_recording_realsense#20260205193056.npy'

IP = '169.254.128.19' 
FPS = 30
data = sync_data(camera_path, encoder_path, pose_path)

for i, (frame_idx, encoder_avg, pose_val) in enumerate(data):
    