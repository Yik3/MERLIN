import numpy as np
from Finger_API import *
import matplotlib.pyplot as plt
# read csv file
import pandas as pd

# Global Dictionary mapping finger to channel
motor_channel_map = {
    'Index': 2,
    'Middle': 1,
    'Pinky': 4
}
# Monotoniticy Map (1 for large value means close, 0 for large value means open)
motor_monotonicity = {
    'Index': 1,
    'Middle': 1,
    'Pinky': 1
}
def read_and_test_csv(file_path):
    df = pd.read_csv(file_path)
    print("Loaded data shape:", df.shape)
    print(df.head())
    
    return df
def normalize_and_visualize(df):
    normalized_data = {}
    for finger in motor_channel_map.keys():
        # df['CHX'] where X is the channel number
        channel_num = motor_channel_map[finger]
        col_name = f'CH{channel_num}'
        data = df[col_name].values
        # Normalize
        norm_data = (data - np.min(data)) / 4096 * 255
        # plot the normalized data
        plt.plot(norm_data, label=finger)
        normalized_data[finger] = norm_data

    plt.xlabel('Frame')
    plt.ylabel('Normalized Position (0-255)')
    plt.title('Finger Position Normalization')
    plt.legend()
    plt.show()
    
    return normalized_data
def test_finger_with_csv(normalized_data, fps=30, gain=1.0):
    right_hand = AoyiHand(hand_side='right')
    fist_gesture = [0, 0, 0, 0, 0, 0]
    right_hand.set_hand_6d(fist_gesture)
    time.sleep(1)
    num_frames = len(next(iter(normalized_data.values())))
    last_gest = [0,0,0,0,0,0]
    for i in range(num_frames):
        cur_gest = [0,0,0,0,0,0]
        for finger, norm_data in normalized_data.items():
            value = norm_data[i] * gain
            if value > 255:
                value = 255
            if motor_monotonicity[finger] == 1:
                cur_gest_index = list(motor_channel_map.keys()).index(finger)
                cur_gest[cur_gest_index] = value
            else:
                cur_gest_index = list(motor_channel_map.keys()).index(finger)
                cur_gest[cur_gest_index] = 255 - value
        # If action is too small, just don't do it
        for j in range(len(cur_gest)):
            if abs(cur_gest[j] - last_gest[j]) < 5:
                cur_gest[j] = last_gest[j]
        last_gest = cur_gest
        right_hand.set_hand_6d(cur_gest)
        time.sleep(1/fps)
    
        
if __name__ == '__main__':
    df = read_and_test_csv('indexmiddlepicky.csv')
    norm_data = normalize_and_visualize(df)
    test_finger_with_csv(norm_data, fps=30, gain=1.0)
    