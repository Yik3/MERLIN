import numpy as np
from Finger_API import *

# read csv file
import pandas as pd

# Global Dictionary mapping finger to channel
range_map = {
    'CH0-ThumbLower': (1152, 2560),
    'CH1-ThumbUpper': (1400, 2083),
    'CH2-Pointer': (1780, 2730),
    'CH3-Middle': (1700, 2630),
    'CH4-Ring': (1830, 2580),
    'CH5-Pinky': (3950, 4010)
}
# Monotonicity Map (1 for large value means close, 0 for large value means open)
motor_monotonicity = {
    'CH0-ThumbLower': 1,
    'CH1-ThumbUpper': 1,
    'CH2-Pointer': 1,
    'CH3-Middle': 1,
    'CH4-Ring': 1,
    'CH5-Pinky': 1
}
def read_and_test_csv(file_path):
    df = pd.read_csv(file_path)
    print("Loaded data shape:", df.shape)
    print(df.head())
    
    return df
def normalize_and_visualize(df,if_plot=False):
    if if_plot:
        import matplotlib.pyplot as plt
    normalized_data = {}
    for finger in range_map.keys():
        # df['CHX'] where X is the channel number
        col_name = finger
        data = df[col_name].values
        # Normalize
        min_val, max_val = range_map[finger]
        norm_data = (data - min_val) / (max_val - min_val) * 255
        # plot the normalized data
        if if_plot:
            plt.plot(norm_data, label=finger)
        normalized_data[finger] = norm_data

    if if_plot:
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
            # print(finger, norm_data[i])
            value = norm_data[i] * gain
            if value > 255:
                value = 255
            # convert value to int
            value = int(np.round(value).astype(int))
            if motor_monotonicity[finger] == 1:
                if finger == 'CH0-ThumbLower':
                    cur_gest[0] = value
                elif finger == 'CH1-ThumbUpper':
                    cur_gest[5] = value
                elif finger == 'CH2-Pointer':
                    cur_gest[1] = value
                elif finger == 'CH3-Middle':
                    cur_gest[2] = value
                elif finger == 'CH4-Ring':
                    cur_gest[3] = value
                elif finger == 'CH5-Pinky':
                    cur_gest[4] = value
            else:
                if finger == 'CH0-ThumbLower':
                    cur_gest[0] = 255 - value
                elif finger == 'CH1-ThumbUpper':
                    cur_gest[5] = 255 - value
                elif finger == 'CH2-Pointer':
                    cur_gest[1] = 255 - value
                elif finger == 'CH3-Middle':
                    cur_gest[2] = 255 - value
                elif finger == 'CH4-Ring':
                    cur_gest[3] = 255 - value
                elif finger == 'CH5-Pinky':
                    cur_gest[4] = 255 - value
        # If action is too small, just don't do it
        for j in range(len(cur_gest)):
            if abs(cur_gest[j] - last_gest[j]) < 0:
                cur_gest[j] = last_gest[j]
        
        
        NOT_THUMB = True
        NOT_FOUR = False
        if NOT_THUMB:
            cur_gest[0] = 0
            cur_gest[5] = 0
        if NOT_FOUR:
            cur_gest[1] = 0
            cur_gest[2] = 0
            cur_gest[3] = 0
            cur_gest[4] = 0
        cur_gest[4] = 0  # Disable pinky for testing
        print(f"Frame {i+1}/{num_frames}: Setting gesture {cur_gest}")
        last_gest = cur_gest
        right_hand.set_hand_6d(cur_gest)
        time.sleep(1/fps)
    
        
if __name__ == '__main__':
    df = read_and_test_csv('Demo1.csv')
    norm_data = normalize_and_visualize(df)
    # print(norm_data)
    test_finger_with_csv(norm_data, fps=10, gain=1.0)
    