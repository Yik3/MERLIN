import numpy as np
from Total_API import *

# read csv file
import pandas as pd
MAX_VAL = 65535
RIGHT_ARM_IP = '169.254.128.19'
# Global Dictionary mapping finger to channel
from scipy.signal import butter, filtfilt

RAW_FREQ = 990   
TARGET_FREQ = 30
range_map = {
    'CH0-ThumbLow': (1286, 2400),
    'CH1-ThumbUp': (1413, 1840),
    'CH2-Pointer': (2074, 2883),
    'CH3-Middle': (1902, 2742),
    'CH4-Ring': (1750, 2580),
    'CH5-Pinky': (1970, 2667) 
}
# Monotonicity Map (1 for large value means close, 0 for large value means open)
motor_monotonicity = {
    'CH0-ThumbLow': 1,
    'CH1-ThumbUp': 1,
    'CH2-Pointer': 1,
    'CH3-Middle': 1,
    'CH4-Ring': 1,
    'CH5-Pinky': 1
}
def butter_lowpass_filter(data, cutoff, fs, order=2):
    """
    cutoff: Cut off freq(Hz). Small->smooth, but high latency. Recommended values 2-5
    fs: FPS of the data. For 30 FPS, fs=30
    order: default to 2.
    """
    nyq = 0.5 * fs # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def downsample_average(data, original_freq, target_freq):
    """
    Downsample the data from original_freq to target_freq by averaging.
    data: 1D numpy array of the original data
    """
    if target_freq >= original_freq:
        return data 

    step = int(original_freq / target_freq)
    
    n_samples = len(data)
    trunc_len = (n_samples // step) * step

    truncated_data = data[:trunc_len]

    downsampled_data = truncated_data.reshape(-1, step).mean(axis=1)
    
    print(f"Downsampling: {n_samples} -> {len(downsampled_data)} frames (Window size: {step})")
    return downsampled_data

def read_and_test_csv(file_path):
    df = pd.read_csv(file_path)
    print("Loaded data shape:", df.shape)
    print(df.head())
    
    return df
def normalize_and_visualize(df,if_plot=False,enable_filter=True, cutoff=0.6, fs=30, downsample = True):
    if if_plot:
        import matplotlib.pyplot as plt
    normalized_data = {}
    for finger in range_map.keys():
        # df['CHX'] where X is the channel number
        col_name = finger
        raw_data = df[col_name].values
        # Pass to filter
        if enable_filter:
            # Nah dealing
            if np.isnan(raw_data).any():
                raw_data = pd.Series(raw_data).fillna(method='ffill').fillna(0).values
            
            filtered_data = butter_lowpass_filter(raw_data, cutoff=cutoff, fs=fs)
        else:
            filtered_data = raw_data
        # Normalize
        min_val, max_val = range_map[finger]
        norm_data = (filtered_data - min_val) * MAX_VAL / (max_val - min_val) 
        
        final_data = downsample_average(norm_data, RAW_FREQ, TARGET_FREQ) 
        if downsample:      
            normalized_data[finger] = final_data
        else:
            normalized_data[finger] = norm_data
        # plot the normalized data
        if if_plot:
            plt.plot(normalized_data[finger], label=finger)

    if if_plot:
        plt.xlabel('Frame')
        plt.ylabel('Normalized Position (0-max)')
        plt.title('Finger Position Normalization')
        plt.legend()
        plt.show()

    return normalized_data
def test_finger_with_csv(normalized_data, fps=30, gain=1.0):
    print("Initializing Robot API...")
    right_hand = RobotControlAPI(RIGHT_ARM_IP)
    fist_gesture = [0, 0, 0, 0, 0, 0]
    right_hand.set_hand_position(fist_gesture)
    time.sleep(3)
    num_frames = len(next(iter(normalized_data.values())))
    last_gest = [0,0,0,0,0,0]
    for i in range(num_frames):
        cur_gest = [0,0,0,0,0,0]
        for finger, norm_data in normalized_data.items():
            # print(finger, norm_data[i])
            value = norm_data[i] * gain
            if value > MAX_VAL:
                value = MAX_VAL
            # convert value to int
            value = int(np.round(value).astype(int))
            if motor_monotonicity[finger] == 1:
                if finger == 'CH0-ThumbLow':
                    cur_gest[0] = value
                elif finger == 'CH1-ThumbUp':
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
                if finger == 'CH0-ThumbLow':
                    cur_gest[0] = MAX_VAL - value
                elif finger == 'CH1-ThumbUp':
                    cur_gest[5] = MAX_VAL - value
                elif finger == 'CH2-Pointer':
                    cur_gest[1] = MAX_VAL - value
                elif finger == 'CH3-Middle':
                    cur_gest[2] = MAX_VAL - value
                elif finger == 'CH4-Ring':
                    cur_gest[3] = MAX_VAL - value
                elif finger == 'CH5-Pinky':
                    cur_gest[4] = MAX_VAL - value
        # For testing, disable some fingers
        NOT_THUMB = False
        NOT_FOUR = False
        if NOT_THUMB:
            cur_gest[0] = 0
            cur_gest[5] = 0
        if NOT_FOUR:
            cur_gest[1] = 0
            cur_gest[2] = 0
            cur_gest[3] = 0
            cur_gest[4] = 0
        # cur_gest[4] = 0  # Disable pinky for testing
        print(f"Frame {i+1}/{num_frames}: Setting gesture {cur_gest}")
        last_gest = cur_gest
        right_hand.set_hand_position(cur_gest)
        time.sleep(1/fps)
    
        
if __name__ == '__main__':
    df = read_and_test_csv('adc_data_20260205193055.csv')
    norm_data = normalize_and_visualize(df, if_plot=False)
    # print(norm_data)
    test_finger_with_csv(norm_data, fps=10, gain=1.0)
    