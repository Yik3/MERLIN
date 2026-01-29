from merlin_replay import * 
import matplotlib.pyplot as plt

def normalize_with_gain_and_action_threshold(data, gain=1.0, action_threshold=0,visualize=False):
    # Normalize data to 0-255 with gain and apply action threshold
    normalized_data = {}
    for finger in motor_channel_map.keys():
        channel_num = motor_channel_map[finger]
        col_name = f'CH{channel_num}'
        raw_data = data[col_name].values
        # Normalize
        norm_data = (raw_data - np.min(raw_data)) / 4096 * 255
        # Apply gain
        norm_data = norm_data * gain
        norm_data = np.round(norm_data).astype(int)
        # Clip to 0-255
        norm_data = np.clip(norm_data, 0, 255)
        # Apply action threshold
        processed_data = np.copy(norm_data)
        for i in range(1, len(norm_data)):
            if abs(norm_data[i] - norm_data[i-1]) < action_threshold:
                processed_data[i] = processed_data[i-1]
        normalized_data[finger] = processed_data

        # Visualization
        if visualize:
            plt.plot(processed_data, label=finger)
    if visualize:
        plt.xlabel('Frame')
        plt.ylabel('Normalized Position (0-255)')
        plt.title('Finger Position Normalization with Gain and Threshold')
        plt.legend()
        plt.show()
df = read_and_test_csv('indexmiddlepicky.csv')
normalized_data = normalize_with_gain_and_action_threshold(df, gain=5.0, action_threshold=5,visualize=True)
