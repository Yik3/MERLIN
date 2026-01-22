import numpy as np
from Finger_API import *

def test_finger_with_npy(dir,fps=60,gain=1.0):
    # Load the numpy array from file
    data = np.load(dir,allow_pickle=True)
    print("Loaded data shape:", data.shape)
    print(data[0])
    right_hand = AoyiHand(hand_side='right')
    fist_gesture = [0, 0, 0, 0, 0, 0]
    right_hand.set_hand_6d(fist_gesture)
    time.sleep(1)
    for i in range(data.shape[0]):
        frame = data[i]
        cur_gest = [0,0,0,0,0,0]
        cur_gest[1] = (frame['Index']*gain) if frame['Index']*gain <=255 else 255
        cur_gest[2] = (frame['Middle']*gain) if frame['Middle']*gain <=255 else 255
        cur_gest[3] = (frame['Ring']*gain) if frame['Ring']*gain <=255 else 255
        cur_gest[4] = (frame['Pinky']*gain) if frame['Pinky']*gain <=255 else 255
        right_hand.set_hand_6d(cur_gest)
        time.sleep(1/fps)

if __name__ == '__main__':
    # Example usage
    test_finger_with_npy('finger_control_values_1.npy', fps=30,gain=1.0)