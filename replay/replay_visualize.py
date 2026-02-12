import numpy as np
import os
import pandas as pd
import matplotlib
# Set backend to Agg for server use (no display)
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# Output directory for saved plots
OUTPUT_DIR = "replay_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def smooth_data(data, window=8):
    """
    Apply rolling average to match DataLoader preprocessing.
    """
    padded = np.pad(data, ((window, window), (0, 0)), mode='edge')
    kernel_size = window * 2 + 1
    smoothed = np.zeros_like(data)
    
    for dim in range(data.shape[1]):
        col = padded[:, dim]
        ret = np.cumsum(col, dtype=float)
        ret[kernel_size:] = ret[kernel_size:] - ret[:-kernel_size]
        moving_sum = ret[kernel_size - 1 : kernel_size - 1 + len(data)]
        smoothed[:, dim] = moving_sum / kernel_size
        
    return smoothed

def load_raw_csv_data(npy_path):
    """
    Infers and loads the raw CSV file based on the NPY path.
    Assumes structure: .../encoder/processed_encoder/xxx.npy -> .../encoder/xxx.csv
    """
    try:
        base_name = os.path.splitext(os.path.basename(npy_path))[0]
        processed_dir = os.path.dirname(npy_path)
        encoder_root_dir = os.path.dirname(processed_dir)
        
        csv_path = os.path.join(encoder_root_dir, f"{base_name}.csv")
        
        if not os.path.exists(csv_path):
            print(f"⚠️ Raw CSV not found at: {csv_path}")
            return None
            
        print(f"Loading Raw CSV: {os.path.basename(csv_path)}")
        df = pd.read_csv(csv_path)
        
        range_map_keys = ['CH0-ThumbLow', 'CH1-ThumbUp', 'CH2-Pointer', 
                          'CH3-Middle', 'CH4-Ring', 'CH5-Pinky']
        extracted_data = []
        for col in range_map_keys:
            if col in df.columns:
                vals = df[col].values
                if np.isnan(vals).any():
                     vals = pd.Series(vals).fillna(method='ffill').fillna(0).values
                extracted_data.append(vals)
            else:
                print(f"Column {col} missing in CSV")
                return None
                
        return np.column_stack(extracted_data)
        
    except Exception as e:
        print(f"Error loading raw CSV: {e}")
        return None

def visualize_pair(enc_path, act_path):
    print(f"Loading Processed Encoder: {os.path.basename(enc_path)}")
    print(f"Loading Processed Action:  {os.path.basename(act_path)}")
    
    base_name = os.path.splitext(os.path.basename(act_path))[0]

    # 1. Load Processed NPY Data
    try:
        proc_enc = np.load(enc_path) # [T_enc, 6]
        raw_act = np.load(act_path)  # [T_act, 6]
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # 2. Try loading Raw CSV Data
    raw_csv_enc = load_raw_csv_data(enc_path)

    # 3. Report Lengths (No Cropping)
    len_enc = len(proc_enc)
    len_act = len(raw_act)
    print(f"\n--- Data Check ---")
    print(f"Processed Encoder Length: {len_enc}")
    print(f"Action Length:            {len_act}")
    if raw_csv_enc is not None:
        print(f"Raw CSV Length:           {len(raw_csv_enc)}")
    
    # 4. Process Data
    print("Processing: Smoothing Encoder (Window=8)...")
    final_input_enc = smooth_data(proc_enc, window=8)

    print("Processing: Integrating Action Deltas...")
    trajectory = np.cumsum(raw_act, axis=0) 

    # 5. Plotting
    # Create independent time axes for action and encoder
    time_steps_act = np.arange(len(trajectory))
    time_steps_enc = np.arange(len(final_input_enc))
    
    # If CSV exists, it might have a slightly different length than NPY (though usually same)
    time_steps_csv = np.arange(len(raw_csv_enc)) if raw_csv_enc is not None else None

    # --- Figure 1: XYZ Position ---
    plt.figure(figsize=(10, 6))
    labels = ['X', 'Y', 'Z']
    colors = ['r', 'g', 'b']
    for i in range(3):
        plt.plot(time_steps_act, trajectory[:, i], label=f'{labels[i]}', color=colors[i])
    plt.title(f"Action Trajectory: XYZ (Length {len_act})\nFile: {base_name}")
    plt.ylabel("Relative Position (m)")
    plt.xlabel("Time Step (Action)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{base_name}_xyz.png"))
    plt.close()
    print(f"Saved XYZ plot.")

    # --- Figure 2: Euler Angles ---
    plt.figure(figsize=(10, 6))
    labels = ['Rx', 'Ry', 'Rz']
    colors = ['c', 'm', 'y']
    for i in range(3):
        plt.plot(time_steps_act, trajectory[:, i+3], label=f'{labels[i]}', color=colors[i])
    plt.title(f"Action Orientation: Euler (Length {len_act})\nFile: {base_name}")
    plt.ylabel("Relative Rotation (rad)")
    plt.xlabel("Time Step (Action)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{base_name}_euler.png"))
    plt.close()
    print(f"Saved Euler plot.")

    # --- Figure 3: Encoder Comparison (Raw vs LPF vs Smoothed) ---
    finger_names = ['ThumbLow', 'ThumbUp', 'Pointer', 'Middle', 'Ring', 'Pinky']
    plt.figure(figsize=(18, 10))
    
    for i in range(6):
        plt.subplot(2, 3, i+1)
        
        # 1. Plot Raw CSV
        if raw_csv_enc is not None:
            plt.plot(time_steps_csv, raw_csv_enc[:, i], color='gray', alpha=0.3, label='Raw CSV (Noisy)')
            
        # 2. Plot Processed NPY (LPF)
        plt.plot(time_steps_enc, proc_enc[:, i], color='blue', alpha=0.5, linewidth=1, label='LPF NPY')
        
        # 3. Plot Final Input (Rolling Avg)
        plt.plot(time_steps_enc, final_input_enc[:, i], color='red', linewidth=1.5, label='Window Avg')
        
        plt.title(f"{finger_names[i]} (Len {len_enc})")
        if i == 0: 
            plt.legend(loc='upper right', fontsize='small')
        plt.grid(True, alpha=0.2)

    plt.suptitle(f"Encoder Data Pipeline Check: {base_name}", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{base_name}_encoder_compare.png"))
    plt.close()
    print(f"Saved Encoder Comparison plot.")
    
    print(f"\nAll plots saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    # === FILL IN PATHS HERE ===
    ENC_FILE = '/home/classysh/MERLIN/MERLIN/data/210data/encoder/processed_encoder/adc_data_20260210234340.npy'
    ACT_FILE = '/home/classysh/MERLIN/MERLIN/data/210data/action/processed_action/iphone_data_20260210_234313.npy'
    
    if not os.path.exists(ENC_FILE) or not os.path.exists(ACT_FILE):
        print("Error: File not found.")
    else:
        visualize_pair(ENC_FILE, ACT_FILE)