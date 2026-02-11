import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from torchvision import transforms

class Preloaded12DDataset(Dataset):
    def __init__(self, 
                 vid_dir, 
                 encoder_dir,   # 存 .npy (Pre-processed Absolute Encoder)
                 action_dir,    # 存 .npy (Pre-processed Delta Arm Actions)
                 sync_dir,      # 存 .npz (indices)
                 first_cut=30,  
                 num_queries=100, 
                 pad=True, 
                 transform='resnet_normalization',
                 max_demos=None,
                 encoder_window=8): # Encoder 平滑窗口
        
        super().__init__()
        self.encoder_window = encoder_window
        self.num_queries = num_queries
        self.pad = pad
        
        # --- Image Transforms ---
        normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        base_normalization = transforms.Compose([transforms.ToTensor(), normalize_transform])
        
        augment_pipeline = transforms.Compose([
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            normalize_transform
        ])
        
        if 'resnet_normalization' in transform:
            self.image_transforms = [base_normalization]
        else:
            self.image_transforms = [transforms.ToTensor()]
        if 'augment' in transform:
            self.image_transforms.append(augment_pipeline)

        # --- Files ---
        self.vid_files = sorted([os.path.join(vid_dir, f) for f in os.listdir(vid_dir) if f.endswith('.mp4')])
        self.enc_files = sorted([os.path.join(encoder_dir, f) for f in os.listdir(encoder_dir) if f.endswith('.npy')])
        self.act_files = sorted([os.path.join(action_dir, f) for f in os.listdir(action_dir) if f.endswith('.npy')])
        self.sync_files = sorted([os.path.join(sync_dir, f) for f in os.listdir(sync_dir) if f.endswith('.npz')])
        
        if max_demos is not None:
            self.vid_files = self.vid_files[:max_demos]
            self.enc_files = self.enc_files[:max_demos]
            self.act_files = self.act_files[:max_demos]
            self.sync_files = self.sync_files[:max_demos]

        assert len(self.vid_files) == len(self.sync_files), "Mismatch files count."

        # --- Global Buffers ---
        self.images_list = [] 
        self.state_list = [] # 统一存储 12D 向量 [Delta_Arm(6), Abs_Encoder(6)]
        self.episode_lengths = []
        
        print(f"Loading and processing data from {len(self.vid_files)} demos...")
        
        for i in range(len(self.vid_files)):
            try:
                # 1. Load Raw Data
                raw_enc = np.load(self.enc_files[i]) # [T_enc, 6]
                raw_arm = np.load(self.act_files[i]) # [T_act, 6] (Deltas)
                sync_data = np.load(self.sync_files[i])
                
                frame_idxs = sync_data['frame_idx'].astype(int)
                enc_idxs = sync_data['encoder_idx'].astype(int)
                pose_idxs = sync_data['pose_idx'].astype(int)
            except Exception as e:
                print(f"Error loading demo {i}: {e}")
                continue

            # 2. Smooth Encoder (Pre-calculation)
            smoothed_enc = self._apply_rolling_average(raw_enc, window=self.encoder_window)

            # 3. First Cut
            if len(frame_idxs) <= first_cut:
                continue
            
            frame_idxs = frame_idxs[first_cut:]
            enc_idxs = enc_idxs[first_cut:]
            pose_idxs = pose_idxs[first_cut:]
            
            # 4. Load Video
            all_frames = self._load_video_frames(self.vid_files[i])
            
            # 5. Align and Construct 12D State
            ep_images = []
            ep_states = [] # List of [12] tensors
            
            max_frame = len(all_frames) - 1
            max_arm = len(raw_arm) - 1
            max_enc = len(smoothed_enc) - 1
            
            valid_steps = 0
            for t in range(len(frame_idxs)):
                f_idx = min(frame_idxs[t], max_frame)
                p_idx = min(pose_idxs[t], max_arm)
                e_idx = min(enc_idxs[t], max_enc)
                
                # Image
                ep_images.append(all_frames[f_idx])
                
                # Construct 12D Vector: [Arm (6), Encoder (6)]
                arm_vec = raw_arm[p_idx]
                enc_vec = smoothed_enc[e_idx]
                combined_state = np.concatenate([arm_vec, enc_vec], axis=0) # [12]
                
                ep_states.append(torch.from_numpy(combined_state).float())
                valid_steps += 1
            
            self.images_list.extend(ep_images)
            self.state_list.extend(ep_states)
            self.episode_lengths.append(valid_steps)
            print(f"  Demo {i} loaded: {valid_steps} steps.")

        # --- 6. Calculate Normalization Stats on 12D Vector ---
        print("Calculating Global Normalization Stats (12D)...")
        
        # Stack all states into a big [N, 12] tensor
        all_states = torch.stack(self.state_list)
        
        # Calculate per-dimension mean and std
        self.qpos_mean = all_states.mean(dim=0) # Shape [12]
        self.qpos_std = all_states.std(dim=0).clamp(min=1e-3) # Shape [12]
        
        print(f"  Total Samples: {len(all_states)}")
        print(f"  Mean Shape: {self.qpos_mean.shape} (Expect [12])")
        print(f"  Std Shape:  {self.qpos_std.shape} (Expect [12])")
        print(f"  Mean Values (First 6 Arm): {self.qpos_mean[:6].numpy().round(3)}")
        print(f"  Mean Values (Last 6 Enc):  {self.qpos_mean[6:].numpy().round(3)}")

    def _apply_rolling_average(self, data, window=8):
        padded = np.pad(data, ((window, window), (0, 0)), mode='edge')
        kernel_size = window * 2 + 1
        smoothed = np.zeros_like(data)
        for dim in range(data.shape[1]):
            col = padded[:, dim]
            ret = np.cumsum(col, dtype=float)
            ret[kernel_size:] = ret[kernel_size:] - ret[:-kernel_size]
            smoothed[:, dim] = ret[kernel_size - 1 : kernel_size - 1 + len(data)] / kernel_size
        return smoothed.astype(np.float32)

    def _load_video_frames(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return frames

    def __len__(self):
        return len(self.episode_lengths)

    def __getitem__(self, idx):
        view_type = idx // len(self.episode_lengths)
        ep_idx = idx % len(self.episode_lengths)
        
        ep_len = self.episode_lengths[ep_idx]
        ep_start = sum(self.episode_lengths[:ep_idx])
        
        # Random Start Time
        t = np.random.randint(0, ep_len)
        sample_idx = ep_start + t
        
        # 1. Image
        img_np = self.images_list[sample_idx]
        tf = self.image_transforms[view_type] if view_type < len(self.image_transforms) else self.image_transforms[0]
        image = tf(img_np).unsqueeze(0)
        
        # 2. Qpos (Current State, 12D)
        raw_qpos = self.state_list[sample_idx]
        qpos_norm = (raw_qpos - self.qpos_mean) / self.qpos_std
        
        # 3. Actions (Future Sequence, 12D)
        global_end = ep_start + ep_len
        slice_end = min(global_end, sample_idx + self.num_queries)
        
        raw_actions_list = self.state_list[sample_idx : slice_end]
        
        if len(raw_actions_list) > 0:
            raw_actions = torch.stack(raw_actions_list) # [T_chunk, 12]
        else:
            raw_actions = torch.zeros((0, 12))
            
        # Normalize Future Actions
        # Broadcasting: [T, 12] - [12] / [12]
        actions_norm = (raw_actions - self.qpos_mean) / self.qpos_std
        
        # Padding
        pad_len = self.num_queries - len(actions_norm)
        is_pad = torch.zeros(self.num_queries, dtype=torch.bool)
        
        if self.pad and pad_len > 0:
            last_action = actions_norm[-1] if len(actions_norm) > 0 else torch.zeros(12)
            pad_block = last_action.unsqueeze(0).repeat(pad_len, 1)
            actions_norm = torch.cat([actions_norm, pad_block], dim=0)
            is_pad[-pad_len:] = True
            
        return qpos_norm, image, actions_norm, is_pad

# --- Sanity Check ---
def batch_sanity_check(train_loader):
    print("\n=== DataLoader 12D Unified Stats Check ===")
    batch = next(iter(train_loader))
    qpos, images, actions, is_pad = batch
    
    print(f"Batch Size:   {qpos.shape[0]}")
    print(f"Qpos Shape:   {qpos.shape} \t(Expect [B, 12])")
    print(f"Action Shape: {actions.shape} \t(Expect [B, 100, 12])")
    
    # Check if we are using the correct mean/std
    ds = train_loader.dataset
    print("\n--- Stats Verification ---")
    print(f"Mean Vector Shape: {ds.qpos_mean.shape} (Must be 12)")
    print(f"Std Vector Shape:  {ds.qpos_std.shape} (Must be 12)")
    
    # Check Normalization effect
    print(f"\nBatch Qpos Mean: {qpos.mean().item():.3f} (Ideal ~0)")
    print(f"Batch Qpos Std:  {qpos.std().item():.3f} (Ideal ~1)")
    print("==========================================")