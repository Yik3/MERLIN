import argparse
import os
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# 假设 dataset.py 就在同一目录下
from dataset import Preloaded12DDataset, batch_sanity_check
# 假设 core 文件夹里有 build 和 kl_divergence
from core import build, kl_divergence
from torch.cuda.amp import autocast, GradScaler

# ===========================================================================
# CONFIGURATION
# ===========================================================================
# 请确保这些路径指向你刚才生成的数据
VID_DIR = '/home/classysh/MERLIN/MERLIN/data/211data/camera/mp4_files'
ENC_DIR = '/home/classysh/MERLIN/MERLIN/data/211data/encoder/processed_encoder'
ACT_DIR = '/home/classysh/MERLIN/MERLIN/data/211data/action/processed_action'
SYNC_DIR = '/home/classysh/MERLIN/MERLIN/data/211data/syncs'

# ===========================================================================
# ARGUMENTS
# ===========================================================================
parser = argparse.ArgumentParser(description="MERLIN Single-Cam 12D Training")
parser.add_argument('-e', '--epochs', type=int, default=20000, help='number of epochs')
parser.add_argument('-b', '--batch', type=int, default=8, help='batch size') # 显存够的话可以大一点
parser.add_argument('-n', '--norm_path', type=str, default="normalization_stats_12d.npz", help='stats save path')
parser.add_argument('-q', '--num_queries', type=int, default=70, help='chunk size')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5, help='learning rate')
args = parser.parse_args()

# ===========================================================================
# SETUP DEVICE
# ===========================================================================
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===========================================================================
# LOAD DATA
# ===========================================================================
print("Initializing Dataset...")
dataset = Preloaded12DDataset(
    vid_dir=VID_DIR,
    encoder_dir=ENC_DIR,
    action_dir=ACT_DIR,
    sync_dir=SYNC_DIR,
    first_cut=40,
    num_queries=args.num_queries,
    pad=True,
    transform='resnet_normalization' # or 'augment'
)

# 保存 Normalization Stats (推理时必须用同样的 mean/std)
qpos_mean = dataset.qpos_mean.numpy()
qpos_std = dataset.qpos_std.numpy()
np.savez(args.norm_path, qpos_mean=qpos_mean, qpos_std=qpos_std)
print(f"Normalization statistics saved to {args.norm_path}")

# Split Train/Val
n = len(dataset)
train_len = int(0.95 * n)
test_len  = n - train_len
train_dataset, test_dataset = random_split(dataset, [train_len, test_len])

train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(test_dataset,   batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)

# Sanity Check (Optional)
# batch_sanity_check(train_loader)

# ===========================================================================
# MODEL CONFIG
# ===========================================================================
class ModelArgs:
    def __init__(self):
        self.num_queries = args.num_queries
        
        # --- 关键修改 1: 摄像头列表 ---
        # 我们的 dataset 返回 shape [B, 1, 3, H, W]，对应这里的一个名字
        self.camera_names = ["cam_high"] 
        
        # --- 关键修改 2: 状态维度 ---
        # 6 (Arm) + 6 (Encoder) = 12
        self.state_dim = 12 
        
        # Transformer Hyperparameters (ACT Standard)
        self.hidden_dim = 512
        self.dropout = 0.1
        self.nheads = 8
        self.dim_feedforward = 3200
        self.enc_layers = 4
        self.dec_layers = 7
        self.pre_norm = False
        
        # Backbone
        self.position_embedding = 'sine'
        self.backbone = 'resnet18'
        self.lr_backbone = 1e-5
        self.masks = False
        self.dilation = False

model_args = ModelArgs()
model = build(model_args) # 这里的 build 函数通常在 detr/models 里的逻辑
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
scaler = GradScaler()

# ===========================================================================
# TRAINING LOOP
# ===========================================================================
loss_history, val_loss_history = [], []
os.makedirs("weights", exist_ok=True)

print(f"Start Training for {args.epochs} epochs...")

for epoch in range(args.epochs):
    # --- TRAIN ---
    model.train()
    running_loss = 0.0
    
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
    
    for batch in loop:
        # data structure: qpos, image, actions, is_pad
        qpos_batch   = batch[0].to(device) # [B, 12]
        image_batch  = batch[1].to(device) # [B, 1, 3, H, W]
        action_batch = batch[2].to(device) # [B, 100, 12]
        is_pad_batch = batch[3].to(device) # [B, 100]

        optimizer.zero_grad()

        with autocast():
            # env_state is typically None for standard ACT
            # Model Forward: returns (pred_action, _, (mu, logvar))
            pred_action, _, (mu, logvar) = model(qpos_batch, image_batch, env_state=None, actions=action_batch, is_pad=is_pad_batch)
            
            # KL Divergence Loss
            total_kld, _, _ = kl_divergence(mu, logvar)
            
            # L1 Reconstruction Loss
            # --- 关键修改 3: 计算所有 12 个维度的 Loss ---
            # output[..., :12] 其实就是全部，因为 pred_action 也是 12D
            loss_l1 = F.l1_loss(pred_action, action_batch, reduction='none') # [B, 100, 12]
            
            # Masking Padding
            mask = (~is_pad_batch).unsqueeze(-1) # [B, 100, 1]
            loss_l1 = (loss_l1 * mask).sum() / mask.sum()
            
            # Total Loss (beta is usually 10 or 100 depending on dataset scale, ALOHA uses 10-100)
            loss = loss_l1 + total_kld[0] * 10

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss_l1.item() # 只记录 L1 以观察重建效果
        loop.set_postfix(l1_loss=loss_l1.item())

    avg_train_loss = running_loss / len(train_loader)
    loss_history.append(avg_train_loss)

    # --- VALIDATION ---
    if epoch % 10 == 0: # Save time by valid every 10 epochs
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for batch in val_loader:
                qpos   = batch[0].to(device)
                image  = batch[1].to(device)
                action = batch[2].to(device)
                is_pad = batch[3].to(device)

                with autocast():
                    pred, _, _ = model(qpos, image, None) # eval mode no need actions for VAE
                    loss_val = F.l1_loss(pred, action, reduction='none')
                    mask = (~is_pad).unsqueeze(-1)
                    loss_val = (loss_val * mask).sum() / mask.sum()
                
                val_running += loss_val.item()
        
        avg_val_loss = val_running / len(val_loader)
        val_loss_history.append(avg_val_loss)
        print(f"Epoch {epoch+1} | Train L1: {avg_train_loss:.5f} | Val L1: {avg_val_loss:.5f}")
    
    # --- SAVE ---
    # 每 500 epoch 或在最后几个 epoch 频繁保存
    if (epoch + 1) % 1000 == 0 and epoch > args.epochs - 8000:
        save_path = f"weights/policy_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), save_path)

# Final Save
torch.save(model.state_dict(), "weights/policy_last.pth")
print("Training Complete. Weights saved.")

# Plot Loss
plt.figure()
plt.plot(loss_history, label='Train L1')
# Only display values that are smaller than 2
plt.ylim(0, 2)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_curve.png')