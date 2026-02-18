import os
import cv2
import torch
import numpy as np
import glob
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor

# ================= 配置区域 =================
INPUT_DIR = '../data/211data/camera/mp4_files'       # 输入文件夹
OUTPUT_DIR = '../data/211data/camera/mask_vid'   # 输出文件夹

# SAM 1 配置 (请确保下载了对应的权重文件)
# 下载地址: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
CHECKPOINT = "./sam_vit_h_4b8939.pth" 
MODEL_TYPE = "vit_h" # 对应 checkpoints: vit_h, vit_l, or vit_b

DARKEN_FACTOR = 0.2  # 0.0=全黑, 0.2=保留20%亮度
# ===========================================

# 全局变量存储点击
global_points = []
global_labels = []

def mouse_callback(event, x, y, flags, param):
    global global_points, global_labels
    if event == cv2.EVENT_LBUTTONDOWN:
        global_points.append([x, y])
        global_labels.append(1) # Positive
        print(f"  [+] 选中点: ({x}, {y})")
        # 在画面上画点反馈
        cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("CLICK ON FRAME 1 (Space to Finish)", param)
    elif event == cv2.EVENT_RBUTTONDOWN:
        global_points.append([x, y])
        global_labels.append(0) # Negative
        print(f"  [-] 排除点: ({x}, {y})")
        cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("CLICK ON FRAME 1 (Space to Finish)", param)

def get_initial_prompts(first_video_path):
    """
    打开第一个视频的第一帧，获取用户点击
    """
    global global_points, global_labels
    # 重置
    global_points = []
    global_labels = []

    cap = cv2.VideoCapture(first_video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret: raise ValueError("无法读取第一个视频")

    window_name = "CLICK ON FRAME 1 (Space to Finish)"
    cv2.namedWindow(window_name)
    # 传入 frame copy 以便实时绘图
    display_img = frame.copy()
    cv2.setMouseCallback(window_name, mouse_callback, param=display_img)

    print("\n=== 标注阶段 ===")
    print("1. 左键点击要把哪里变黑 (Robot/Hand)")
    print("2. 右键点击背景 (排除)")
    print("3. 按【空格】结束，开始批处理所有视频")

    while True:
        cv2.imshow(window_name, display_img)
        k = cv2.waitKey(20)
        if k == 32: # Space
            if not global_points:
                print("请至少点一个点！")
                continue
            break
        elif k == 27: # ESC
            print("退出程序")
            exit()

    cv2.destroyAllWindows()
    return np.array(global_points), np.array(global_labels)

def process_single_video(predictor, video_path, output_path, points, labels):
    """
    SAM 1 处理逻辑：逐帧读取 -> 逐帧推理 (无时序记忆)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 初始化视频写入
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 进度条
    pbar = tqdm(total=total_frames, desc=f"Processing {os.path.basename(video_path)}", leave=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # SAM 1 需要 RGB 格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. 设置图像 (SAM 1 这一步比较耗时，因为它要计算 Image Embedding)
        predictor.set_image(frame_rgb)

        # 2. 推理
        # multimask_output=False 意味着我们只要一个最好的 mask
        masks, scores, logits = predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=False
        )
        
        # SAM 返回的 masks 形状是 (1, H, W)，取第一个
        mask = masks[0]

        # 3. 应用 Mask (变黑)
        # mask 是 bool 类型，True 的地方变黑
        frame = frame.astype(np.float32)
        frame[mask] *= DARKEN_FACTOR
        frame = frame.astype(np.uint8)

        out.write(frame)
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

def main():
    # Setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    video_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.mp4")))
    if not video_files:
        print(f"在 {INPUT_DIR} 找不到视频文件！")
        return

    # 1. 交互标注 (只对第一个视频)
    print(f"正在加载第一个视频用于标注: {video_files[0]}")
    points, labels = get_initial_prompts(video_files[0])
    print(f"标注完成。捕获了 {len(points)} 个点。")

    # 2. 加载 SAM 1 模型
    print(f"正在加载 SAM 1 ({MODEL_TYPE})...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT)
        sam.to(device=device)
        predictor = SamPredictor(sam)
    except Exception as e:
        print(f"模型加载失败: {e}")
        print(f"请检查路径: {CHECKPOINT} 是否存在，且与 MODEL_TYPE={MODEL_TYPE} 匹配。")
        return

    # 3. 批量循环
    print("开始批处理 (SAM 1 逐帧处理速度较慢，请耐心等待)...")
    for vid_path in tqdm(video_files, desc="Total Progress"):
        filename = os.path.basename(vid_path)
        out_path = os.path.join(OUTPUT_DIR, filename)
        
        # 跳过已存在的
        if os.path.exists(out_path):
            continue
            
        try:
            process_single_video(predictor, vid_path, out_path, points, labels)
        except Exception as e:
            print(f"\n处理视频 {filename} 失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n全部完成！")

if __name__ == "__main__":
    main()