import numpy as np
import torch
import time


# 假设你的 inference class 保存在 merlin_inference.py 中
from merlin_inference import MerlinPolicyInference


def run_dummy_test():
    print("=== 开始 MERLIN Dummy 推理测试 ===")
   
    # 1. 设置模型路径 (请替换为你的真实权重文件路径)
    checkpoint_path = "../weights/diff_policy.ckpt" # 或者 .ckpt / .pth
   
    print(f"尝试加载模型: {checkpoint_path}")
    try:
        policy_runner = MerlinPolicyInference(
            checkpoint_path=checkpoint_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            action_mode="all"  # 选择 "all" 以便我们查看它输出的所有 action 维度
        )
        print("✅ 模型加载成功！")
        print(f"模型设备: {policy_runner.device}")
    except Exception as e:
        print(f"❌ 模型加载失败 (如果路径是 dummy 的，这是正常现象): \n{e}")
        print("请替换为真实的 Checkpoint 路径后再运行。")
        return


    # 2. 构建 Dummy 数据
    print("\n生成 Dummy 输入数据...")
   
    # 图像: 480 (H) x 640 (W) x 3 (C), 模拟普通 RGB 摄像头 (uint8, 0-255)
    dummy_image = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
   
    # 状态: 12 维向量 (通常是 6维机械臂 + 6维夹爪/手部状态)
    dummy_state = np.random.randn(12).astype(np.float32)
   
    print(f" - Image shape: {dummy_image.shape}, dtype: {dummy_image.dtype}")
    print(f" - State shape: {dummy_state.shape}, dtype: {dummy_state.dtype}")


    # 3. 运行推理 (模拟运行几个控制周期)
    print("\n执行推理循环 (3 个步长)...")
   
    # 记得调用 reset 清理内部的 history 队列
    policy_runner.reset()
   
    for step in range(10):
        print(f"\n--- Step {step + 1} ---")
        try:
            # 我们可以每一跳稍微改变一下 dummy 数据，或者直接传一样的
            time_start = time.time()
            output = policy_runner.predict(image=dummy_image, robot_state=dummy_state)
            time_end = time.time()
           
            # 4. 解析并打印输出
            latency_ms = output['latency_sec'] * 1000
            latency_measured = time_end - time_start
            print(f"⏱️ 推理延迟: {latency_ms:.2f} ms (实际测量: {latency_measured*1000:.2f} ms)")
           
            # "action" 是当前应该执行的单步动作
            action = output.get("action")
            if action is not None:
                print(f"🤖 单步 Action (shape {action.shape}): {action}")
               
            # "action_chunk" 是当前预测窗口内的动作序列 (用于 Action Chunking)
            action_chunk = output.get("action_chunk")
            if action_chunk is not None:
                print(f"📦 Action Chunk shape: {action_chunk.shape}")
               
        except Exception as e:
            print(f"❌ 推理步骤出错: {e}")
            break


if __name__ == "__main__":
    run_dummy_test()
