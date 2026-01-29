#!/usr/bin/python3
# -*- coding: utf-8 -*-
import socket
import time
import json
import numpy as np
import matplotlib.pyplot as plt

# 寄存器配置
BASE_ADDR_ANGLE = 1165  # 角度起始地址 (ROH_FINGER_ANGLE0)
NODE_ID = 2
FINGER_NAMES = ["Thumb (Bend)", "Index", "Middle", "Ring", "Little", "Thumb (Rot)"]

class AoyiHandReader:
    def __init__(self, ip='169.254.128.19', port=8080):
        self.ip = ip
        self.port = port
        self.client = None

    def connect(self):
        """建立连接并初始化 Modbus 模式"""
        try:
            self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client.settimeout(2.0)
            self.client.connect((self.ip, self.port))
            
            # 初始化命令
            cmd = '{"command":"set_modbus_mode","port":1,"baudrate":115200,"timeout":2}\r\n'
            self.client.send(cmd.encode('utf-8'))
            time.sleep(0.5)
            # 清空一下缓冲区
            try:
                self.client.recv(1024)
            except socket.timeout:
                pass
            print(f">>> 已连接到灵巧手 ({self.ip})")
            return True
        except Exception as e:
            print(f"连接失败: {e}")
            return False

    def close(self):
        if self.client:
            self.client.close()
            print(">>> 连接已关闭")

    def _parse_angle(self, raw_val):
        """将 uint16 原始值转换为实际角度 (float)"""
        if raw_val is None:
            return np.nan # 用 NaN 标记读取失败
            
        # 补码处理：大于 32767 的视为负数
        if raw_val > 32767:
            raw_val -= 65536
            
        return raw_val / 100.0

    def _read_chunk(self):
        """一次性读取 6 个寄存器 (如果网关支持) 或 循环读取"""
        # 尝试一次性读取 6 个寄存器 (地址 1165, 数量 6)
        # 这样比循环读 6 次快得多，延迟更低
        cmd = {
            "command": "read_holding_registers",
            "port": 1,
            "address": BASE_ADDR_ANGLE,
            "num": 6, 
            "device": NODE_ID
        }
        
        try:
            self.client.send((json.dumps(cmd) + '\r\n').encode('utf-8'))
            resp_str = self.client.recv(1024).decode('utf-8')
            
            data_json = json.loads(resp_str)
            if "data" in data_json:
                raw_data = data_json["data"]
                # 确保返回的是列表且长度足够
                if isinstance(raw_data, list) and len(raw_data) >= 6:
                    # 转换所有数据
                    return [self._parse_angle(x) for x in raw_data[:6]]
        except Exception as e:
            print(f"读取错误: {e}") # 调试时可打开
            pass
            
        return [np.nan] * 6 # 读取失败返回全 NaN

def record_angles(ip='169.254.128.19', interval=0.05, plot=True):
    """
    持续读取灵巧手角度数据。
    
    Args:
        ip (str): 灵巧手 IP 地址
        interval (float): 采样间隔 (秒)
        plot (bool): 是否在结束时画图

    Returns:
        np.array: N x 6 的数据矩阵
    """
    hand = AoyiHandReader(ip)
    
    if not hand.connect():
        return np.array([])

    data_log = []
    start_time = time.time()
    
    print("\n" + "="*50)
    print("开始录制数据... 请按 Ctrl + C 停止")
    print("="*50)

    try:
        while True:
            # 1. 读取当前时刻的一帧数据 (6个角度)
            angles = hand._read_chunk()
            
            # 2. 存入列表
            data_log.append(angles)
            
            # 3. 打印实时状态 (可选，仅打印第一行覆盖刷新，避免刷屏)
            # 使用 \r 回到行首
            print(f"\r正在录制... 样本数: {len(data_log)} | Thumb: {angles[0]:.2f}° | Index: {angles[1]:.2f}°", end="", flush=True)
            
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\n>>> 捕获 Ctrl+C，停止录制。")
    
    finally:
        hand.close()

    # 转换为 Numpy 数组 (N x 6)
    data_array = np.array(data_log)
    
    # 去除包含 NaN 的行 (可选，看你是否需要保留丢包时刻)
    # data_array = data_array[~np.isnan(data_array).any(axis=1)]

    print(f"录制完成。总帧数: {data_array.shape[0]}")

    if plot and data_array.shape[0] > 0:
        _plot_data(data_array)

    return data_array

def _plot_data(data):
    """内部绘图函数"""
    print("正在绘图...")
    
    # 创建时间轴 (假设采样是均匀的，仅用于大概展示)
    x_axis = np.arange(len(data))
    
    plt.figure(figsize=(12, 8))
    plt.suptitle("Aoyi Hand Motor Angles Over Time", fontsize=16)

    # 创建 2x3 的子图布局
    for i in range(6):
        ax = plt.subplot(2, 3, i + 1)
        
        # 绘制曲线，处理 NaN 数据 matplotlib 会自动断开线条
        ax.plot(x_axis, data[:, i], label=FINGER_NAMES[i], color=f'C{i}')
        
        ax.set_title(FINGER_NAMES[i])
        ax.set_ylabel("Angle (deg)")
        ax.set_xlabel("Sample Frame")
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # 简单显示该手指的最大最小值
        # 使用 nanmin/nanmax 防止 NaN 导致报错
        try:
            min_val = np.nanmin(data[:, i])
            max_val = np.nanmax(data[:, i])
            ax.text(0.05, 0.9, f"Range: [{min_val:.1f}, {max_val:.1f}]", 
                    transform=ax.transAxes, fontsize=9, 
                    bbox=dict(facecolor='white', alpha=0.7))
        except:
            pass

    plt.tight_layout()
    plt.show()

# --- 主程序入口 ---
if __name__ == "__main__":
    # 调用函数，plot=True 会在结束时画图
    angles_history = record_angles(plot=True)
    
    # 打印前 5 行数据供检查
    if len(angles_history) > 0:
        print("\n数据预览 (前5帧):")
        print(angles_history[:5])


