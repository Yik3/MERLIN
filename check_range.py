# Read the csv, plot the data, and find min and max values for each finger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def read_csv_and_plot(file_path):
    # 1. 读取 CSV 文件
    df = pd.read_csv(file_path)
    
    # 2. 打印前几行数据以确认正确读取
    print("数据预览:")
    print(df.head())
    
    # 3. 提取每个手指的数据列 (假设列名包含 'CH')
    finger_columns = [col for col in df.columns if 'CH' in col]
    
    # 4. 绘制每个手指的曲线
    plt.figure(figsize=(12, 8))
    for col in finger_columns:
        plt.plot(df[col], label=col)
    
    plt.title("Finger Sensor Data Over Time")
    plt.xlabel("Sample Index")
    plt.ylabel("Sensor Value")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 5. 找到每个手指的最小值和最大值
    for col in finger_columns:
        min_val = df[col].min()
        max_val = df[col].max()
        average_val = df[col].mean()
        print(f"{col}: Min = {min_val}, Max = {max_val}, Average = {average_val}")

if __name__ == "__main__":
    # 替换为你的 CSV 文件路径
    csv_file_path = 'adc_data_20260207203111.csv'
    read_csv_and_plot(csv_file_path)

# close
'''
CH0-ThumbLow: Min = 1280, Max = 1410, Average = 1286.424117647059
CH1-ThumbUp: Min = 1707, Max = 1968, Average = 1856.1176470588234 #1413
CH2-Pointer: Min = 2878, Max = 2918, Average = 2883.1170588235295
CH3-Middle: Min = 2738, Max = 2749, Average = 2742.4194117647057
CH4-Ring: Min = 3988, Max = 4095, Average = 4039.59
CH5-Pinky: Min = 2657, Max = 2972, Average = 2667.6188235294117
'''

# open
'''
CH0-ThumbLow: Min = 2354, Max = 2479, Average = 2359.640776699029
CH1-ThumbUp: Min = 1678, Max = 1899, Average = 1826.2705207413944 TAKE 1840
CH2-Pointer: Min = 2049, Max = 2104, Average = 2074.978375992939
CH3-Middle: Min = 1746, Max = 1926, Average = 1902.5445719329214
CH4-Ring: Min = 4020, Max = 4095, Average = 4034.3764342453665
CH5-Pinky: Min = 1938, Max = 2180, Average = 1971.9015887025596
'''