import numpy as np

MAX_VAL = 65535
finger_range_map = {
    "CH0": [1740, 1801, 2075, 2360, 2595, 2750],
    "CH1": [2050, 2129, 2240, 2390, 2480, 2500],
    "CH2": [1100, 1260, 1480, 1650, 1849, 2055],
    "CH3": [1191, 1349, 1478, 1666, 1837, 2064],
    "CH4": [1245, 1353, 1535, 1760, 1970, 2208],
    "CH5": [1142, 1273, 1465, 1648, 1825, 2130]
}

monotonicity_map = {
    "CH0": 0,
    "CH1": 1,
    "CH2": 0,
    "CH3": 0,
    "CH4": 0,
    "CH5": 0
}

def map_encoder_to_motor(encoder_vals, mode="Linear", gain=1.1):
    """
    将 6D Encoder 数组转化为 Motor 信号，支持 Linear 分段插值和 Regression 二次回归。
    
    参数:
    encoder_vals (list/array): 长度为 6 的 Encoder 读数列表
    mode (str): "Linear" 或 "Regression"
    gain (float): Motor 控制信号的增益缩放系数 (默认 0.9)
    
    返回:
    list: 根据硬件映射重排后的 6D Motor 控制列表
    """
    motor_vals = []
    
    for i in range(6):
        ch_name = f'CH{i}'
        
        if ch_name in finger_range_map:
            encoder_points = finger_range_map[ch_name]
            mono = monotonicity_map[ch_name]
            
            # 1. 限制输入范围 (Clip to range) 避免越界
            # 字典中的 range_map 从小到大排列，所以索引 0 是最小值，-1 是最大值
            min_enc = encoder_points[0]
            max_enc = encoder_points[-1]
            clipped_encoder_val = max(min(encoder_vals[i], max_enc), min_enc)
            
            # 2. 生成对应的 Motor Reference Points，并应用 gain 缩放
            # 分布在 0 到 (gain * MAX_VAL) 之间
            motor_points = [gain * MAX_VAL / 5 * j for j in range(6)]
            
            # 3. 根据单调性调整映射方向 (0 = Decreasing, 即输入越小输出越大)
            if mono == 0:
                motor_points = motor_points[::-1]
                
            # 4. 执行映射计算
            if mode == "Linear":
                # np.interp 执行分段线性映射
                mapped_val = np.interp(clipped_encoder_val, encoder_points, motor_points)
                
            elif mode == "Regression":
                # 拟合二次函数
                coeffs = np.polyfit(encoder_points, motor_points, 2)
                mapped_val = np.polyval(coeffs, clipped_encoder_val)
                # 为防止多项式在端点微小波动导致越界，强制限制在 [0, gain * MAX_VAL]
                mapped_val = max(0, min(gain * MAX_VAL, mapped_val))
                
            else:
                raise ValueError("Mode must be either 'Linear' or 'Regression'")
                
            motor_vals.append(int(round(mapped_val)))
            
        else:
            motor_vals.append(0)  # 如果字典里找不到对应通道，默认归零
            
    # 5. 根据硬件接线进行位置重排 (Reorder)
    ret_motor_vals = [
        motor_vals[1], 
        motor_vals[2], 
        motor_vals[4], 
        motor_vals[3], 
        motor_vals[5], 
        motor_vals[0]
    ]
    
    return ret_motor_vals
