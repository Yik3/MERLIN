#!/usr/bin/python3
# -*- coding: utf-8 -*-
import socket
import time
import json

# 寄存器基地址
BASE_ADDR_STATUS = 1085  # 状态起始地址
BASE_ADDR_ANGLE  = 1165  # 角度起始地址
NODE_ID = 2

class AoyiHand():
    def __init__(self):
        ip = '169.254.128.19'
        port_no = 8080
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect((ip, port_no))
        self.client.settimeout(2) # 设置超时防止卡死
        self.get_power_ready()

    def send_cmd(self, cmd_str=''):
        try:
            self.client.send(cmd_str.encode('utf-8'))
            response = self.client.recv(1024).decode('utf-8')
            return response
        except Exception as e:
            print(f"Socket Error: {e}")
            return None

    def get_power_ready(self):
        cmd = '{"command":"set_modbus_mode","port":1,"baudrate":115200,"timeout":2}\r\n'
        self.send_cmd(cmd)
        time.sleep(0.5)
        print(">>> 模式已配置为 ModbusRTU")

    def read_single_register(self, address):
        """
        读取单个寄存器
        返回: int 数值 (如果是 None 代表读取失败)
        """
        cmd = {
            "command": "read_holding_registers",
            "port": 1,
            "address": address,
            "num": 1,  # 重点：每次只读 1 个
            "device": NODE_ID
        }
        
        resp_str = self.send_cmd(json.dumps(cmd) + '\r\n')
        
        if resp_str:
            try:
                # 假设返回格式例如: {"data":[12345], "status":0} 或者直接 {"data": 12345}
                # 根据你的反馈，这里可能直接由网关处理成某种格式，我们做个兼容处理
                data_json = json.loads(resp_str)
                
                if "data" in data_json:
                    val = data_json["data"]
                    # 如果返回的是 list [12345]，取第一个；如果是 int 12345，直接用
                    if isinstance(val, list):
                        return val[0]
                    return int(val)
                else:
                    print(f"数据异常: {resp_str}")
            except Exception as e:
                print(f"解析错误: {e}, 原始数据: {resp_str}")
        return None

    def parse_angle(self, raw_val):
        """
        将寄存器读到的 int (0~65535) 转换为 实际角度 (float)
        处理负数补码: 例如 65535 -> -0.01 度
        """
        if raw_val is None:
            return 0.0
            
        # 处理 int16 符号位 (补码转换)
        if raw_val > 32767:
            raw_val -= 65536
            
        # 协议规定：实际角度 = 寄存器值 / 100
        return raw_val / 100.0

    def get_all_motors_info(self):
        """
        轮询 6 个马达，分别读取状态和角度
        """
        finger_names = ["大拇指(弯)", "食指", "中指", "无名指", "小指", "大拇指(转)"]
        
        print("\n" + "="*50)
        print(f"{'ID':<4} | {'手指名称':<10} | {'原始Angle(int)':<15} | {'实际角度(deg)':<15} | {'状态码'}")
        print("-" * 50)

        for i in range(6):
            # 1. 计算地址
            addr_angle = BASE_ADDR_ANGLE + i   # 1165 + i
            addr_status = BASE_ADDR_STATUS + i # 1085 + i
            
            # 2. 读取数据 (Int)
            raw_angle = self.read_single_register(addr_angle)
            raw_status = self.read_single_register(addr_status)
            
            # 3. 数据处理
            if raw_angle is not None:
                real_angle = self.parse_angle(raw_angle)
            else:
                raw_angle = "ERR"
                real_angle = 0.0

            if raw_status is None:
                raw_status = "ERR"

            # 4. 打印结果
            print(f"{i:<4} | {finger_names[i]:<10} | {str(raw_angle):<15} | {real_angle:<15.2f} | {raw_status}")
            
            # 稍微延时一点点，防止请求发太快网关处理不过来
            time.sleep(0.02) 
        print("="*50 + "\n")

if __name__ == '__main__':
    hand = AoyiHand()
    
    # 循环监控几次看看数据变不变
    for _ in range(3):
        hand.get_all_motors_info()
        time.sleep(1)


