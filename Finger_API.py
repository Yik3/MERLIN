#!/usr/bin/python3
# -*- coding: utf-8 -*-
import socket
import time
import json

class AoyiHand:
    def __init__(self, hand_side='right'):
        """
        Initialize the Aoyi Hand Controller.
        
        Args:
            hand_side (str): 'left' or 'right'. Determines the IP address.
        """
        # Select IP based on side
        if hand_side.lower() == 'left':
            self.ip = '169.254.128.18'
        else:
            self.ip = '169.254.128.19'  # Default to right
            
        self.port_no = 8080
        
        # Initialize Socket connection
        try:
            self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client.connect((self.ip, self.port_no))
            print(f"Connected to {hand_side} hand at {self.ip}")
        except Exception as e:
            print(f"Connection Failed: {e}")

        # -----------------------------------------------------------
        # STATE VARIABLES
        # We need to record the current position because the motor 
        # commands for bending (indices 0-9) are action-based 
        # (speed/direction), not absolute.
        # Format: [Thumb, Index, Middle, Ring, Little, Thumb_Rotation]
        # Range: 0 (Open/Extended) to 255 (Closed/Inward)
        # -----------------------------------------------------------
        self.current_positions = [0, 0, 0, 0, 0, 0]

        # Initialize Modbus communication mode
        self.get_power_ready()

    def get_power_ready(self):
        """Initializes the Modbus RTU communication mode."""
        cmd = {
            "command": "set_modbus_mode",
            "port": 1,
            "baudrate": 115200,
            "timeout": 2
        }
        # Send as JSON string + newline
        self.send_cmd(json.dumps(cmd) + '\r\n')
        time.sleep(2)
        print("Communication Port Configured to ModbusRTU")

    def send_cmd(self, cmd_str):
        """Sends the encoded command string to the socket."""
        try:
            self.client.send(cmd_str.encode('utf-8'))
            return True
        except Exception as e:
            print(f"Send Error: {e}")
            return False

    def set_hand_6d(self, target_positions):
        """
        Main function to control all fingers using 6D absolute coordinates.
        
        Args:
            target_positions (list): A list of 6 integers [0-255].
            [Thumb_Bend, Index, Middle, Ring, Little, Thumb_Rotation]
            0 = Fully Open / Outward
            255 = Fully Closed / Inward
        """
        if len(target_positions) != 6:
            print("Error: Input must be a list of 6 integers.")
            return

        # 1. Prepare the data array for the Modbus command (12 integers)
        # Structure: 
        # [Thumb_In, Thumb_Out, Index_In, Index_Out, Middle_In, Middle_Out, 
        #  Ring_In, Ring_Out, Little_In, Little_Out, Thumb_Rot, Reserved]
        packet_data = [0] * 12
        
        # 2. Loop through the 5 fingers (indices 0-4 in target_positions)
        # We calculate the difference between Target and Current to decide direction.
        for i in range(5):
            target = target_positions[i]
            current = self.current_positions[i]
            
            # Calculate power/speed based on distance (Simple P-Controller)
            # You might want to clamp this to a max value if 255 is too fast
            diff = int(target - current)
            power = min(abs(diff), 255) # max 255
            
            # Logic: Even index (0,2,4..) is INWARD/CLOSE. Odd (1,3,5..) is OUTWARD/OPEN.
            motor_in_idx = i * 2      # 0, 2, 4, 6, 8
            motor_out_idx = i * 2 + 1 # 1, 3, 5, 7, 9
            
            if diff > 0:
                # Target is greater than current -> Move INWARD (Close)
                packet_data[motor_in_idx] = power
                packet_data[motor_out_idx] = 0
            elif diff < 0:
                # Target is less than current -> Move OUTWARD (Open)
                packet_data[motor_in_idx] = 0
                packet_data[motor_out_idx] = power
            else:
                # Position reached, stop motors
                packet_data[motor_in_idx] = 0
                packet_data[motor_out_idx] = 0

        # 3. Handle Thumb Rotation (Index 5 in target, Index 10 in packet)
        # This axis uses absolute position natively
        packet_data[10] = target_positions[5]
        
        # 4. Construct Command
        cmd_dict = {
            "command": "write_registers",
            "port": 1,
            "address": 1135,
            "num": 6,
            "data": packet_data,
            "device": 2
        }
        
        # 5. Send Command
        cmd_str = json.dumps(cmd_dict) + '\r\n'
        self.send_cmd(cmd_str)
        
        # 6. UPDATE STATE
        # We assume the hand successfully moves to the target positions.
        # In a real closed-loop system, we would read sensors here.
        self.current_positions = list(target_positions)
        
        # Optional: Sleep briefly to allow hardware to process (adjust as needed)
        time.sleep(0.02) 

if __name__ == '__main__':
    # Usage Example
    
    # 1. Initialize Right Hand
    right_hand = AoyiHand(hand_side='left')
    
    # 2. Define a gesture (e.g., Fist)
    # [Thumb, Index, Middle, Ring, Little, Rotation]
    # All 255 -> Closed fist
    fist_gesture = [0, 255, 255, 255, 255, 0]
    
    # 3. Define a gesture (e.g., Open Palm)
    # All 0 -> Open hand
    open_gesture = [0, 0, 0, 0, 0, 125]
    
    print("Closing Hand...")
    # Because we are simulating position control on velocity hardware,
    # we might need to send the command a few times or use a loop 
    # if the hardware stops moving after a single packet.
    right_hand.set_hand_6d(fist_gesture)
    time.sleep(1)
    
    print("Opening Hand...")
    right_hand.set_hand_6d(open_gesture)
    time.sleep(1)