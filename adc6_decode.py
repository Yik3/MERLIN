import serial
import struct
import time
import csv
import pytz
from datetime import datetime

SERIAL_PORT = 'COM8'  
BAUD_RATE = 115200    


TARGET_TZ = pytz.timezone('America/Los_Angeles') 
# ===========================================

def run_logger():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"connected to {SERIAL_PORT} @ {BAUD_RATE}bps")
        print(f"TimeZone: {TARGET_TZ}")
    except Exception as e:
        print(f"❌ Can't open port: {e}")
        return

    start_time = datetime.now(TARGET_TZ)
    filename = f"adc_data_{start_time.strftime('%Y%m%d_%H%M%S')}.csv"
    
    print("-" * 75)
    print(f"{'Timestamp':<28} | {'CH0':<6} {'CH1':<6} {'CH2':<6} {'CH3':<6} {'CH4':<6} {'CH5':<6}")
    print("-" * 75)

    try:
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp (ISO)', 'CH0', 'CH1', 'CH2', 'CH3', 'CH4', 'CH5'])

            while True:
                if ser.read(1) != b'\xAA': continue
                if ser.read(1) != b'\xBB': continue
                
                data_bytes = ser.read(12)
                
                if len(data_bytes) == 12:
                    adc_values = struct.unpack('<HHHHHH', data_bytes)
                    
                    now = datetime.now(TARGET_TZ)
                    
                    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + now.strftime("%z")
                    
                    print_time = now.strftime("%H:%M:%S.%f")[:-3]
                    print(f"[{print_time}] | {adc_values[0]:4d}   {adc_values[1]:4d}   {adc_values[2]:4d}   {adc_values[3]:4d}   {adc_values[4]:4d}   {adc_values[5]:4d}")
                    
                    writer.writerow([timestamp_str, *adc_values])
                    
    except KeyboardInterrupt:
        print(f"\n🛑 Saved data to: {filename}")
    finally:
        ser.close()

if __name__ == '__main__':
    run_logger()