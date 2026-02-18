import sys
import os

# 1. 打印当前工作目录
print(f"Current Working Directory: {os.getcwd()}")
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
# 2. 尝试导入 train
try:
    import train
    print(f"\n[SUCCESS] Successfully imported 'train'")
    print(f"Type of 'train': {type(train)}")
    print(f"Location of 'train': {getattr(train, '__file__', 'No __file__ attribute (Is this a folder without __init__.py?)')}")
    print(f"Path of 'train': {getattr(train, '__path__', 'No __path__ attribute')}")
except ImportError as e:
    print(f"\n[ERROR] Could not import 'train': {e}")

# 3. 检查文件夹结构
train_dir = os.path.join(os.getcwd(), "train")
if os.path.isdir(train_dir):
    print(f"\n[CHECK] 'train' directory exists at: {train_dir}")
    init_file = os.path.join(train_dir, "__init__.py")
    if os.path.exists(init_file):
        print(f"[OK] Found __init__.py at: {init_file}")
    else:
        print(f"[MISSING] NO __init__.py found in {train_dir} <-- THIS IS LIKELY THE PROBLEM")
else:
    print(f"\n[ERROR] 'train' directory DOES NOT exist at {train_dir}")

