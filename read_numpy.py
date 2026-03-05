import numpy as np

# Read a npy file from a path and output the data
def read_npy_file(file_path):
    try:
        data = np.load(file_path, allow_pickle=True)
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
    
if __name__ == "__main__":
    file_path = '/home/classysh/MERLIN/MERLIN/data/222data/encoder/processed_encoder_t/adc_data_20260222220212.npy'  # Example file path
    data = read_npy_file(file_path)
    if data is not None:
        print("Data contents:")
        print(data) 