import os
import shutil
from pathlib import Path

def organize_files(source_dir):
    # Convert string input to a Path object
    source_path = Path(source_dir)

    # Check if directory exists
    if not source_path.exists() or not source_path.is_dir():
        print(f"Error: The directory '{source_dir}' does not exist.")
        return

    # Define destination folders
    npy_dest = source_path / "npy_files"
    mp4_dest = source_path / "mp4_files"

    # Create destinations if they don't exist
    npy_dest.mkdir(exist_ok=True)
    mp4_dest.mkdir(exist_ok=True)

    # Counters for feedback
    npy_count = 0
    mp4_count = 0

    # Iterate through files in the source directory
    for file_path in source_path.iterdir():
        # Skip if it's a directory
        if file_path.is_dir():
            continue

        # Check extensions and move files
        if file_path.suffix.lower() == '.npy':
            try:
                shutil.move(str(file_path), str(npy_dest / file_path.name))
                npy_count += 1
                print(f"Moved: {file_path.name} -> npy_files/")
            except Exception as e:
                print(f"Error moving {file_path.name}: {e}")

        elif file_path.suffix.lower() == '.mp4':
            try:
                shutil.move(str(file_path), str(mp4_dest / file_path.name))
                mp4_count += 1
                print(f"Moved: {file_path.name} -> mp4_files/")
            except Exception as e:
                print(f"Error moving {file_path.name}: {e}")

    # Final summary
    print("\n--- Summary ---")
    print(f"Moved {npy_count} .npy files.")
    print(f"Moved {mp4_count} .mp4 files.")
    print("Done.")

if __name__ == "__main__":

    target_dir = '../data/211data/camera'
    organize_files(target_dir)