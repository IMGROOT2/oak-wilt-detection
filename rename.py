import os
import glob

def rename_files(input_dir, prefix, buffer=0):
    # Ensure the output directory exists
    os.makedirs(input_dir, exist_ok=True)

    # Gather all image files in the directory
    files = sorted(glob.glob(os.path.join(input_dir, "*.*")))

    for idx, file_path in enumerate(files):
        # Get the file extension
        ext = os.path.splitext(file_path)[1]

        # Build new filename, zero-padded 3 digits
        new_name = f"{prefix}-{(idx + buffer):03d}{ext}"
        new_path = os.path.join(input_dir, new_name)

        # Rename the file
        os.rename(file_path, new_path)
        print(f"Renamed: {file_path} -> {new_path}")

if __name__ == "__main__":
    # Rename files in the processed-healthy directory
    rename_files("dataset/processed-healthy", "Healthy", buffer=0)

    # Rename files in the processed-unhealthy directory
    rename_files("dataset/processed-unhealthy", "Unhealthy", buffer=0)