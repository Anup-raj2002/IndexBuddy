import os
import shutil
import kagglehub

print("[INFO] Downloading retail sales dataset from Kaggle...")
cache_path = kagglehub.dataset_download("lindacheruto/retail-sales-dataset")
print(f"[INFO] Kaggle cache path: {cache_path}")

base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(base_dir, "dataset")
os.makedirs(dataset_dir, exist_ok=True)

# Recursively copy all files from the download cache into your dataset folder
copied_count = 0
for root, dirs, files in os.walk(cache_path):
    for file in files:
        src_path = os.path.join(root, file)
        dest_path = os.path.join(dataset_dir, file)
        
        shutil.copy(src_path, dest_path)
        print(f"[SUCCESS] Copied: {file} -> {dataset_dir}")
        copied_count += 1

print(f"[COMPLETE] Successfully saved {copied_count} file(s) to the dataset folder.")