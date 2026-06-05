"""
Consolidate and Clean Multi-Crop Dataset
----------------------------------------
This script takes the downloaded Kaggle datasets and existing data, and 
consolidates them into a clean, single-level structure for train/val splitting later.
It does NOT perform data augmentation. It only gathers real, existing images.

Expected Target Structure:
data/
└── dataset_cleaned/
    ├── Apple___Apple_scab/
    ├── Apple___Black_rot/
    ├── Tomato___Bacterial_spot/
    └── ...
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Configurations
KAGGLE_RAW_DIR = Path(r"C:\Projects\Plant project\data\raw_downloads\new_plant_diseases")
EXISTING_TOMATO_DIR = Path(r"C:\Projects\Plant project\data\tomato")
OUTPUT_DIR = Path(r"C:\Projects\Plant project\data\dataset_cleaned")

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.JPG', '.PNG')

def get_image_files(directory):
    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(directory.glob(f"**/*{ext}"))
    return files

import concurrent.futures

def copy_image(img, target_class_dir):
    out_path = target_class_dir / img.name
    if not out_path.exists():
        shutil.copy2(img, out_path)
        return True
    return False

def main():
    print("="*60)
    print("DATASET CLEANING & CONSOLIDATION")
    print("="*60)
    
    if OUTPUT_DIR.exists():
        print("Cleaning old output directory...")
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    stats = defaultdict(int)

    # 2. Process Kaggle 'new-plant-diseases-dataset'
    print("\nScanning Kaggle downloaded folders...")
    kaggle_source_folders = []
    for root, dirs, files in os.walk(KAGGLE_RAW_DIR):
        for d in dirs:
            if "___" in d:
                dir_path = Path(root) / d
                if any(f.endswith(IMAGE_EXTENSIONS) for f in os.listdir(dir_path)):
                    kaggle_source_folders.append(dir_path)

    print(f"Found {len(kaggle_source_folders)} class folders in Kaggle dataset.")
    
    print("\nConsolidating Kaggle images...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        for folder in tqdm(kaggle_source_folders, desc="Processing Kaggle Folders"):
            class_name = folder.name.replace(" ", "_")
            target_class_dir = OUTPUT_DIR / class_name
            target_class_dir.mkdir(parents=True, exist_ok=True)
            
            images = get_image_files(folder)
            
            # Submit copies
            futures = [executor.submit(copy_image, img, target_class_dir) for img in images]
            
            # Tally stats
            for f in concurrent.futures.as_completed(futures):
                if f.result():
                    stats[class_name] += 1

    # 3. Process existing old tomato dataset
    print("\nScanning existing original Tomato dataset...")
    if EXISTING_TOMATO_DIR.exists():
        existing_classes = [d for d in EXISTING_TOMATO_DIR.iterdir() if d.is_dir()]
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            for folder in tqdm(existing_classes, desc="Processing Old Tomato"):
                raw_class_name = folder.name
                if raw_class_name.lower() == "healthy":
                    standard_name = "Tomato___healthy"
                else:
                    standard_name = f"Tomato___{raw_class_name}"
                    
                target_class_dir = OUTPUT_DIR / standard_name
                target_class_dir.mkdir(parents=True, exist_ok=True)
                
                images = get_image_files(folder)
                futures = [executor.submit(copy_image, img, target_class_dir) for img in images]
                for f in concurrent.futures.as_completed(futures):
                    if f.result():
                        stats[standard_name] += 1
    else:
        print(f"Warning: Did not find existing tomato directory at {EXISTING_TOMATO_DIR}")

    # 4. Print Statistics
    print("\n" + "="*60)
    print("CONSOLIDATION COMPLETE - RAW STATS")
    print("="*60)
    
    total_images = 0
    # Group by crop
    crops = defaultdict(int)
    
    for class_name, count in sorted(stats.items()):
        crop = class_name.split("___")[0]
        crops[crop] += count
        total_images += count
        
    for crop, count in sorted(crops.items()):
        print(f"{crop.ljust(20)} : {count:,} images")
        
    print("-"*60)
    print(f"TOTAL UNIQUE CLASSES: {len(stats)}")
    print(f"TOTAL REAL IMAGES : {total_images:,}")
    print("="*60)
    
if __name__ == "__main__":
    main()
