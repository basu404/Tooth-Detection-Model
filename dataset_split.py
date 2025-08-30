import os
import shutil
import random
from pathlib import Path

def split_dataset(images_folder, labels_folder, output_folder, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    output_path = Path(output_folder)
    for split in ['train', 'val', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    images_path = Path(images_folder)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in images_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    labels_path = Path(labels_folder)
    paired_files = []
    
    for img_file in image_files:
        label_file = labels_path / f"{img_file.stem}.txt"
        if label_file.exists():
            paired_files.append((img_file, label_file))
        else:
            print(f"Warning: No label file found for {img_file.name}")
    
    print(f"Found {len(paired_files)} image-label pairs")
    
    random.shuffle(paired_files)
    
    total_files = len(paired_files)
    train_size = int(total_files * train_ratio)
    val_size = int(total_files * val_ratio)
    
    train_files = paired_files[:train_size]
    val_files = paired_files[train_size:train_size + val_size]
    test_files = paired_files[train_size + val_size:]
    
    print(f"Train: {len(train_files)} files")
    print(f"Val: {len(val_files)} files") 
    print(f"Test: {len(test_files)} files")
    
    def copy_files(file_pairs, split_name):
        for img_file, label_file in file_pairs:
            shutil.copy2(img_file, output_path / split_name / 'images' / img_file.name)
            shutil.copy2(label_file, output_path / split_name / 'labels' / label_file.name)
    
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')
    
    print("Dataset split completed successfully!")

if __name__ == "__main__":
    images_folder = "./images"
    labels_folder = "./labels"
    output_folder = "./datasets/tooth_detection"
    
    random.seed(42)
    
    split_dataset(images_folder, labels_folder, output_folder)