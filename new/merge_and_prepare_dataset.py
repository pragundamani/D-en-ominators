"""
Merge and prepare finalized dataset from multiple sources.
"""

import argparse
from pathlib import Path
import shutil
from collections import defaultdict
import json
from tqdm import tqdm


def extract_gardiner_number(filename):
    """Extract Gardiner number from filename."""
    # Try to extract from patterns like "M17 (35).png", "I10.png", "N35 (17).png"
    name = Path(filename).stem
    
    # Remove parentheses and numbers after them
    if '(' in name:
        name = name.split('(')[0].strip()
    
    # Extract Gardiner number (e.g., "M17", "I10", "N35")
    # Gardiner numbers are typically 1-3 characters followed by optional numbers
    import re
    match = re.match(r'^([A-Za-z]+[0-9]+[a-z]*)', name)
    if match:
        return match.group(1)
    
    return None


def organize_archaehack_data(source_dir, target_dir, is_test=False):
    """
    Organize Archaehack data by Gardiner number.
    
    Args:
        source_dir: Source directory (archive or test)
        target_dir: Target directory in finalized dataset
        is_test: Whether this is test data
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    gardiner_files = defaultdict(list)
    
    # Process files in root
    for img_file in source_dir.glob("*.png"):
        gardiner = extract_gardiner_number(img_file.name)
        if gardiner:
            gardiner_files[gardiner].append(img_file)
        else:
            # If no Gardiner number found, use filename stem
            gardiner_files[img_file.stem].append(img_file)
    
    # Process subdirectories (organized by Gardiner number)
    for subdir in source_dir.iterdir():
        if subdir.is_dir():
            gardiner = subdir.name
            for img_file in subdir.glob("*.png"):
                gardiner_files[gardiner].append(img_file)
    
    # Copy files to organized structure
    for gardiner, files in gardiner_files.items():
        if is_test:
            # For test, keep in test directory
            dest_dir = target_dir / "test" / gardiner
        else:
            # For archive, put in train directory
            dest_dir = target_dir / "train" / gardiner
        
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        for img_file in files:
            # Create unique filename
            dest_file = dest_dir / img_file.name
            if dest_file.exists():
                # Add counter if file exists
                counter = 1
                while dest_file.exists():
                    stem = img_file.stem
                    suffix = img_file.suffix
                    dest_file = dest_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
            
            shutil.copy2(img_file, dest_file)
    
    return len(gardiner_files), sum(len(files) for files in gardiner_files.values())


def merge_datasets():
    """Merge all datasets into finalized_dataset."""
    base_dir = Path("/Users/aayankhare/Desktop/D-en-ominators")
    finalized_dir = base_dir / "finalized_dataset"
    
    # Create finalized dataset structure
    finalized_dir.mkdir(exist_ok=True)
    (finalized_dir / "train").mkdir(exist_ok=True)
    (finalized_dir / "test").mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Merging Datasets")
    print("=" * 60)
    
    # 1. Copy already homogenized archaeohack-starterpack data
    print("\n1. Copying archaeohack-starterpack data...")
    starterpack_homogenized = base_dir / "datasets" / "archaeohack-starterpack" / "data" / "homogenized"
    
    if starterpack_homogenized.exists():
        # Copy utf-pngs to train
        utf_pngs = starterpack_homogenized / "utf-pngs"
        if utf_pngs.exists():
            for img_file in tqdm(utf_pngs.glob("*.png"), desc="  Copying utf-pngs"):
                gardiner = img_file.stem
                dest_dir = finalized_dir / "train" / gardiner
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_file, dest_dir / img_file.name)
        
        # Copy me-sign-examples-pjb to train
        me_sign = starterpack_homogenized / "me-sign-examples-pjb"
        if me_sign.exists():
            for subdir in me_sign.iterdir():
                if subdir.is_dir():
                    gardiner = subdir.name
                    dest_dir = finalized_dir / "train" / gardiner
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    for img_file in tqdm(subdir.glob("*.png"), desc=f"  Copying {gardiner}"):
                        dest_file = dest_dir / img_file.name
                        if dest_file.exists():
                            counter = 1
                            while dest_file.exists():
                                dest_file = dest_dir / f"{img_file.stem}_{counter}{img_file.suffix}"
                                counter += 1
                        shutil.copy2(img_file, dest_file)
    
    # 2. Organize and copy Archaehack/archive data
    print("\n2. Organizing Archaehack/archive data...")
    archive_dir = base_dir / "Archaehack" / "archive"
    if archive_dir.exists():
        classes, files = organize_archaehack_data(
            archive_dir,
            finalized_dir,
            is_test=False
        )
        print(f"  ✓ Organized {files} files into {classes} classes")
    
    # 3. Organize and copy Archaehack/test data
    print("\n3. Organizing Archaehack/test data...")
    test_dir = base_dir / "Archaehack" / "test"
    if test_dir.exists():
        classes, files = organize_archaehack_data(
            test_dir,
            finalized_dir,
            is_test=True
        )
        print(f"  ✓ Organized {files} files into {classes} classes")
    
    # Count final dataset
    train_files = sum(1 for _ in (finalized_dir / "train").rglob("*.png"))
    test_files = sum(1 for _ in (finalized_dir / "test").rglob("*.png"))
    train_classes = len([d for d in (finalized_dir / "train").iterdir() if d.is_dir()])
    test_classes = len([d for d in (finalized_dir / "test").iterdir() if d.is_dir()])
    
    print("\n" + "=" * 60)
    print("Dataset Merge Complete")
    print("=" * 60)
    print(f"Train: {train_files} images in {train_classes} classes")
    print(f"Test: {test_files} images in {test_classes} classes")
    print(f"Total: {train_files + test_files} images")
    print(f"\nFinalized dataset location: {finalized_dir}")
    
    return finalized_dir


if __name__ == "__main__":
    merge_datasets()

