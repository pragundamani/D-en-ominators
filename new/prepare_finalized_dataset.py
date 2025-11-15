"""
Prepare finalized dataset for training by creating a flat structure.
"""

from pathlib import Path
import shutil
from tqdm import tqdm


def flatten_dataset_for_training(source_dir, target_dir):
    """
    Flatten nested dataset structure to match training script expectations.
    Creates a flat structure: target_dir/GardinerNumber.png
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all images with their Gardiner numbers
    images = []
    for gardiner_dir in source_dir.iterdir():
        if gardiner_dir.is_dir():
            gardiner_num = gardiner_dir.name
            for img_file in gardiner_dir.glob("*.png"):
                images.append((gardiner_num, img_file))
    
    print(f"Found {len(images)} images in {len([d for d in source_dir.iterdir() if d.is_dir()])} classes")
    
    # Copy images with unique names
    copied = {}
    for gardiner_num, img_file in tqdm(images, desc="Flattening dataset"):
        # Create filename: GardinerNumber_counter.png
        base_name = f"{gardiner_num}.png"
        if base_name in copied:
            counter = copied[base_name]
            copied[base_name] = counter + 1
            base_name = f"{gardiner_num}_{counter}.png"
        else:
            copied[base_name] = 1
        
        dest = target_dir / base_name
        shutil.copy2(img_file, dest)
    
    print(f"✓ Flattened {len(images)} images to {target_dir}")
    return len(images)


if __name__ == "__main__":
    base = Path("/Users/aayankhare/Desktop/D-en-ominators")
    
    # Flatten train set
    print("Flattening train set...")
    train_source = base / "finalized_dataset" / "homogenized" / "train"
    train_target = base / "finalized_dataset" / "homogenized" / "train_flat"
    flatten_dataset_for_training(train_source, train_target)
    
    # Flatten test set
    print("\nFlattening test set...")
    test_source = base / "finalized_dataset" / "homogenized" / "test"
    test_target = base / "finalized_dataset" / "homogenized" / "test_flat"
    flatten_dataset_for_training(test_source, test_target)
    
    print("\n✓ Dataset preparation complete!")

