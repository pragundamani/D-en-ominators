#!/usr/bin/env python3
"""
Quick test script to validate the notebook changes work correctly
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as albu
from albumentations.pytorch import ToTensorV2

# Test parameters
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 4  # Small batch for testing

print("=" * 60)
print("Testing Notebook Implementation")
print("=" * 60)

# ======================
# Test 1: Check directory exists
# ======================
print("\n[Test 1] Checking dataset directory...")
train_data_dir = "finalized_dataset/homogenized/train_flat"
if not os.path.exists(train_data_dir):
    print(f"❌ ERROR: Directory not found: {train_data_dir}")
    sys.exit(1)
print(f"✓ Directory exists: {train_data_dir}")

# ======================
# Test 2: Load files and extract classes
# ======================
print("\n[Test 2] Loading files and extracting classes...")
all_files = [f for f in os.listdir(train_data_dir) if f.endswith('.png')]
if len(all_files) == 0:
    print(f"❌ ERROR: No PNG files found in {train_data_dir}")
    sys.exit(1)
print(f"✓ Found {len(all_files)} PNG files")

# Extract class names
def extract_class_name(filename):
    name = filename.replace('.png', '')
    class_name = name.split('_')[0]
    return class_name

data = []
for filename in all_files:
    class_name = extract_class_name(filename)
    data.append({'file_name': filename, 'label': class_name})

all_df = pd.DataFrame(data)
print(f"✓ Created DataFrame with {len(all_df)} rows")
print(f"  Sample classes: {sorted(all_df['label'].unique())[:10]}")

# ======================
# Test 3: Create class mapping
# ======================
print("\n[Test 3] Creating class mapping...")
unique_classes = sorted(all_df['label'].unique())
class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
num_classes = len(unique_classes)
print(f"✓ Total classes: {num_classes}")
print(f"  Sample mapping: {dict(list(class_to_idx.items())[:5])}")

# Map to numeric indices
all_df['label'] = all_df['label'].map(class_to_idx)
print(f"✓ Mapped labels to numeric indices (0 to {num_classes-1})")

# ======================
# Test 4: Train/test split
# ======================
print("\n[Test 4] Testing train/test split...")
train_df, test_df = train_test_split(
    all_df,
    test_size=0.2,
    random_state=SEED,
    stratify=all_df['label']
)

train_df, val_df = train_test_split(
    train_df,
    test_size=0.125,
    random_state=SEED,
    stratify=train_df['label']
)

print(f"✓ Split completed:")
print(f"  Train: {len(train_df)} samples ({100*len(train_df)/len(all_df):.1f}%)")
print(f"  Val:   {len(val_df)} samples ({100*len(val_df)/len(all_df):.1f}%)")
print(f"  Test:  {len(test_df)} samples ({100*len(test_df)/len(all_df):.1f}%)")

# ======================
# Test 5: CustomDataset class
# ======================
print("\n[Test 5] Testing CustomDataset class...")

class CustomDataset(Dataset):
    def __init__(self, df, data_dir, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(self.data_dir, fname) for fname in df['file_name'].values
        ]
        self.labels = df['label'].values

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        if self.transform is not None:
            image = self.transform(image=image)['image']
        label = self.labels[idx]
        return image, label

# Create transforms
train_transform = albu.Compose([
    albu.HorizontalFlip(p=0.5),
    albu.Resize(IMG_SIZE, IMG_SIZE),
    albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Test with small subset
test_subset = train_df.head(10)
test_dataset = CustomDataset(
    df=test_subset,
    data_dir=train_data_dir,
    transform=train_transform
)

print(f"✓ Created dataset with {len(test_dataset)} samples")

# ======================
# Test 6: Load a batch
# ======================
print("\n[Test 6] Testing DataLoader...")
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0  # Use 0 for testing to avoid multiprocessing issues
)

try:
    for images, labels in test_loader:
        print(f"✓ Successfully loaded batch:")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Labels: {labels.tolist()}")
        print(f"  Image dtype: {images.dtype}")
        print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
        break
except Exception as e:
    print(f"❌ ERROR loading batch: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ======================
# Test 7: Verify file paths exist
# ======================
print("\n[Test 7] Verifying file paths...")
sample_files = train_df['file_name'].head(5).tolist()
all_exist = True
for fname in sample_files:
    path = os.path.join(train_data_dir, fname)
    if os.path.exists(path):
        print(f"  ✓ {fname}")
    else:
        print(f"  ❌ {fname} - NOT FOUND")
        all_exist = False

if not all_exist:
    print("❌ Some files not found!")
    sys.exit(1)

# ======================
# Test 8: Check label distribution
# ======================
print("\n[Test 8] Checking label distribution...")
train_label_counts = train_df['label'].value_counts().sort_index()
val_label_counts = val_df['label'].value_counts().sort_index()
test_label_counts = test_df['label'].value_counts().sort_index()

print(f"✓ Train labels range: {train_df['label'].min()} to {train_df['label'].max()}")
print(f"✓ Val labels range: {val_df['label'].min()} to {val_df['label'].max()}")
print(f"✓ Test labels range: {test_df['label'].min()} to {test_df['label'].max()}")

# Check if all classes are represented
train_classes = set(train_df['label'].unique())
val_classes = set(val_df['label'].unique())
test_classes = set(test_df['label'].unique())
all_classes = set(range(num_classes))

missing_train = all_classes - train_classes
missing_val = all_classes - val_classes
missing_test = all_classes - test_classes

if missing_train:
    print(f"⚠ Warning: {len(missing_train)} classes missing from train set")
else:
    print("✓ All classes present in train set")

if missing_val:
    print(f"⚠ Warning: {len(missing_val)} classes missing from val set")
else:
    print("✓ All classes present in val set")

if missing_test:
    print(f"⚠ Warning: {len(missing_test)} classes missing from test set")
else:
    print("✓ All classes present in test set")

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print(f"\nSummary:")
print(f"  Total images: {len(all_df)}")
print(f"  Total classes: {num_classes}")
print(f"  Train: {len(train_df)} ({100*len(train_df)/len(all_df):.1f}%)")
print(f"  Val:   {len(val_df)} ({100*len(val_df)/len(all_df):.1f}%)")
print(f"  Test:  {len(test_df)} ({100*len(test_df)/len(all_df):.1f}%)")
print("\nNotebook is ready for supercomputer!")

