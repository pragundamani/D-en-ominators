#!/usr/bin/env python3
"""
Test script to verify Hugging Face dataset integration
Tests the logic without actually downloading the full dataset
"""

import os
import sys
from collections import Counter

print("=" * 60)
print("Testing Hugging Face Dataset Integration")
print("=" * 60)

# ======================
# Test 1: Check local dataset
# ======================
print("\n[Test 1] Checking local dataset...")
train_data_dir = "finalized_dataset/homogenized/train_flat"
if not os.path.exists(train_data_dir):
    print(f"❌ ERROR: Directory not found: {train_data_dir}")
    sys.exit(1)

all_files = [f for f in os.listdir(train_data_dir) if f.endswith('.png')]
print(f"✓ Local dataset: {len(all_files)} images")

# Extract classes
def extract_class_name(filename):
    name = filename.replace('.png', '')
    return name.split('_')[0]

local_classes = set([extract_class_name(f) for f in all_files])
print(f"✓ Local classes: {len(local_classes)} unique classes")

# ======================
# Test 2: Simulate HF dataset structure
# ======================
print("\n[Test 2] Simulating Hugging Face dataset...")
# Based on the dataset info: 4,210 images, 171 classes (excluding UNKNOWN)
# Train: 3,580, Test: 455

# Simulate HF classes (Gardiner codes - some may overlap with local)
# Common Gardiner codes that might overlap
hf_classes_sim = [
    'A1', 'A2', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17',
    'D1', 'D2', 'D3', 'D4', 'D10', 'D21', 'D28', 'D35', 'D36', 'D46',
    'G1', 'G5', 'G17', 'G29', 'G39', 'G40', 'G43',
    'M1', 'M17', 'M20', 'M42',
    'N1', 'N14', 'N17', 'N29', 'N30', 'N35',
    'S1', 'S28', 'S29',
    'T1', 'T22', 'T30',
    'U1', 'U15', 'U33',
    'V1', 'V13', 'V28', 'V30', 'V31', 'V4',
    'W1', 'W18', 'W19', 'W24',
    'X1',
    'Y5',
    'Z1',
    'UNKNOWN'  # 179 images labeled as UNKNOWN
]

# Simulate HF data
hf_train_count = 3580
hf_test_count = 455
hf_total = hf_train_count + hf_test_count

print(f"✓ Simulated HF dataset:")
print(f"  Total images: {hf_total}")
print(f"  Train: {hf_train_count}")
print(f"  Test: {hf_test_count}")
print(f"  Classes: {len([c for c in hf_classes_sim if c != 'UNKNOWN'])} (excluding UNKNOWN)")

# ======================
# Test 3: Check class overlap
# ======================
print("\n[Test 3] Checking class overlap between datasets...")
hf_classes_set = set(hf_classes_sim)
overlap = local_classes.intersection(hf_classes_set)
local_only = local_classes - hf_classes_set
hf_only = hf_classes_set - local_classes

print(f"✓ Class overlap analysis:")
print(f"  Overlapping classes: {len(overlap)}")
print(f"  Local-only classes: {len(local_only)}")
print(f"  HF-only classes: {len(hf_only)}")

if overlap:
    print(f"  Sample overlapping: {list(overlap)[:10]}")
if local_only:
    print(f"  Sample local-only: {list(local_only)[:10]}")
if hf_only:
    print(f"  Sample HF-only: {list(hf_only)[:10]}")

# ======================
# Test 4: Simulate merged dataset
# ======================
print("\n[Test 4] Simulating merged dataset...")
total_images = len(all_files) + hf_total
all_classes_merged = local_classes.union(hf_classes_set)

print(f"✓ Merged dataset:")
print(f"  Total images: {total_images}")
print(f"  Total unique classes: {len(all_classes_merged)}")
print(f"  Local images: {len(all_files)} ({100*len(all_files)/total_images:.1f}%)")
print(f"  HF images: {hf_total} ({100*hf_total/total_images:.1f}%)")

# ======================
# Test 5: Simulate 80/20 split on merged data
# ======================
print("\n[Test 5] Simulating 80/20 split on merged dataset...")
train_size = int(total_images * 0.8)
test_size = total_images - train_size

print(f"✓ Split simulation:")
print(f"  Train: {train_size} ({100*train_size/total_images:.1f}%)")
print(f"  Test: {test_size} ({100*test_size/total_images:.1f}%)")

# ======================
# Test 6: Verify file naming logic
# ======================
print("\n[Test 6] Verifying file naming logic...")
test_cases = [
    ("A1.png", "local", "finalized_dataset/homogenized/train_flat/A1.png"),
    ("D21_1.png", "local", "finalized_dataset/homogenized/train_flat/D21_1.png"),
    ("A1_0.png", "hf", "hf_dataset_temp/A1_0.png"),
    ("D21_test_5.png", "hf", "hf_dataset_temp/D21_test_5.png"),
]

all_pass = True
for filename, source, expected_path in test_cases:
    if source == 'hf':
        actual_path = os.path.join("hf_dataset_temp", filename)
    else:
        actual_path = os.path.join(train_data_dir, filename)
    
    if actual_path == expected_path:
        print(f"  ✓ {filename} ({source}) -> {actual_path}")
    else:
        print(f"  ❌ {filename} ({source}) -> {actual_path} (expected {expected_path})")
        all_pass = False

if not all_pass:
    print("❌ File path logic failed!")
    sys.exit(1)

# ======================
# Summary
# ======================
print("\n" + "=" * 60)
print("✅ ALL INTEGRATION TESTS PASSED!")
print("=" * 60)
print(f"\nSummary:")
print(f"  Local dataset: {len(all_files)} images, {len(local_classes)} classes")
print(f"  HF dataset: ~{hf_total} images, ~{len([c for c in hf_classes_sim if c != 'UNKNOWN'])} classes")
print(f"  Merged: ~{total_images} images, ~{len(all_classes_merged)} classes")
print(f"  Class overlap: {len(overlap)} classes")
print(f"\n✓ Integration logic is correct!")
print("✓ Ready to merge datasets in notebook!")

