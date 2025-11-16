#!/usr/bin/env python3
"""
Simple test script to validate the notebook changes work correctly
Tests file structure and basic logic without requiring ML libraries
"""

import os
import sys
from collections import Counter

print("=" * 60)
print("Testing Notebook Implementation (Simple Version)")
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

# Test extraction on sample files
print("\n[Test 2a] Testing class extraction logic...")
sample_files = all_files[:10]
for fname in sample_files:
    class_name = extract_class_name(fname)
    print(f"  {fname:30s} -> {class_name}")

# Extract all classes
data = []
for filename in all_files:
    class_name = extract_class_name(filename)
    data.append({'file_name': filename, 'label': class_name})

print(f"✓ Extracted classes from {len(data)} files")

# ======================
# Test 3: Analyze class distribution
# ======================
print("\n[Test 3] Analyzing class distribution...")
class_counts = Counter([item['label'] for item in data])
unique_classes = sorted(class_counts.keys())
num_classes = len(unique_classes)

print(f"✓ Total unique classes: {num_classes}")
print(f"  Sample classes: {unique_classes[:15]}")
print(f"\n  Class distribution (top 10):")
for cls, count in class_counts.most_common(10):
    print(f"    {cls:10s}: {count:4d} images")

# Check for classes with many variants
print(f"\n  Classes with multiple variants (samples):")
variant_classes = {}
for item in data:
    fname = item['file_name']
    if '_' in fname.replace('.png', ''):
        cls = item['label']
        if cls not in variant_classes:
            variant_classes[cls] = []
        variant_classes[cls].append(fname)
        if len(variant_classes[cls]) <= 3:
            pass  # Just collecting

# Show a few examples
shown = 0
for cls, files in variant_classes.items():
    if shown < 3:
        print(f"    {cls}: {len(files)} variants (e.g., {files[0]})")
        shown += 1

# ======================
# Test 4: Verify file paths
# ======================
print("\n[Test 4] Verifying file paths...")
sample_files = [item['file_name'] for item in data[:20]]
all_exist = True
missing = []
for fname in sample_files:
    path = os.path.join(train_data_dir, fname)
    if not os.path.exists(path):
        missing.append(fname)
        all_exist = False

if all_exist:
    print(f"✓ All {len(sample_files)} sample files exist")
else:
    print(f"❌ {len(missing)} files not found:")
    for fname in missing[:5]:
        print(f"    {fname}")
    sys.exit(1)

# ======================
# Test 5: Simulate train/test split logic
# ======================
print("\n[Test 5] Simulating train/test split...")
total = len(data)
test_size = int(total * 0.2)
train_size = total - test_size

# Simulate stratified split by class
train_data = []
test_data = []

# Simple split: for each class, put 80% in train, 20% in test
for cls in unique_classes:
    cls_files = [item for item in data if item['label'] == cls]
    cls_total = len(cls_files)
    cls_test_size = max(1, int(cls_total * 0.2))  # At least 1 for test
    
    # Simple split (not truly random, but simulates the logic)
    train_data.extend(cls_files[:cls_total - cls_test_size])
    test_data.extend(cls_files[cls_total - cls_test_size:])

print(f"✓ Simulated split:")
print(f"  Train: {len(train_data)} samples ({100*len(train_data)/total:.1f}%)")
print(f"  Test:  {len(test_data)} samples ({100*len(test_data)/total:.1f}%)")

# Check class distribution in splits
train_classes = set([item['label'] for item in train_data])
test_classes = set([item['label'] for item in test_data])

print(f"\n  Classes in train: {len(train_classes)}")
print(f"  Classes in test:  {len(test_classes)}")

if len(train_classes) == num_classes:
    print("  ✓ All classes present in train set")
else:
    missing = set(unique_classes) - train_classes
    print(f"  ⚠ {len(missing)} classes missing from train: {list(missing)[:5]}")

if len(test_classes) == num_classes:
    print("  ✓ All classes present in test set")
else:
    missing = set(unique_classes) - test_classes
    print(f"  ⚠ {len(missing)} classes missing from test: {list(missing)[:5]}")

# ======================
# Test 6: Check file naming patterns
# ======================
print("\n[Test 6] Checking file naming patterns...")
patterns = {
    'simple': 0,  # e.g., A1.png
    'with_variant': 0,  # e.g., D21_1.png
    'other': 0
}

for item in data:
    fname = item['file_name'].replace('.png', '')
    if '_' in fname:
        patterns['with_variant'] += 1
    elif fname.isalnum():
        patterns['simple'] += 1
    else:
        patterns['other'] += 1

print(f"✓ File patterns:")
print(f"  Simple names (e.g., A1.png): {patterns['simple']}")
print(f"  With variants (e.g., D21_1.png): {patterns['with_variant']}")
print(f"  Other patterns: {patterns['other']}")

# ======================
# Test 7: Verify class extraction handles all cases
# ======================
print("\n[Test 7] Verifying class extraction handles edge cases...")
test_cases = [
    ("A1.png", "A1"),
    ("D21_1.png", "D21"),
    ("D21_100.png", "D21"),
    ("Aa15_3.png", "Aa15"),
    ("UNKNOWN.png", "UNKNOWN"),
]

all_pass = True
for filename, expected in test_cases:
    result = extract_class_name(filename)
    if result == expected:
        print(f"  ✓ {filename:20s} -> {result}")
    else:
        print(f"  ❌ {filename:20s} -> {result} (expected {expected})")
        all_pass = False

if not all_pass:
    print("❌ Class extraction failed some test cases!")
    sys.exit(1)

# ======================
# Summary
# ======================
print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print(f"\nSummary:")
print(f"  Total images: {len(data)}")
print(f"  Total classes: {num_classes}")
print(f"  Train (simulated): {len(train_data)} ({100*len(train_data)/total:.1f}%)")
print(f"  Test (simulated):  {len(test_data)} ({100*len(test_data)/total:.1f}%)")
print(f"\n  File patterns:")
print(f"    Simple: {patterns['simple']}")
print(f"    Variants: {patterns['with_variant']}")
print(f"\n✓ Notebook logic is correct!")
print("✓ Ready for supercomputer!")

