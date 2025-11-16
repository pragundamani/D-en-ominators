# Solutions for "Disk Quota Exceeded" Error

## Quick Fixes

### 1. Use Scratch/Tmp Directories
The notebook automatically tries scratch/tmp directories first (they usually have more space). If it's still using your home directory:

```python
# In Cell 2, force scratch directory:
WORK_DIR = Path("/scratch") / os.getenv("USER", "user") / "D-en-ominators"
```

### 2. Clean Up Old Files
Run the cleanup cell (Cell 12) to remove:
- Old checkpoints (keeps only best + latest 3)
- Temporary files (__pycache__, .pyc files)
- Hugging Face cache (if not needed)

### 3. Download Only Essential Datasets
Comment out datasets you don't need in Cell 10:

```python
datasets = [
    # Comment out datasets you don't need:
    # {
    #     "name": "HamdiJr/Egyptian_hieroglyphs",
    #     ...
    # },
    {
        "name": "Homogenized Finalized Dataset",  # Keep this one
        ...
    },
]
```

### 4. Check Current Disk Usage
The notebook shows disk space in:
- Cell 1: System-wide disk space
- Cell 10: After dataset downloads

## Detailed Solutions

### Option A: Move to Scratch Directory

1. **Check scratch space**:
   ```bash
   df -h /scratch
   ```

2. **Modify Cell 2** to use scratch:
   ```python
   # Force scratch directory
   scratch_dir = Path("/scratch") / os.getenv("USER", "user")
   scratch_dir.mkdir(parents=True, exist_ok=True)
   WORK_DIR = scratch_dir / "D-en-ominators"
   ```

### Option B: Clean Up Existing Files

1. **Remove old checkpoints** (Cell 12):
   ```python
   cleanup_old_checkpoints(keep_best=True, keep_latest=2)  # Keep only 2 latest
   ```

2. **Remove temporary files**:
   ```python
   cleanup_temporary_files()
   ```

3. **Remove Hugging Face cache** (if datasets already downloaded):
   ```python
   cleanup_dataset_cache()
   ```

### Option C: Selective Dataset Download

Modify the datasets list to download only what you need:

```python
# Minimal dataset list (only homogenized dataset)
datasets = [
    {
        "name": "Homogenized Finalized Dataset",
        "type": "github",
        "url": "https://github.com/pragundamani/D-en-ominators/tree/dev/finalized_dataset/homogenized",
        "output": "homogenized_dataset",
        "subpath": "finalized_dataset/homogenized"
    },
]
```

### Option D: Use Symbolic Links

If you have datasets elsewhere, create symlinks:

```python
# In Cell 4, after directory setup:
import os
if Path("/path/to/existing/datasets").exists():
    if DATASETS_DIR.exists():
        DATASETS_DIR.rmdir()
    os.symlink("/path/to/existing/datasets", str(DATASETS_DIR))
    print("✓ Using symlink to existing datasets")
```

## Prevention

### Before Starting:
1. **Check available space** (Cell 1 shows this)
2. **Estimate needed space**:
   - Datasets: ~5-20 GB (depends on which ones)
   - Checkpoints: ~1-5 GB per model
   - Total: ~10-30 GB recommended

### During Execution:
1. **Monitor disk space** - notebook shows it after downloads
2. **Clean up as you go** - remove old checkpoints periodically
3. **Use scratch** - it's designed for large temporary files

## Space Requirements by Component

| Component | Typical Size | Can Skip? |
|-----------|--------------|-----------|
| Hugging Face dataset | 1-5 GB | Yes |
| GitHub repos | 0.5-2 GB each | Some |
| Kaggle datasets | 1-10 GB | Yes |
| Homogenized dataset | 2-5 GB | No (essential) |
| Model checkpoints | 1-5 GB | Keep best only |
| Training history | <1 MB | No |

## Emergency Cleanup Script

If you need to free space immediately:

```python
# Run this in a new cell
import shutil
from pathlib import Path

# Remove old checkpoints (keep only best)
checkpoints_dir = Path("checkpoints")  # Adjust path
if checkpoints_dir.exists():
    for cp in checkpoints_dir.glob("*.pth"):
        if "best" not in cp.name:
            cp.unlink()
            print(f"Removed: {cp}")

# Remove __pycache__
for pycache in Path(".").rglob("__pycache__"):
    shutil.rmtree(pycache)
    print(f"Removed: {pycache}")

# Check space
import psutil
disk = psutil.disk_usage(".")
print(f"Free space: {disk.free / 1e9:.2f} GB")
```

## Contact System Administrator

If none of these work:
1. Request quota increase
2. Ask about scratch space allocation
3. Request temporary space for large datasets

## Best Practices

1. **Always use scratch/tmp** for large files
2. **Clean up regularly** - don't keep all checkpoints
3. **Download selectively** - only what you need
4. **Monitor space** - check before large operations
5. **Use compression** - some datasets can be compressed

