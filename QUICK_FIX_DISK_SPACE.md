# Quick Fix for Low Disk Space (7.71 GB free)

## Immediate Solutions

### Option 1: Force Scratch Directory (Recommended)

If scratch/tmp directories have more space, add this at the **beginning** of Cell 4 (before the directory detection code):

```python
# FORCE SCRATCH DIRECTORY - Uncomment and modify if needed
# WORK_DIR = Path("/scratch") / os.getenv("USER", "user") / "D-en-ominators"
# Or try:
# WORK_DIR = Path("/tmp") / os.getenv("USER", "user") / "D-en-ominators"
```

### Option 2: Download Only Essential Dataset

In Cell 10, comment out all datasets except the homogenized one:

```python
datasets = [
    # Comment out these if low on space:
    # {
    #     "name": "HamdiJr/Egyptian_hieroglyphs",
    #     ...
    # },
    # {
    #     "name": "GlyphDataset",
    #     ...
    # },
    # ... (comment out others)
    
    # Keep only this essential one:
    {
        "name": "Homogenized Finalized Dataset",
        "type": "github",
        "url": "https://github.com/pragundamani/D-en-ominators/tree/dev/finalized_dataset/homogenized",
        "output": "homogenized_dataset",
        "subpath": "finalized_dataset/homogenized"
    },
]
```

### Option 3: Clean Up Before Starting

Run this in a new cell before Cell 4:

```python
# Quick cleanup
import shutil
from pathlib import Path

# Remove Python cache
for pycache in Path(".").rglob("__pycache__"):
    shutil.rmtree(pycache)
    print(f"Removed: {pycache}")

# Check space
import psutil
disk = psutil.disk_usage(".")
print(f"Free space: {disk.free / 1e9:.2f} GB")
```

### Option 4: Check Scratch Space

Run this to see if scratch has more space:

```python
import psutil
from pathlib import Path

scratch_paths = [
    Path("/scratch") / os.getenv("USER", "user"),
    Path("/tmp") / os.getenv("USER", "user"),
    Path.home() / "scratch",
]

print("Checking scratch/tmp directories:")
for path in scratch_paths:
    try:
        disk = psutil.disk_usage(str(path))
        print(f"{path}: {disk.free / 1e9:.2f} GB free")
    except:
        print(f"{path}: Not available")
```

## What the Notebook Does Automatically

The notebook now:
1. ✅ Checks space in all possible directories
2. ✅ Prioritizes scratch/tmp (usually more space)
3. ✅ Shows which directory has the most space
4. ✅ Selects the directory with most free space automatically

## If Still Low on Space

1. **Use cleanup cell** (Cell 12) after downloads
2. **Download datasets one at a time** - modify the loop to download only what you need
3. **Contact admin** - request quota increase or scratch space allocation

## Estimated Space Needs

- **Minimal setup** (only homogenized dataset): ~3-5 GB
- **Full setup** (all datasets): ~15-25 GB
- **With checkpoints**: +2-5 GB per model

With 7.71 GB free, you can:
- ✅ Download only the homogenized dataset
- ✅ Train one model
- ⚠️ May need cleanup after training

