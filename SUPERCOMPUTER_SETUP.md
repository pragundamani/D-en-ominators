# Supercomputer Setup - Notebook Adaptations

## Changes Made

The notebook has been adapted for a supercomputer environment with:
- **64GB RAM**
- **NVIDIA RTX 8000 GPU** (48GB VRAM)
- **Read-only filesystem** (except user uploads)

## Key Adaptations

### 1. **Package Installation**
- Checks if packages are already installed before installing
- Uses `--user` flag for installations (works in read-only systems)
- Removed Colab-specific commands (`!apt-get`)

### 2. **Directory Setup**
- Automatically detects writable directories:
  - Current working directory
  - `$HOME/work`, `$HOME/scratch`, `$HOME/tmp`
  - `/tmp/$USER`, `/scratch/$USER`
- Tests write permissions before using directories
- Falls back to current directory if needed

### 3. **System Information**
- Displays GPU information (name, memory, CUDA version, compute capability)
- Shows system RAM and available memory
- Detects number of GPUs
- RTX 8000 specific optimizations (48GB VRAM, larger batch sizes)

### 4. **Git Clone**
- Uses `subprocess` instead of shell commands
- Better error handling
- Works in environments without shell access

### 5. **File Uploads**
- Removed Colab-specific `google.colab.files`
- Scans current directory for uploaded files
- Works with any file upload mechanism

### 6. **Training/Inference/Evaluation**
- Uses `subprocess.run()` instead of shell commands
- Proper path handling for checkpoints
- GPU environment variable setup
- Better error reporting

### 7. **Results Saving**
- Saves to user home directory (`$HOME/hieroglyph_classifier_results`)
- Shows file sizes
- Works without Google Drive

## Usage

1. **Upload the notebook** to your supercomputer
2. **Run cells sequentially**
3. **Upload code files** to the current directory (if not using git clone)
4. **Datasets will download** to user-writable location
5. **Results save** to `$HOME/hieroglyph_classifier_results`

## Environment Compatibility

- ✅ Jupyter Notebook/Lab
- ✅ SLURM job scripts
- ✅ Interactive supercomputer sessions
- ✅ Read-only filesystems
- ✅ Multi-GPU systems (uses GPU 0 by default)

## Notes

- All paths are relative to user-writable directories
- No system-level installations required
- Works with module systems (if packages are pre-loaded)
- GPU detection is automatic
- RTX 8000 optimizations: Uses batch size 64 (can be adjusted based on model size)
- RTX 8000 has 48GB VRAM - can handle very large models and datasets

