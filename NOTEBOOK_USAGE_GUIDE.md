# How to Use the Supercomputer Notebook

## Quick Start Guide

### Step 1: Upload the Notebook
1. Upload `colab_notebook.ipynb` to your supercomputer
2. Open it in Jupyter Notebook/Lab or your preferred notebook environment

### Step 2: Run Cells Sequentially

#### Cell 1: Install Dependencies
- **What it does**: Checks for and installs required Python packages
- **Time**: 2-5 minutes (if packages need installation)
- **Output**: Shows system information including GPU detection

#### Cell 2: Set Up Code Structure
- **What it does**: Creates working directories in user-writable locations
- **Time**: < 1 second
- **Output**: Shows where files will be stored

#### Cell 3: Clone Repository (Option A) OR Upload Files (Option B)

**Option A - Git Clone (Recommended):**
- **What it does**: Clones your GitHub repository
- **Time**: 1-2 minutes
- **Requirements**: Git must be available
- **Output**: Confirms repository cloned and files copied

**Option B - Manual Upload:**
- **What it does**: Processes uploaded files from current directory
- **Time**: < 1 second
- **Requirements**: Upload files to current working directory first
- **Output**: Lists found files

#### Cell 4: Download Datasets
- **What it does**: Downloads all required datasets with progress bars
- **Time**: 10-60 minutes (depends on dataset sizes and internet speed)
- **Features**: 
  - Automatically skips datasets that already exist
  - Shows progress for each download
  - Downloads from multiple sources (Hugging Face, GitHub, Kaggle, direct)
- **Output**: Summary of downloaded datasets

#### Cell 5: Verify Setup
- **What it does**: Checks that all required files are in place
- **Time**: < 1 second
- **Output**: Status of code files, datasets, Python path, and GPU

#### Cell 6: Run Training
- **What it does**: Trains the hieroglyph classifier model
- **Time**: 30 minutes - several hours (depends on dataset size and epochs)
- **RTX 8000 Optimizations**: 
  - Automatically uses batch size 64
  - Optimized for 48GB VRAM
- **Output**: Training progress and final model checkpoint

#### Cell 7: Run Inference (Optional)
- **What it does**: Classifies hieroglyph images using trained model
- **Time**: < 1 minute per image
- **Requirements**: 
  - Trained model checkpoint
  - Image file uploaded to current directory
- **Output**: Top-k predictions with confidence scores

#### Cell 8: Evaluate Model (Optional)
- **What it does**: Evaluates model performance on test set
- **Time**: 5-15 minutes
- **Output**: Evaluation metrics and results saved to JSON

#### Cell 9: Save Results (Optional)
- **What it does**: Copies checkpoints and results to persistent location
- **Time**: < 1 minute
- **Output**: Confirms files saved to `$HOME/hieroglyph_classifier_results`

## Detailed Instructions

### Prerequisites

1. **Access to supercomputer** with:
   - Jupyter Notebook/Lab or similar
   - Python 3.7+
   - Internet connection (for downloads)
   - User-writable directory

2. **Optional but recommended**:
   - Git installed (for repository cloning)
   - Kaggle API credentials (for Kaggle datasets)

### Running the Notebook

#### Method 1: Interactive Session

1. **Start Jupyter**:
   ```bash
   # If using SLURM
   srun --gres=gpu:rtx8000:1 --mem=64G --time=4:00:00 --pty bash
   jupyter notebook --ip=0.0.0.0 --no-browser --port=8888
   
   # Or use Jupyter Lab
   jupyter lab --ip=0.0.0.0 --no-browser --port=8888
   ```

2. **Open notebook** in browser (use port forwarding if needed)

3. **Run cells sequentially** using Shift+Enter

#### Method 2: Batch Job (SLURM)

Create a SLURM script:

```bash
#!/bin/bash
#SBATCH --job-name=hieroglyph_train
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

# Load modules if needed
# module load python/3.9
# module load cuda/11.8

# Convert notebook to script and run
jupyter nbconvert --to script colab_notebook.ipynb
python colab_notebook.py
```

Run with:
```bash
sbatch train_job.sh
```

### File Upload Methods

#### For Code Files (Option B):

1. **Using Jupyter interface**:
   - Click "Upload" button in Jupyter
   - Select your files or zip archive
   - Files appear in current directory

2. **Using command line**:
   ```bash
   # Upload to current directory
   scp -r new/* user@supercomputer:/path/to/notebook/directory/
   ```

3. **Using SFTP**:
   - Connect via SFTP client
   - Upload files to notebook directory

#### For Images (Inference):

1. Upload image files (.png, .jpg, .jpeg) to current directory
2. Run inference cell - it will automatically find uploaded images

### Customization

#### Adjust Training Parameters

Edit Cell 6 (Training) to modify:
- `--batch_size`: Adjust based on GPU memory (RTX 8000 uses 64)
- `--num_epochs`: Number of training epochs
- `--learning_rate`: Learning rate (default: 1e-4)
- `--model_name`: Change model architecture (e.g., "resnet50", "efficientnet_b0")

Example:
```python
train_cmd = [
    sys.executable,
    str(NEW_DIR / "train.py"),
    "--json_path", json_path,
    "--images_dir", images_dir,
    "--model_name", "vit_base_patch16_224",
    "--batch_size", "128",  # Increase for RTX 8000
    "--num_epochs", "100",  # More epochs
    "--learning_rate", "5e-5",  # Lower learning rate
    "--save_dir", str(CHECKPOINTS_DIR),
    "--val_split", "0.2"
]
```

#### Change Working Directory

If you want to use a specific directory, modify Cell 2:

```python
# Force specific directory
WORK_DIR = Path("/your/custom/path/D-en-ominators")
```

### Troubleshooting

#### Issue: "Permission denied" errors
**Solution**: The notebook automatically finds writable directories. If issues persist, check:
- Your home directory permissions
- Available space in user directories
- Contact system administrator

#### Issue: GPU not detected
**Solution**: 
- Check CUDA installation: `nvidia-smi`
- Verify PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`
- May need to load CUDA module: `module load cuda/11.8`

#### Issue: Datasets not downloading
**Solution**:
- Check internet connection
- For Kaggle datasets: Upload `kaggle.json` to `/content/.kaggle/` (Colab) or `~/.kaggle/` (supercomputer)
- Some datasets may require authentication

#### Issue: Out of memory errors
**Solution**:
- Reduce batch size in training cell
- RTX 8000 has 48GB VRAM, but very large models may still need smaller batches
- Check available RAM: The notebook shows this in Cell 1

#### Issue: Files not found
**Solution**:
- Verify file paths in output
- Check that uploads went to correct directory
- Use Cell 5 (Verify Setup) to check file locations

### Monitoring Progress

#### During Training:
- Watch the output for epoch progress
- Check GPU utilization: `watch -n 1 nvidia-smi`
- Monitor memory: The notebook shows memory usage

#### Check Results:
- Checkpoints saved to: `{WORK_DIR}/checkpoints/`
- Training history: `{WORK_DIR}/checkpoints/training_history.json`
- Evaluation results: `{WORK_DIR}/evaluation_results.json`

### Best Practices

1. **Run cells in order** - Each cell depends on previous ones
2. **Check outputs** - Verify each step completed successfully
3. **Save frequently** - Notebook saves automatically, but save important results
4. **Monitor resources** - Watch GPU and memory usage
5. **Use batch jobs** - For long training runs, use SLURM instead of interactive
6. **Backup results** - Use Cell 9 to save results to persistent location

### Expected Output Locations

After running the notebook, you'll find:

```
{WORK_DIR}/
├── new/                    # Code files
├── datasets/               # Downloaded datasets
│   ├── huggingface_egyptian_hieroglyphs/
│   ├── glyphnet/
│   ├── homogenized_dataset/
│   └── ...
├── checkpoints/            # Model checkpoints
│   ├── best_model.pth
│   ├── training_history.json
│   └── ...
└── evaluation_results.json # Evaluation results

$HOME/hieroglyph_classifier_results/  # Saved results
├── checkpoints/
└── evaluation_results.json
```

### Quick Reference

| Cell | Purpose | Time | Required |
|------|---------|------|----------|
| 1 | Install packages | 2-5 min | Always |
| 2 | Setup directories | <1 sec | Always |
| 3 | Get code files | 1-2 min | Always |
| 4 | Download datasets | 10-60 min | Always |
| 5 | Verify setup | <1 sec | Always |
| 6 | Train model | 30 min - hours | Always |
| 7 | Run inference | <1 min | Optional |
| 8 | Evaluate model | 5-15 min | Optional |
| 9 | Save results | <1 min | Optional |

### Getting Help

If you encounter issues:
1. Check the error messages in notebook output
2. Verify all prerequisites are met
3. Check file paths and permissions
4. Review the troubleshooting section above
5. Check system logs if using SLURM

## Example Workflow

```bash
# 1. Upload notebook to supercomputer
scp colab_notebook.ipynb user@supercomputer:~/work/

# 2. Start interactive session
srun --gres=gpu:rtx8000:1 --mem=64G --time=4:00:00 --pty bash

# 3. Start Jupyter
cd ~/work
jupyter notebook --ip=0.0.0.0 --no-browser --port=8888

# 4. In browser, open notebook and run cells 1-6

# 5. For inference, upload image and run cell 7

# 6. Save results with cell 9
```

That's it! The notebook handles everything else automatically.

