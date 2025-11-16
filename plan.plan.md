# Adapt Notebook for Folder-Based Multi-Class Dataset with 80/20 Split

## Overview
Modify `image_class_og.ipynb` to work with `finalized_dataset` structure. Use sklearn's `train_test_split` for an 80/20 split on all data from `homogenized/train_flat/`, replacing the existing CSV-based approach.

## Steps

### 1. Load all data and extract classes
- Scan `homogenized/train_flat/` directory for all PNG files
- Extract class name from filename:
  - `A1.png` → class `A1`
  - `D21_1.png` → class `D21` (strip `_1`, `_2`, etc. suffixes)
- Create DataFrame with columns: `file_name`, `label` (class name as string)

### 2. Create class mapping
- Get unique classes from DataFrame
- Map class names to numeric indices (0, 1, 2, ..., N)
- Store mapping dictionary for later reference
- Calculate total number of classes for model config

### 3. Apply 80/20 train/test split
- Use `train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)`
- Stratified split ensures balanced class distribution in both sets
- Result: `train_df` (80%), `test_df` (20%)
- Optionally split `train_df` further into train/val if needed for validation

### 4. Update Cell 4 (Data Loading & Split)
- Remove CSV reading code (`pd.read_csv(train_csv_path)`)
- Replace with code from steps 1-3 above
- Remove existing train/val/test split logic (the 30/50/20 split)
- Keep only the 80/20 split from sklearn
- Update `train_data_dir` to point to `homogenized/train_flat/`
- Update dataset creation to use split DataFrames

### 5. Update Cell 5 (Model Configuration)
- Change `num_classes` from `2` to total number of unique classes (from step 2)
- Model head will output N classes instead of 2

### 6. Update Cell 9 (Test Dataset)
- Remove test CSV reading code
- Test data now comes from the 80/20 split (step 3), not a separate folder
- Update `TestDataset` to work with the split test DataFrame
- Or remove this cell entirely if test is handled in Cell 4

### 7. Remove/Update Cell 0-1 (Kaggle Download)
- Remove Kaggle dataset download code (not needed for local dataset)
- Keep only necessary imports

## Files to Modify
- `image_class_og.ipynb`:
  - Cell 0-1: Remove Kaggle download, keep imports
  - Cell 4: Replace CSV reading with filename scanning + 80/20 split
  - Cell 5: Update `num_classes`
  - Cell 9: Remove or update test CSV reading

## Key Changes Summary
- **Data source**: `homogenized/train_flat/` instead of CSV + flat directory
- **Split method**: sklearn 80/20 instead of predefined train/test folders
- **Classification**: Multi-class (N classes) instead of binary (2 classes)
- **Class extraction**: From filenames instead of CSV labels

---

## LoRA Integration Plan - Dual Model Training with Progress Tracking

### Overview
Train two models simultaneously: one with full fine-tuning and one with LoRA. Compare their performance, training speed, and memory usage. Add progress visualization to track training metrics.

### Steps for Dual Model Training

### 1. Install Required Libraries
- Add `peft` and `matplotlib` to Cell 0
- Update: `!pip install -q timm datasets peft matplotlib`

### 2. Add LoRA Configuration (New Cell after Cell 5)
- Import LoRA components: `from peft import LoraConfig, get_peft_model, TaskType`
- Define LoRA hyperparameters:
  - `r=16` (rank - low-rank dimension)
  - `lora_alpha=32` (scaling factor)
  - `lora_dropout=0.1` (dropout for LoRA layers)
  - `target_modules=["qkv", "proj", "fc1", "fc2"]` (ViT attention and MLP layers)
- Create `LoraConfig` object

### 3. Create Two Models (Cell 5 or New Cell)
- **Model 1 (Full Fine-tuning)**: Standard ViT with all parameters trainable
- **Model 2 (LoRA)**: Same base ViT with LoRA adapters applied
- Print parameter counts for both models
- Show memory efficiency comparison

### 4. Create Separate Optimizers (Cell 6)
- `optimizer_full`: For full fine-tuning model
- `optimizer_lora`: For LoRA model (only trains LoRA parameters)
- Both use AdamW with same learning rate (or slightly higher for LoRA)

### 5. Update Training Loop (Cell 8)
- Train both models in the same loop
- Track metrics separately:
  - `train_loss_full`, `train_acc_full`, `val_loss_full`, `val_acc_full`
  - `train_loss_lora`, `train_acc_lora`, `val_loss_lora`, `val_acc_lora`
- Save best models for both (separate checkpoint files)
- Track training time for comparison

### 6. Add Progress Visualization (New Cell after Cell 8)
- Plot training curves:
  - Loss curves (train/val for both models)
  - Accuracy curves (train/val for both models)
  - Side-by-side comparison
- Use matplotlib to create plots
- Update plots after each epoch or every N epochs

### 7. Model Comparison (New Cell)
- Final comparison table:
  - Best validation accuracy
  - Training time
  - Model size (checkpoint file size)
  - Number of trainable parameters
  - Final test accuracy

### 8. Model Saving (Cell 10)
- Save both models:
  - Full model: `VIT-full-{val_acc:.4f}.pth`
  - LoRA model: `VIT-lora-{val_acc:.4f}.pth` (or use `save_pretrained()` for adapters)

### Files to Modify
- `image_class_og.ipynb`:
  - Cell 0: Add `peft matplotlib` to pip install
  - Cell 5: Create both models (full + LoRA)
  - Cell 6: Create two optimizers
  - Cell 8: Update training loop to train both models
  - New Cell: Add progress visualization
  - New Cell: Add final comparison
  - Cell 10: Save both models

### Benefits of Dual Training
- **Direct Comparison**: See LoRA vs full fine-tuning side-by-side
- **Progress Tracking**: Visualize training progress in real-time
- **Performance Metrics**: Compare accuracy, speed, and memory usage
- **Flexibility**: Choose best model based on results

### Visualization Features
- Real-time loss/accuracy plots
- Epoch-by-epoch updates
- Side-by-side model comparison
- Training time tracking

