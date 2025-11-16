# Hieroglyph Image Classifier

A complete image classification system for Egyptian hieroglyphs using the Gardiner classification system.

## Overview

This classifier uses the Gardiner hieroglyph data from `gardiner_hieroglyphs_with_unicode_hex.json` and corresponding PNG images from the `utf-pngs` directory to train a deep learning model that can classify hieroglyph images.

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Structure

- **JSON file**: Contains hieroglyph metadata with Gardiner numbers, descriptions, and Unicode information
- **Images**: PNG files named by Gardiner number (e.g., `A1.png`, `M17.png`) in the `utf-pngs` directory

## Usage

### Training

Train a model from scratch:

```bash
python train.py \
    --json_path datasets/archaeohack-starterpack/data/gardiner_hieroglyphs_with_unicode_hex.json \
    --images_dir datasets/archaeohack-starterpack/data/utf-pngs \
    --model_name vit_base_patch16_224 \
    --batch_size 32 \
    --num_epochs 50 \
    --learning_rate 1e-4 \
    --save_dir checkpoints
```

**Key Arguments:**
- `--json_path`: Path to the Gardiner hieroglyphs JSON file
- `--images_dir`: Path to the directory containing PNG images
- `--model_name`: Timm model name (e.g., `vit_base_patch16_224`, `resnet50`, `efficientnet_b0`)
- `--batch_size`: Batch size for training
- `--num_epochs`: Number of training epochs
- `--learning_rate`: Learning rate
- `--filter_priority`: Only use priority hieroglyphs (if specified)
- `--use_class_weights`: Use class weights for imbalanced data
- `--val_split`: Validation split ratio (default: 0.2)

### Inference

Classify a single image:

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --json_path datasets/archaeohack-starterpack/data/gardiner_hieroglyphs_with_unicode_hex.json \
    --images_dir datasets/archaeohack-starterpack/data/utf-pngs \
    --image path/to/image.png \
    --top_k 5
```

Classify multiple images:

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --json_path datasets/archaeohack-starterpack/data/gardiner_hieroglyphs_with_unicode_hex.json \
    --images_dir datasets/archaeohack-starterpack/data/utf-pngs \
    --image_dir path/to/images/ \
    --output results.json
```

### Evaluation

Evaluate model performance:

```bash
python evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --json_path datasets/archaeohack-starterpack/data/gardiner_hieroglyphs_with_unicode_hex.json \
    --images_dir datasets/archaeohack-starterpack/data/utf-pngs \
    --output evaluation_results.json
```

## Model Architecture

The classifier uses a Vision Transformer (ViT) or CNN backbone from the `timm` library with a custom classification head. Default model is `vit_base_patch16_224`, but you can use any timm model.

## Data Augmentation

Training uses data augmentation including:
- Random horizontal flips
- Random rotations
- Shift, scale, and rotate transformations
- Random brightness and contrast adjustments

## Output

- **Checkpoints**: Saved in the `checkpoints/` directory
- **Training history**: Saved as `training_history.json`
- **Best model**: Saved as `best_model.pth` based on validation accuracy

## Example

```python
from hieroglyph_dataset import HieroglyphDataset, get_transforms
from model import create_model
from inference import load_model, predict_image

# Load dataset
dataset = HieroglyphDataset(
    json_path="datasets/archaeohack-starterpack/data/gardiner_hieroglyphs_with_unicode_hex.json",
    images_dir="datasets/archaeohack-starterpack/data/utf-pngs"
)

# Load trained model
model, checkpoint = load_model("checkpoints/best_model.pth")

# Predict
predictions = predict_image(
    model,
    "path/to/image.png",
    dataset,
    top_k=5
)

for pred in predictions:
    print(f"{pred['gardiner_num']}: {pred['confidence']:.4f}")
```

## Notes

- The model automatically maps Gardiner numbers from the JSON to image files
- Images are resized to 224x224 by default
- The classifier supports any number of classes based on available images
- Class weights can be used to handle imbalanced datasets

