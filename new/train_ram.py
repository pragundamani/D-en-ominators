"""
Training Script for Hieroglyph Classifier with RAM Cache Support
Modified version that uses preloaded images from RAM for faster training.
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm

# Import RAM-based dataset if available, otherwise fall back to regular
try:
    from hieroglyph_dataset_ram import HieroglyphDatasetRAM, get_transforms
    RAM_AVAILABLE = True
except ImportError:
    from hieroglyph_dataset import HieroglyphDataset, get_transforms
    RAM_AVAILABLE = False

from model import create_model
from train import SubsetWithTransform, FocalLoss, Trainer


def main():
    parser = argparse.ArgumentParser(description='Train hieroglyph classifier with RAM cache support')
    parser.add_argument('--json_path', type=str, required=False,
                        help='Path to gardiner_hieroglyphs_with_unicode_hex.json (not needed if using RAM cache)')
    parser.add_argument('--images_dir', type=str, required=False,
                        help='Path to utf-pngs directory (not needed if using RAM cache)')
    parser.add_argument('--ram_cache', type=str, default=None,
                        help='Path to RAM cache file (.pt file with preloaded images)')
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224',
                        help='Timm model name')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--filter_priority', action='store_true',
                        help='Only use priority hieroglyphs')
    parser.add_argument('--use_class_weights', action='store_true', default=True,
                        help='Use class weights for imbalanced data (default: True)')
    parser.add_argument('--use_focal_loss', action='store_true', default=True,
                        help='Use Focal Loss instead of CrossEntropy (default: True)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Gamma parameter for Focal Loss (default: 2.0)')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing factor (default: 0.1)')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Load dataset (from RAM cache or disk)
    print("Loading dataset...")
    if args.ram_cache and Path(args.ram_cache).exists() and RAM_AVAILABLE:
        print(f"✓ Loading from RAM cache: {args.ram_cache}")
        ram_cache_data = torch.load(args.ram_cache, map_location='cpu')
        full_dataset = HieroglyphDatasetRAM(
            ram_cache_data=ram_cache_data,
            transform=None,  # Will set transforms on subsets
            filter_priority=args.filter_priority
        )
        print("✓ Using RAM-based dataset (faster training, no disk I/O)")
    else:
        if not args.json_path or not args.images_dir:
            raise ValueError("Either --ram_cache or both --json_path and --images_dir must be provided")
        print("Using disk-based dataset")
        full_dataset = HieroglyphDataset(
            json_path=args.json_path,
            images_dir=args.images_dir,
            transform=None,
            filter_priority=args.filter_priority
        )
    
    # Split dataset (same as original train.py)
    indices = np.arange(len(full_dataset))
    unique_labels, counts = np.unique(full_dataset.labels, return_counts=True)
    can_stratify = np.all(counts >= 2)
    
    if can_stratify:
        train_indices, val_indices = train_test_split(
            indices,
            test_size=args.val_split,
            random_state=args.seed,
            stratify=full_dataset.labels
        )
    else:
        train_indices, val_indices = train_test_split(
            indices,
            test_size=args.val_split,
            random_state=args.seed
        )
    
    # Create transforms
    train_transform = get_transforms(
        image_size=args.image_size,
        is_training=True,
        augment=True
    )
    val_transform = get_transforms(
        image_size=args.image_size,
        is_training=False,
        augment=False
    )
    
    # Create subsets with transforms
    train_dataset = SubsetWithTransform(full_dataset, train_indices, train_transform)
    val_dataset = SubsetWithTransform(full_dataset, val_indices, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device == "cuda" else False
    )
    
    # Create model
    model = create_model(
        model_name=args.model_name,
        num_classes=full_dataset.num_classes,
        pretrained=True,
        dropout_rate=args.dropout_rate,
        device=device
    )
    
    # Class weights
    class_weights = None
    if args.use_class_weights:
        unique_labels = np.unique(full_dataset.labels)
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_labels,
            y=full_dataset.labels
        )
        class_weights = torch.FloatTensor(class_weights).to(device)
        print(f"Using class weights for imbalanced data")
    
    # Create trainer and train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        class_weights=class_weights,
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing
    )
    
    history = trainer.train(num_epochs=args.num_epochs, save_dir=save_dir)
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    print(f"Best validation accuracy: {trainer.best_val_acc:.4f}")
    print(f"Checkpoints saved to: {save_dir}")


if __name__ == "__main__":
    main()

