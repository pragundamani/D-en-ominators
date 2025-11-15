"""
Training Script for Hieroglyph Classifier
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
from PIL import Image
from tqdm import tqdm

from hieroglyph_dataset import HieroglyphDataset, get_transforms
from model import create_model


class SubsetWithTransform(torch.utils.data.Dataset):
    """Custom subset dataset that applies transforms correctly."""
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        sample = self.dataset.samples[original_idx]
        label = self.dataset.labels[original_idx]
        
        # Load image
        try:
            image = Image.open(sample['image_path']).convert('RGB')
            image = np.array(image)
        except Exception as e:
            print(f"Error loading {sample['image_path']}: {e}")
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        metadata = {
            'gardiner_num': sample.get('gardiner_num', ''),
            'hieroglyph': sample.get('hieroglyph', ''),
            'description': sample.get('description', '') or ''
        }
        
        return image, label, metadata


class Trainer:
    """Training manager for hieroglyph classifier."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.class_weights = class_weights
        
        # Loss function
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        self.best_val_acc = 0.0
        self.best_model_state = None
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels, _ in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return {'loss': epoch_loss, 'acc': epoch_acc}
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for images, labels, _ in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return {'loss': epoch_loss, 'acc': epoch_acc, 'f1': epoch_f1}
    
    def train(self, num_epochs: int, save_dir: Optional[Path] = None):
        """Train the model for multiple epochs."""
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_metrics = self.train_epoch()
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['acc'])
            
            # Validate
            val_metrics = self.validate()
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['acc'])
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Print metrics
            print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['acc']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['acc']:.4f}, Val F1: {val_metrics['f1']:.4f}")
            
            # Save best model (always save first model, then only if better)
            if val_metrics['acc'] >= self.best_val_acc or self.best_val_acc == 0.0:
                self.best_val_acc = val_metrics['acc']
                self.best_model_state = self.model.state_dict().copy()
                print(f"✓ New best validation accuracy: {self.best_val_acc:.4f}")
                
                if save_dir:
                    save_path = save_dir / "best_model.pth"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.best_model_state,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_acc': self.best_val_acc,
                        'history': self.history,
                        'model_name': 'efficientnet_b0',  # Add model name for loading
                        'dropout_rate': 0.3
                    }, save_path)
                    print(f"  Saved to {save_path}")
        
        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            print(f"\n✓ Loaded best model with validation accuracy: {self.best_val_acc:.4f}")
        
        return self.history


def main():
    parser = argparse.ArgumentParser(description='Train hieroglyph classifier')
    parser.add_argument('--json_path', type=str, required=True,
                        help='Path to gardiner_hieroglyphs_with_unicode_hex.json')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Path to utf-pngs directory')
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
    parser.add_argument('--use_class_weights', action='store_true',
                        help='Use class weights for imbalanced data')
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
    
    # Load full dataset
    print("Loading dataset...")
    full_dataset = HieroglyphDataset(
        json_path=args.json_path,
        images_dir=args.images_dir,
        transform=None,  # Will set transforms on subsets
        filter_priority=args.filter_priority
    )
    
    # Split dataset
    indices = np.arange(len(full_dataset))
    
    # Check if we can use stratification (need at least 2 samples per class)
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
        print(f"Warning: Some classes have only 1 sample. Using random split without stratification.")
        train_indices, val_indices = train_test_split(
            indices,
            test_size=args.val_split,
            random_state=args.seed
        )
    
    # Create subset datasets with different transforms
    train_dataset = SubsetWithTransform(
        full_dataset,
        train_indices,
        get_transforms(args.image_size, is_training=True, augment=True)
    )
    
    val_dataset = SubsetWithTransform(
        full_dataset,
        val_indices,
        get_transforms(args.image_size, is_training=False, augment=False)
    )
    
    # Compute class weights if requested
    class_weights = None
    if args.use_class_weights:
        labels_array = np.array(full_dataset.labels)
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels_array),
            y=labels_array
        )
        class_weights = torch.FloatTensor(class_weights).to(device)
        print(f"Computed class weights (min: {class_weights.min():.3f}, max: {class_weights.max():.3f})")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues on some systems
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues on some systems
        pin_memory=False
    )
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        model_name=args.model_name,
        num_classes=full_dataset.num_classes,
        pretrained=True,
        dropout_rate=args.dropout_rate,
        device=device
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        class_weights=class_weights
    )
    
    # Train
    history = trainer.train(
        num_epochs=args.num_epochs,
        save_dir=save_dir
    )
    
    # Save training history
    history_path = save_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()

