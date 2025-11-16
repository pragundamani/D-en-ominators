"""
LoRA Training Script for Hieroglyph Classifier
Uses PEFT (Parameter-Efficient Fine-Tuning) with LoRA
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from PIL import Image
from tqdm import tqdm
import timm
from peft import LoraConfig, get_peft_model, TaskType

from hieroglyph_dataset import HieroglyphDataset, get_transforms
from model import create_model


class LoRATrainer:
    """Training manager with LoRA for hieroglyph classifier."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1
    ):
        # Apply LoRA to the model
        print("\n" + "="*60)
        print("APPLYING LoRA TO MODEL")
        print("="*60)
        
        # LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,  # For vision models
            r=16,  # Rank (low-rank dimension)
            lora_alpha=32,  # LoRA alpha scaling
            target_modules=["qkv", "proj", "fc1", "fc2"],  # Target attention and MLP layers
            lora_dropout=0.1,
            bias="none",
        )
        
        # Convert model to PEFT with LoRA
        self.model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters (LoRA): {trainable_params:,}")
        print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
        print("="*60)
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.class_weights = class_weights
        
        # Loss function
        if use_focal_loss:
            self.criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
            print(f"Using Focal Loss (gamma={focal_gamma})")
        else:
            if class_weights is not None:
                self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
            else:
                self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # Optimizer - only optimize LoRA parameters
        self.optimizer = optim.AdamW(
            self.model.parameters(),  # Only LoRA parameters are trainable
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
                
                pbar.set_postfix({'loss': loss.item()})
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return {'loss': epoch_loss, 'acc': epoch_acc, 'f1': f1}
    
    def train(self, num_epochs: int, save_dir: Path) -> Dict:
        """Train the model."""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"TRAINING WITH LoRA")
        print(f"{'='*60}")
        print(f"Epochs: {num_epochs}")
        print(f"Save directory: {save_dir}")
        print(f"{'='*60}\n")
        
        for epoch in range(1, num_epochs + 1):
            print(f"Epoch {epoch}/{num_epochs}")
            print("-" * 60)
            
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
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_metrics['acc'] > self.best_val_acc:
                self.best_val_acc = val_metrics['acc']
                self.best_model_state = self.model.state_dict()
                
                # Save LoRA adapters
                best_path = save_dir / "best_model_lora.pth"
                self.model.save_pretrained(str(save_dir / "lora_adapters"))
                torch.save({
                    'model_state_dict': self.best_model_state,
                    'epoch': epoch,
                    'val_acc': self.best_val_acc,
                    'history': self.history
                }, best_path)
                print(f"✓ Saved best model (val_acc: {self.best_val_acc:.4f})")
            
            print()
        
        return self.history


class FocalLoss(nn.Module):
    """Focal Loss for handling imbalanced classes."""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def main():
    parser = argparse.ArgumentParser(description='Train hieroglyph classifier with LoRA')
    parser.add_argument('--json_path', type=str, required=True, help='Path to JSON metadata file')
    parser.add_argument('--images_dir', type=str, required=True, help='Path to images directory')
    parser.add_argument('--ram_cache', type=str, default=None, help='Path to RAM cache file')
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224', help='Timm model name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save checkpoints')
    parser.add_argument('--use_focal_loss', action='store_true', help='Use focal loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing')
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load dataset
    print("\nLoading dataset...")
    if args.ram_cache and Path(args.ram_cache).exists():
        print(f"Loading from RAM cache: {args.ram_cache}")
        cache_data = torch.load(args.ram_cache, map_location='cpu')
        # Use regular dataset but we'll need to modify it to use RAM cache
        # For now, use regular dataset loading
        full_dataset = HieroglyphDataset(
            json_path=cache_data.get('json_path', args.json_path),
            images_dir=cache_data.get('images_dir', args.images_dir),
            transform=get_transforms(train=True)
        )
        print("Note: RAM cache loaded but dataset will load from disk. To use RAM cache, modify HieroglyphDataset.__getitem__")
    else:
        full_dataset = HieroglyphDataset(
            json_path=args.json_path,
            images_dir=args.images_dir,
            transform=get_transforms(train=True)
        )
    
    # Split dataset
    train_size = int((1 - args.val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Compute class weights
    train_labels = [full_dataset[i][1] for i in train_dataset.indices]
    unique_labels = np.unique(train_labels)
    class_weights = compute_class_weight('balanced', classes=unique_labels, y=train_labels)
    label_to_weight = {label: weight for label, weight in zip(unique_labels, class_weights)}
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    # Create weighted sampler
    sample_weights = np.array([label_to_weight.get(label, 1.0) for label in train_labels])
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        model_name=args.model_name,
        num_classes=full_dataset.num_classes,
        pretrained=True,
        dropout_rate=0.3,
        device=device
    )
    
    # Create LoRA trainer
    trainer = LoRATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        class_weights=class_weights,
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing if not args.use_focal_loss else 0.0
    )
    
    # Train
    save_dir = Path(args.save_dir)
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

