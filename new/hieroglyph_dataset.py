"""
Hieroglyph Dataset Loader
Loads images from utf-pngs directory and maps them to Gardiner numbers from JSON.
"""

import json
import os
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class HieroglyphDataset(Dataset):
    """
    Dataset for Egyptian hieroglyph classification.
    
    Maps Gardiner numbers from JSON to corresponding PNG images.
    """
    
    def __init__(
        self,
        json_path: str,
        images_dir: str,
        transform: Optional[A.Compose] = None,
        filter_priority: bool = False
    ):
        """
        Initialize the dataset.
        
        Args:
            json_path: Path to gardiner_hieroglyphs_with_unicode_hex.json
            images_dir: Path to directory containing PNG images (utf-pngs)
            transform: Albumentations transform pipeline
            filter_priority: If True, only include hieroglyphs with is_priority=True
        """
        self.json_path = Path(json_path)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.filter_priority = filter_priority
        
        # Load JSON data
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.hieroglyph_data = json.load(f)
        
        # Build mapping of gardiner_num -> image path
        self.samples = []
        self.gardiner_to_idx = {}
        self.idx_to_gardiner = {}
        
        valid_gardiner_nums = set()
        idx = 0
        
        for entry in self.hieroglyph_data:
            gardiner_num = entry['gardiner_num']
            
            # Filter by priority if requested
            if self.filter_priority and not entry.get('is_priority', False):
                continue
            
            # Check if image exists
            image_path = self.images_dir / f"{gardiner_num}.png"
            if image_path.exists():
                self.samples.append({
                    'gardiner_num': gardiner_num,
                    'image_path': str(image_path),
                    'hieroglyph': entry.get('hieroglyph', ''),
                    'description': entry.get('description', ''),
                    'details': entry.get('details', ''),
                    'unicode_hex': entry.get('unicode_hex', '')
                })
                
                if gardiner_num not in self.gardiner_to_idx:
                    self.gardiner_to_idx[gardiner_num] = idx
                    self.idx_to_gardiner[idx] = gardiner_num
                    idx += 1
                valid_gardiner_nums.add(gardiner_num)
        
        # Create label mapping for all samples
        self.labels = [self.gardiner_to_idx[s['gardiner_num']] for s in self.samples]
        self.num_classes = len(self.gardiner_to_idx)
        
        print(f"Loaded {len(self.samples)} samples")
        print(f"Number of unique classes: {self.num_classes}")
        if self.filter_priority:
            print(f"Filtered to priority hieroglyphs only")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, dict]:
        """
        Get a sample from the dataset.
        
        Returns:
            tuple: (image_tensor, label, metadata_dict)
        """
        sample = self.samples[idx]
        image_path = sample['image_path']
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return black image as fallback
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            # Default: convert to tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        metadata = {
            'gardiner_num': sample['gardiner_num'],
            'hieroglyph': sample['hieroglyph'],
            'description': sample['description']
        }
        
        return image, label, metadata
    
    def get_class_name(self, idx: int) -> str:
        """Get Gardiner number for a class index."""
        return self.idx_to_gardiner.get(idx, 'Unknown')
    
    def get_class_info(self, idx: int) -> dict:
        """Get full information for a class index."""
        gardiner_num = self.idx_to_gardiner.get(idx, None)
        if gardiner_num:
            for entry in self.hieroglyph_data:
                if entry['gardiner_num'] == gardiner_num:
                    return entry
        return {}


def get_transforms(
    image_size: int = 224,
    is_training: bool = True,
    augment: bool = True
) -> A.Compose:
    """
    Get data augmentation transforms.
    
    Args:
        image_size: Target image size
        is_training: Whether this is for training
        augment: Whether to apply augmentation
    
    Returns:
        Albumentations transform pipeline
    """
    if is_training and augment:
        transform = A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    return transform

