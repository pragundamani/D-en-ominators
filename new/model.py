"""
Model Architecture for Hieroglyph Classification
"""

import torch
import torch.nn as nn
import timm


class HieroglyphClassifier(nn.Module):
    """
    Hieroglyph classifier using Vision Transformer or CNN backbone.
    """
    
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        num_classes: int = 768,
        pretrained: bool = True,
        dropout_rate: float = 0.3
    ):
        """
        Initialize the classifier.
        
        Args:
            model_name: Timm model name (e.g., 'vit_base_patch16_224', 'resnet50')
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate for classifier head
        """
        super().__init__()
        
        # Create base model from timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # Remove default head
        )
        
        # Get feature dimension
        if hasattr(self.backbone, 'num_features'):
            feature_dim = self.backbone.num_features
        elif hasattr(self.backbone, 'head'):
            feature_dim = self.backbone.head.in_features
        else:
            # Try to infer from model
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                features = self.backbone(dummy_input)
                feature_dim = features.shape[-1]
        
        # Create classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
        
        Returns:
            Logits tensor of shape (B, num_classes)
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification head."""
        return self.backbone(x)


def create_model(
    model_name: str = "vit_base_patch16_224",
    num_classes: int = 768,
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    device: str = "cuda"
) -> HieroglyphClassifier:
    """
    Create and initialize a model.
    
    Args:
        model_name: Timm model name
        num_classes: Number of classes
        pretrained: Use pretrained weights
        dropout_rate: Dropout rate
        device: Device to move model to
    
    Returns:
        Initialized model
    """
    model = HieroglyphClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate
    )
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: {model_name}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Number of classes: {num_classes}")
    
    return model

