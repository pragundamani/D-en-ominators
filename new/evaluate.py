"""
Evaluation Script for Hieroglyph Classifier
"""

import argparse
from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm

from hieroglyph_dataset import HieroglyphDataset, get_transforms
from model import create_model
from inference import load_model


def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: str = "cuda",
    dataset: HieroglyphDataset = None
):
    """
    Evaluate model on a dataset.
    
    Returns:
        Dictionary with metrics and detailed results
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_metadata = []
    
    with torch.no_grad():
        for images, labels, metadata in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_metadata.extend(metadata)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    # Per-class accuracy
    cm = confusion_matrix(all_labels, all_preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    # Get class names
    if dataset:
        class_names = [dataset.get_class_name(i) for i in range(len(per_class_acc))]
    else:
        class_names = [f"Class_{i}" for i in range(len(per_class_acc))]
    
    # Classification report
    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        output_dict=True
    )
    
    results = {
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'per_class_accuracy': {
            name: float(acc) for name, acc in zip(class_names, per_class_acc)
        },
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate hieroglyph classifier')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--json_path', type=str, required=True,
                        help='Path to gardiner_hieroglyphs_with_unicode_hex.json')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Path to utf-pngs directory')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size')
    parser.add_argument('--output', type=str,
                        help='Output JSON file for results')
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model, checkpoint = load_model(args.checkpoint, device)
    
    # Load dataset
    print("Loading dataset...")
    dataset = HieroglyphDataset(
        json_path=args.json_path,
        images_dir=args.images_dir,
        transform=get_transforms(args.image_size, is_training=False, augment=False)
    )
    
    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Evaluate
    print("\nEvaluating model...")
    results = evaluate_model(model, data_loader, device, dataset)
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score (Macro): {results['f1_macro']:.4f}")
    print(f"F1 Score (Weighted): {results['f1_weighted']:.4f}")
    print("\nTop 10 classes by accuracy:")
    sorted_classes = sorted(
        results['per_class_accuracy'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    for name, acc in sorted_classes[:10]:
        print(f"  {name}: {acc:.4f}")
    
    print("\nBottom 10 classes by accuracy:")
    for name, acc in sorted_classes[-10:]:
        print(f"  {name}: {acc:.4f}")
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

