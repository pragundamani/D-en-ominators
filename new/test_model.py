"""
Test script for hieroglyph classifier using test dataset
"""

import argparse
from pathlib import Path
import json
from collections import defaultdict

import torch
from PIL import Image
import numpy as np
from tqdm import tqdm

from hieroglyph_dataset import HieroglyphDataset, get_transforms
from model import create_model
from inference import load_model, predict_image


def test_with_hand_drawn_samples(
    checkpoint_path: str,
    json_path: str,
    images_dir: str,
    test_dir: str,
    device: str = "cuda"
):
    """
    Test model with hand-drawn samples from me-sign-examples-pjb directory.
    
    Args:
        checkpoint_path: Path to model checkpoint
        json_path: Path to gardiner_hieroglyphs JSON
        images_dir: Path to utf-pngs directory (for class mapping)
        test_dir: Path to me-sign-examples-pjb directory
        device: Device to run inference on
    """
    print("=" * 60)
    print("Testing Model with Hand-Drawn Samples")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    model, checkpoint = load_model(checkpoint_path, device)
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Load dataset for class mapping
    dataset = HieroglyphDataset(
        json_path=json_path,
        images_dir=images_dir,
        transform=None
    )
    
    # Find all test images organized by Gardiner number
    test_dir = Path(test_dir)
    test_results = defaultdict(list)
    
    print(f"\nScanning test directory: {test_dir}")
    for gardiner_dir in test_dir.iterdir():
        if gardiner_dir.is_dir():
            gardiner_num = gardiner_dir.name
            image_files = list(gardiner_dir.glob("*.png"))
            
            if image_files:
                print(f"  Found {len(image_files)} samples for {gardiner_num}")
                test_results[gardiner_num] = image_files
    
    print(f"\nTotal test classes: {len(test_results)}")
    print(f"Total test images: {sum(len(imgs) for imgs in test_results.values())}")
    
    # Test each class
    print("\n" + "=" * 60)
    print("Running Inference")
    print("=" * 60)
    
    correct_predictions = 0
    total_predictions = 0
    class_results = {}
    
    for gardiner_num, image_files in tqdm(test_results.items(), desc="Testing classes"):
        class_correct = 0
        class_total = len(image_files)
        
        # Get the expected class index
        expected_idx = dataset.gardiner_to_idx.get(gardiner_num)
        if expected_idx is None:
            print(f"  ⚠ Warning: {gardiner_num} not in training classes, skipping")
            continue
        
        for image_file in image_files:
            try:
                predictions = predict_image(
                    model,
                    str(image_file),
                    dataset,
                    device,
                    top_k=5
                )
                
                # Check if top prediction matches
                top_pred = predictions[0]
                predicted_gardiner = top_pred['gardiner_num']
                
                if predicted_gardiner == gardiner_num:
                    class_correct += 1
                    correct_predictions += 1
                
                total_predictions += 1
                
            except Exception as e:
                print(f"  ⚠ Error processing {image_file}: {e}")
                continue
        
        accuracy = class_correct / class_total if class_total > 0 else 0
        class_results[gardiner_num] = {
            'correct': class_correct,
            'total': class_total,
            'accuracy': accuracy
        }
    
    # Print results
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    print(f"Overall Accuracy: {correct_predictions}/{total_predictions} = {correct_predictions/total_predictions*100:.2f}%")
    print(f"\nPer-Class Results:")
    print("-" * 60)
    
    # Sort by accuracy
    sorted_results = sorted(class_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    for gardiner_num, results in sorted_results:
        print(f"{gardiner_num:6s}: {results['correct']:3d}/{results['total']:3d} = {results['accuracy']*100:6.2f}%")
    
    # Save results
    results_file = Path(checkpoint_path).parent / "test_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'overall_accuracy': correct_predictions / total_predictions if total_predictions > 0 else 0,
            'total_correct': correct_predictions,
            'total_predictions': total_predictions,
            'class_results': class_results
        }, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")
    
    return class_results


def main():
    parser = argparse.ArgumentParser(description='Test hieroglyph classifier')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--json_path', type=str, required=True,
                        help='Path to gardiner_hieroglyphs_with_unicode_hex.json')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Path to utf-pngs directory')
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Path to test directory (e.g., me-sign-examples-pjb)')
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    test_with_hand_drawn_samples(
        args.checkpoint,
        args.json_path,
        args.images_dir,
        args.test_dir,
        device
    )


if __name__ == "__main__":
    main()

