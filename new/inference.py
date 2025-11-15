"""
Inference Script for Hieroglyph Classifier
"""

import argparse
from pathlib import Path
import json

import torch
from PIL import Image
import numpy as np

from hieroglyph_dataset import HieroglyphDataset, get_transforms
from model import create_model


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to infer num_classes from checkpoint
    if 'model_state_dict' in checkpoint:
        # Get num_classes from the classifier layer
        state_dict = checkpoint['model_state_dict']
        classifier_weight = state_dict.get('classifier.1.weight', None)
        if classifier_weight is not None:
            num_classes = classifier_weight.shape[0]
        else:
            raise ValueError("Could not infer num_classes from checkpoint")
    else:
        raise ValueError("Invalid checkpoint format")
    
    # Create model (model_name might be in checkpoint metadata)
    model_name = checkpoint.get('model_name', 'vit_base_patch16_224')
    dropout_rate = checkpoint.get('dropout_rate', 0.3)
    
    model = create_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
        dropout_rate=dropout_rate,
        device=device
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint


def predict_image(
    model: torch.nn.Module,
    image_path: str,
    dataset: HieroglyphDataset,
    device: str = "cuda",
    top_k: int = 5
):
    """
    Predict hieroglyph class for a single image.
    
    Args:
        model: Trained model
        image_path: Path to image file
        dataset: Dataset object (for transforms and class mapping)
        device: Device to run inference on
        top_k: Number of top predictions to return
    
    Returns:
        List of (gardiner_num, confidence, info_dict) tuples
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    
    # Apply transforms
    transform = get_transforms(is_training=False, augment=False)
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
    
    # Get predictions
    predictions = []
    for i in range(top_k):
        idx = top_indices[0][i].item()
        prob = top_probs[0][i].item()
        gardiner_num = dataset.get_class_name(idx)
        class_info = dataset.get_class_info(idx)
        
        predictions.append({
            'rank': i + 1,
            'gardiner_num': gardiner_num,
            'confidence': prob,
            'hieroglyph': class_info.get('hieroglyph', ''),
            'description': class_info.get('description', ''),
            'details': class_info.get('details', '')
        })
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description='Inference for hieroglyph classifier')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--json_path', type=str, required=True,
                        help='Path to gardiner_hieroglyphs_with_unicode_hex.json')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Path to utf-pngs directory')
    parser.add_argument('--image', type=str,
                        help='Path to single image to classify')
    parser.add_argument('--image_dir', type=str,
                        help='Directory of images to classify')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top predictions to show')
    parser.add_argument('--output', type=str,
                        help='Output JSON file for batch predictions')
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model, checkpoint = load_model(args.checkpoint, device)
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Best validation accuracy: {checkpoint.get('val_acc', 'unknown'):.4f}")
    
    # Load dataset (for transforms and class mapping)
    dataset = HieroglyphDataset(
        json_path=args.json_path,
        images_dir=args.images_dir,
        transform=None  # Will use transforms in predict_image
    )
    
    # Single image prediction
    if args.image:
        print(f"\nClassifying: {args.image}")
        predictions = predict_image(
            model, args.image, dataset, device, args.top_k
        )
        
        print("\nTop predictions:")
        for pred in predictions:
            print(f"  {pred['rank']}. {pred['gardiner_num']} ({pred['hieroglyph']})")
            print(f"     Confidence: {pred['confidence']:.4f}")
            if pred['description']:
                print(f"     Description: {pred['description']}")
            print()
    
    # Batch prediction
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        image_files = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
        
        results = []
        for image_path in image_files:
            predictions = predict_image(
                model, str(image_path), dataset, device, args.top_k
            )
            results.append({
                'image': str(image_path),
                'predictions': predictions
            })
        
        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved results to {args.output}")
        else:
            # Print results
            for result in results:
                print(f"\n{result['image']}:")
                for pred in result['predictions']:
                    print(f"  {pred['rank']}. {pred['gardiner_num']} - {pred['confidence']:.4f}")
    else:
        print("Please provide either --image or --image_dir")


if __name__ == "__main__":
    main()

