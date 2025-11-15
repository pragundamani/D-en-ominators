"""
Example usage of the hieroglyph classifier.
This script demonstrates how to use the classifier components.
"""

from pathlib import Path
from hieroglyph_dataset import HieroglyphDataset, get_transforms
from model import create_model

# Example 1: Load and inspect the dataset
print("=" * 60)
print("Example 1: Loading Dataset")
print("=" * 60)

json_path = "datasets/archaeohack-starterpack/data/gardiner_hieroglyphs_with_unicode_hex.json"
images_dir = "datasets/archaeohack-starterpack/data/utf-pngs"

dataset = HieroglyphDataset(
    json_path=json_path,
    images_dir=images_dir,
    transform=get_transforms(is_training=False, augment=False)
)

print(f"Total samples: {len(dataset)}")
print(f"Number of classes: {dataset.num_classes}")
print(f"\nFirst 10 Gardiner numbers:")
for i in range(min(10, dataset.num_classes)):
    gardiner_num = dataset.get_class_name(i)
    info = dataset.get_class_info(i)
    print(f"  {gardiner_num}: {info.get('hieroglyph', '')} - {info.get('description', 'N/A')}")

# Example 2: Get a sample
print("\n" + "=" * 60)
print("Example 2: Getting a Sample")
print("=" * 60)

if len(dataset) > 0:
    image, label, metadata = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Label: {label}")
    print(f"Metadata: {metadata}")
    print(f"Class name: {dataset.get_class_name(label)}")

# Example 3: Create a model
print("\n" + "=" * 60)
print("Example 3: Creating Model")
print("=" * 60)

device = "cuda" if __import__('torch').cuda.is_available() else "cpu"
model = create_model(
    model_name="vit_base_patch16_224",
    num_classes=dataset.num_classes,
    pretrained=True,
    dropout_rate=0.3,
    device=device
)

print(f"Model created on {device}")
print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

print("\n" + "=" * 60)
print("To train the model, run:")
print("=" * 60)
print(f"""
python train.py \\
    --json_path {json_path} \\
    --images_dir {images_dir} \\
    --model_name vit_base_patch16_224 \\
    --batch_size 32 \\
    --num_epochs 50 \\
    --learning_rate 1e-4 \\
    --save_dir checkpoints
""")

print("\n" + "=" * 60)
print("To run inference, use:")
print("=" * 60)
print(f"""
python inference.py \\
    --checkpoint checkpoints/best_model.pth \\
    --json_path {json_path} \\
    --images_dir {images_dir} \\
    --image path/to/image.png \\
    --top_k 5
""")

