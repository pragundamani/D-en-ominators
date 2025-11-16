"""
Homogenize images in the archaeohack-starterpack dataset.
Standardizes all images to the same size, format, and properties.
"""

import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import json


def homogenize_image(
    image_path: Path,
    output_path: Path,
    target_size: tuple = (224, 224),
    target_mode: str = 'RGB',
    maintain_aspect_ratio: bool = True,
    padding_color: tuple = (255, 255, 255)
):
    """
    Homogenize a single image.
    
    Args:
        image_path: Path to input image
        output_path: Path to save homogenized image
        target_size: Target (width, height)
        target_mode: Target color mode ('RGB', 'L', etc.)
        maintain_aspect_ratio: If True, pad to maintain aspect ratio
        padding_color: Color for padding (RGB tuple)
    
    Returns:
        bool: True if successful
    """
    try:
        # Open image
        img = Image.open(image_path)
        
        # Convert to target mode
        if img.mode != target_mode:
            if target_mode == 'RGB':
                if img.mode == 'L':
                    img = img.convert('RGB')
                elif img.mode == 'RGBA':
                    # Create white background
                    background = Image.new('RGB', img.size, padding_color)
                    background.paste(img, mask=img.split()[3] if len(img.split()) == 4 else None)
                    img = background
                else:
                    img = img.convert(target_mode)
            else:
                img = img.convert(target_mode)
        
        # Resize
        if maintain_aspect_ratio:
            # Calculate aspect ratio
            img_ratio = img.width / img.height
            target_ratio = target_size[0] / target_size[1]
            
            if img_ratio > target_ratio:
                # Image is wider - fit to width
                new_width = target_size[0]
                new_height = int(target_size[0] / img_ratio)
            else:
                # Image is taller - fit to height
                new_height = target_size[1]
                new_width = int(target_size[1] * img_ratio)
            
            # Resize with high-quality resampling
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create new image with target size and padding
            new_img = Image.new(target_mode, target_size, padding_color)
            
            # Paste resized image centered
            paste_x = (target_size[0] - new_width) // 2
            paste_y = (target_size[1] - new_height) // 2
            new_img.paste(img, (paste_x, paste_y))
            img = new_img
        else:
            # Direct resize (may distort aspect ratio)
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, 'PNG', optimize=False)
        return True
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False


def homogenize_directory(
    input_dir: Path,
    output_dir: Path,
    target_size: tuple = (224, 224),
    target_mode: str = 'RGB',
    maintain_aspect_ratio: bool = True,
    padding_color: tuple = (255, 255, 255),
    pattern: str = "*.png"
):
    """
    Homogenize all images in a directory.
    
    Args:
        input_dir: Input directory
        output_dir: Output directory
        target_size: Target size (width, height)
        target_mode: Target color mode
        maintain_aspect_ratio: Maintain aspect ratio with padding
        padding_color: Padding color
        pattern: File pattern to match
    
    Returns:
        dict: Statistics about processed images
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Find all images
    image_files = list(input_dir.rglob(pattern))
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return {'total': 0, 'success': 0, 'failed': 0}
    
    print(f"Found {len(image_files)} images in {input_dir}")
    
    # Process images
    success_count = 0
    failed_count = 0
    
    for img_path in tqdm(image_files, desc="Homogenizing images"):
        # Calculate relative path
        relative_path = img_path.relative_to(input_dir)
        output_path = output_dir / relative_path
        
        if homogenize_image(
            img_path,
            output_path,
            target_size,
            target_mode,
            maintain_aspect_ratio,
            padding_color
        ):
            success_count += 1
        else:
            failed_count += 1
    
    return {
        'total': len(image_files),
        'success': success_count,
        'failed': failed_count
    }


def main():
    parser = argparse.ArgumentParser(description='Homogenize images in archaeohack dataset')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory (e.g., datasets/archaeohack-starterpack/data)')
    parser.add_argument('--output_dir', type=str,
                        help='Output directory (default: input_dir/homogenized)')
    parser.add_argument('--target_size', type=int, nargs=2, default=[224, 224],
                        help='Target size as width height (default: 224 224)')
    parser.add_argument('--target_mode', type=str, default='RGB',
                        choices=['RGB', 'L', 'RGBA'],
                        help='Target color mode (default: RGB)')
    parser.add_argument('--maintain_aspect', action='store_true', default=True,
                        help='Maintain aspect ratio with padding (default: True)')
    parser.add_argument('--no_aspect', action='store_false', dest='maintain_aspect',
                        help='Stretch to target size (may distort)')
    parser.add_argument('--padding_color', type=int, nargs=3, default=[255, 255, 255],
                        help='Padding color RGB (default: 255 255 255)')
    parser.add_argument('--subdirs', type=str, nargs='+',
                        default=['utf-pngs', 'me-sign-examples-pjb'],
                        help='Subdirectories to process (default: utf-pngs me-sign-examples-pjb)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_dir / "homogenized"
    
    target_size = tuple(args.target_size)
    padding_color = tuple(args.padding_color)
    
    print("=" * 60)
    print("Image Homogenization")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target size: {target_size}")
    print(f"Target mode: {args.target_mode}")
    print(f"Maintain aspect ratio: {args.maintain_aspect}")
    print(f"Padding color: {padding_color}")
    print(f"Subdirectories: {args.subdirs}")
    print("=" * 60)
    
    # Process each subdirectory
    all_stats = {}
    
    for subdir in args.subdirs:
        subdir_path = input_dir / subdir
        if not subdir_path.exists():
            print(f"\n⚠ Warning: {subdir_path} does not exist, skipping")
            continue
        
        print(f"\nProcessing {subdir}...")
        output_subdir = output_dir / subdir
        
        stats = homogenize_directory(
            subdir_path,
            output_subdir,
            target_size,
            args.target_mode,
            args.maintain_aspect,
            padding_color
        )
        
        all_stats[subdir] = stats
        print(f"  ✓ Processed: {stats['success']}/{stats['total']} images")
        if stats['failed'] > 0:
            print(f"  ✗ Failed: {stats['failed']} images")
    
    # Save statistics
    stats_file = output_dir / "homogenization_stats.json"
    with open(stats_file, 'w') as f:
        json.dump({
            'target_size': target_size,
            'target_mode': args.target_mode,
            'maintain_aspect_ratio': args.maintain_aspect,
            'padding_color': padding_color,
            'statistics': all_stats
        }, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Homogenization Complete")
    print("=" * 60)
    total_images = sum(s['total'] for s in all_stats.values())
    total_success = sum(s['success'] for s in all_stats.values())
    total_failed = sum(s['failed'] for s in all_stats.values())
    
    print(f"Total images: {total_images}")
    print(f"Successfully processed: {total_success}")
    print(f"Failed: {total_failed}")
    print(f"\nStatistics saved to: {stats_file}")
    print(f"Homogenized images saved to: {output_dir}")


if __name__ == "__main__":
    main()

