#!/usr/bin/env python3
"""
Script to download all datasets from datasets.md
Handles multiple sources: Hugging Face, GitHub, Kaggle, Roboflow, and direct downloads
"""

import os
import sys
import subprocess
import requests
from pathlib import Path
from urllib.parse import urlparse
import json

# Create datasets directory
DATASETS_DIR = Path("datasets")
DATASETS_DIR.mkdir(exist_ok=True)

def run_command(cmd, check=True):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Error: {e.stderr}")
        return None

def download_huggingface_dataset(dataset_name, output_dir):
    """Download a Hugging Face dataset"""
    print(f"\n📦 Downloading Hugging Face dataset: {dataset_name}")
    try:
        from datasets import load_dataset
        output_path = DATASETS_DIR / output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading dataset {dataset_name}...")
        dataset = load_dataset(dataset_name)
        
        # Save dataset
        if isinstance(dataset, dict):
            for split_name, split_data in dataset.items():
                split_path = output_path / split_name
                split_path.mkdir(exist_ok=True)
                split_data.save_to_disk(str(split_path))
                print(f"  ✓ Saved {split_name} split")
        else:
            dataset.save_to_disk(str(output_path))
            print(f"  ✓ Saved dataset")
        
        return True
    except ImportError:
        print("  ⚠ Installing datasets library...")
        run_command(f"{sys.executable} -m pip install datasets", check=False)
        return download_huggingface_dataset(dataset_name, output_dir)
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def download_github_repo(repo_url, output_dir, subpath=None):
    """Download a GitHub repository or specific subdirectory"""
    print(f"\n📦 Downloading GitHub repository: {repo_url}")
    try:
        output_path = DATASETS_DIR / output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Extract repo path from URL
        if "github.com" in repo_url:
            repo_path = repo_url.split("github.com/")[1].split("/tree/")[0]
            if "/tree/" in repo_url:
                branch = repo_url.split("/tree/")[1].split("/")[0]
                subpath = "/".join(repo_url.split("/tree/")[1].split("/")[1:])
            else:
                branch = "main"
            
            repo_url_clean = f"https://github.com/{repo_path}.git"
            
            # Clone repository
            if output_path.exists() and any(output_path.iterdir()):
                print(f"  ⚠ Directory {output_path} already exists, skipping clone")
            else:
                print(f"  Cloning {repo_url_clean} (branch: {branch})...")
                run_command(f"git clone --depth 1 --branch {branch} {repo_url_clean} {output_path}", check=False)
                
                # If subpath specified, move to that subdirectory
                if subpath:
                    subpath_full = output_path / subpath
                    if subpath_full.exists():
                        print(f"  ✓ Found subpath: {subpath}")
                    else:
                        print(f"  ⚠ Subpath {subpath} not found")
            
            return True
        else:
            print(f"  ✗ Invalid GitHub URL: {repo_url}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def download_kaggle_dataset(dataset_name, output_dir):
    """Download a Kaggle dataset"""
    print(f"\n📦 Downloading Kaggle dataset: {dataset_name}")
    try:
        output_path = DATASETS_DIR / output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Check if kaggle is installed
        try:
            import kaggle
        except ImportError:
            print("  ⚠ Installing kaggle library...")
            run_command(f"{sys.executable} -m pip install kaggle", check=False)
            import kaggle
        
        print(f"  Downloading {dataset_name}...")
        kaggle.api.dataset_download_files(dataset_name, path=str(output_path), unzip=True)
        print(f"  ✓ Downloaded to {output_path}")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        print(f"  ⚠ Note: Kaggle requires API credentials. Set up ~/.kaggle/kaggle.json")
        return False

def download_roboflow_dataset(dataset_url, output_dir):
    """Download a Roboflow dataset"""
    print(f"\n📦 Downloading Roboflow dataset: {dataset_url}")
    try:
        output_path = DATASETS_DIR / output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Check if roboflow is installed
        try:
            from roboflow import Roboflow
        except ImportError:
            print("  ⚠ Installing roboflow library...")
            run_command(f"{sys.executable} -m pip install roboflow", check=False)
            from roboflow import Roboflow
        
        # Extract workspace, project, and version from URL
        # URL format: https://universe.roboflow.com/workspace/project/dataset/version
        parts = dataset_url.rstrip('/').split('/')
        if 'universe.roboflow.com' in dataset_url:
            workspace = parts[-4]
            project = parts[-3]
            version = parts[-1]
            
            print(f"  ⚠ Roboflow requires API key. Please set ROBOHOW_API_KEY environment variable")
            print(f"  Workspace: {workspace}, Project: {project}, Version: {version}")
            print(f"  You can download manually from: {dataset_url}")
            return False
        else:
            print(f"  ✗ Unsupported Roboflow URL format")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def download_file(url, output_path):
    """Download a file directly from URL"""
    print(f"\n📦 Downloading file: {url}")
    try:
        output_file = DATASETS_DIR / output_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r  Progress: {percent:.1f}%", end='', flush=True)
        
        print(f"\n  ✓ Downloaded to {output_file}")
        return True
    except Exception as e:
        print(f"\n  ✗ Error: {e}")
        return False

def main():
    """Main function to download all datasets"""
    print("=" * 80)
    print("Downloading all datasets from datasets.md")
    print("=" * 80)
    
    datasets = [
        {
            "name": "HamdiJr/Egyptian_hieroglyphs",
            "type": "huggingface",
            "url": "https://huggingface.co/datasets/HamdiJr/Egyptian_hieroglyphs",
            "output": "huggingface_egyptian_hieroglyphs"
        },
        {
            "name": "GlyphDataset",
            "type": "github",
            "url": "https://github.com/AndreaBarucci/GlyphNet/tree/main/Data/GlyphDataset",
            "output": "glyphnet",
            "subpath": "Data/GlyphDataset"
        },
        {
            "name": "Hieroglyphs_Dataset",
            "type": "kaggle",
            "url": "https://www.kaggle.com/datasets/prasertk/hieroglyphs-dataset",
            "dataset_id": "prasertk/hieroglyphs-dataset",
            "output": "kaggle_hieroglyphs_dataset"
        },
        {
            "name": "Ancient-Language-Decipherer",
            "type": "github",
            "url": "https://github.com/JamesPigott/ancient-language-decipherer",
            "output": "ancient_language_decipherer",
            "subpath": "datasets"
        },
        {
            "name": "JSesh Unas Dataset",
            "type": "github",
            "url": "https://github.com/rosmira/jsesh",
            "output": "jsesh"
        },
        {
            "name": "Egyptian Hieroglyphs Classification",
            "type": "github",
            "url": "https://github.com/morrisfranken/HieroglyphClassification",
            "output": "hieroglyph_classification"
        },
        {
            "name": "Roboflow Egyptian Hieroglyphs",
            "type": "roboflow",
            "url": "https://universe.roboflow.com/roboflow-universe-projects/egyptian-hieroglyphs-roboflow/dataset/1",
            "output": "roboflow_egyptian_hieroglyphs"
        },
        {
            "name": "Egyptian Hieroglyphs (Kaggle)",
            "type": "kaggle",
            "url": "https://www.kaggle.com/datasets/nabilhaggag/egyptian-hieroglyphs",
            "dataset_id": "nabilhaggag/egyptian-hieroglyphs",
            "output": "kaggle_egyptian_hieroglyphs"
        },
        {
            "name": "Neural Style Transfer Synthetic Dataset",
            "type": "direct",
            "url": "https://arxiv.org/pdf/2504.03240.pdf",
            "output": "arxiv_2504.03240.pdf"
        },
    ]
    
    results = []
    
    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"Processing: {dataset['name']}")
        print(f"{'='*80}")
        
        success = False
        if dataset['type'] == 'huggingface':
            success = download_huggingface_dataset(dataset['name'], dataset['output'])
        elif dataset['type'] == 'github':
            success = download_github_repo(dataset['url'], dataset['output'], dataset.get('subpath'))
        elif dataset['type'] == 'kaggle':
            success = download_kaggle_dataset(dataset['dataset_id'], dataset['output'])
        elif dataset['type'] == 'roboflow':
            success = download_roboflow_dataset(dataset['url'], dataset['output'])
        elif dataset['type'] == 'direct':
            success = download_file(dataset['url'], dataset['output'])
        
        results.append({
            "name": dataset['name'],
            "success": success
        })
    
    # Summary
    print("\n" + "=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)
    for result in results:
        status = "✓" if result['success'] else "✗"
        print(f"{status} {result['name']}")
    
    successful = sum(1 for r in results if r['success'])
    print(f"\nSuccessfully downloaded: {successful}/{len(results)} datasets")
    print(f"Datasets saved to: {DATASETS_DIR.absolute()}")

if __name__ == "__main__":
    main()

