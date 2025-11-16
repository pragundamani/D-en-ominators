#!/usr/bin/env python3
"""
Google Colab-adapted script to download all datasets
Handles multiple sources: Hugging Face, GitHub, Kaggle, and direct downloads
"""

import os
import sys
import subprocess
import requests
from pathlib import Path
from urllib.parse import urlparse
import json
from tqdm import tqdm

# Create datasets directory
DATASETS_DIR = Path("datasets")
DATASETS_DIR.mkdir(exist_ok=True)

def run_command(cmd, check=True, show_progress=False):
    """Run a shell command and return the result"""
    try:
        if show_progress:
            process = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, text=True, bufsize=1
            )
            for line in process.stdout:
                print(f"  {line.strip()}")
            process.wait()
            if process.returncode != 0:
                print(f"  ✗ Command failed with exit code {process.returncode}")
                return False
            return True
        else:
            result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
            if result.returncode != 0 and check:
                print(f"  ✗ Command failed: {result.stderr}")
                return False
            return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {cmd}")
        print(f"Error: {e}")
        return False

def download_huggingface_dataset(dataset_name, output_dir):
    """Download a Hugging Face dataset"""
    print(f"\n📦 Downloading Hugging Face dataset: {dataset_name}")
    try:
        from datasets import load_dataset
        output_path = DATASETS_DIR / output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        
        with tqdm(desc="  Loading dataset", bar_format='{desc}: {elapsed}') as pbar:
            print(f"  Loading dataset {dataset_name}...")
            dataset = load_dataset(dataset_name)
            pbar.update(1)
        
        if isinstance(dataset, dict):
            splits = list(dataset.keys())
            with tqdm(total=len(splits), desc="  Saving splits", unit="split") as pbar:
                for split_name, split_data in dataset.items():
                    split_path = output_path / split_name
                    split_path.mkdir(exist_ok=True)
                    split_data.save_to_disk(str(split_path))
                    pbar.set_postfix({"split": split_name})
                    pbar.update(1)
                    print(f"  ✓ Saved {split_name} split")
        else:
            with tqdm(desc="  Saving dataset", bar_format='{desc}: {elapsed}') as pbar:
                dataset.save_to_disk(str(output_path))
                pbar.update(1)
                print(f"  ✓ Saved dataset")
        
        if not any(output_path.rglob("*")):
            print(f"  ✗ Warning: Dataset saved but no files found in {output_path}")
            return False
        
        print(f"  ✓ Verified files saved to {output_path}")
        return True
    except ImportError:
        print("  ⚠ Installing datasets library...")
        with tqdm(desc="  Installing", bar_format='{desc}: {elapsed}') as pbar:
            run_command(f"{sys.executable} -m pip install datasets", check=False)
            pbar.update(1)
        return download_huggingface_dataset(dataset_name, output_dir)
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def download_github_repo(repo_url, output_dir, subpath=None):
    """Download a GitHub repository or specific subdirectory"""
    print(f"\n📦 Downloading GitHub repository: {repo_url}")
    try:
        output_path = DATASETS_DIR / output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        
        if "github.com" in repo_url:
            repo_path = repo_url.split("github.com/")[1].split("/tree/")[0]
            if "/tree/" in repo_url:
                branch = repo_url.split("/tree/")[1].split("/")[0]
                subpath = "/".join(repo_url.split("/tree/")[1].split("/")[1:])
            else:
                branch = "main"
            
            repo_url_clean = f"https://github.com/{repo_path}.git"
            
            if output_path.exists() and any(output_path.iterdir()):
                print(f"  ⚠ Directory {output_path} already exists, skipping clone")
                return True
            else:
                print(f"  Cloning {repo_url_clean} (branch: {branch})...")
                with tqdm(desc="  Cloning repository", bar_format='{desc}: {elapsed}') as pbar:
                    success = run_command(f"git clone --depth 1 --branch {branch} {repo_url_clean} {output_path}", 
                              check=False, show_progress=True)
                    pbar.update(1)
                
                if not success:
                    print(f"  ✗ Failed to clone repository")
                    return False
                
                if not output_path.exists() or not any(output_path.iterdir()):
                    print(f"  ✗ Clone completed but directory is empty")
                    return False
                
                if subpath:
                    subpath_full = output_path / subpath
                    if subpath_full.exists():
                        print(f"  ✓ Found subpath: {subpath}")
                    else:
                        print(f"  ⚠ Subpath {subpath} not found")
                
                print(f"  ✓ Successfully cloned to {output_path}")
                return True
        else:
            print(f"  ✗ Invalid GitHub URL: {repo_url}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def download_kaggle_dataset(dataset_name, output_dir):
    """Download a Kaggle dataset"""
    print(f"\n📦 Downloading Kaggle dataset: {dataset_name}")
    try:
        output_path = DATASETS_DIR / output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            import kaggle
        except ImportError:
            print("  ⚠ Installing kaggle library...")
            with tqdm(desc="  Installing", bar_format='{desc}: {elapsed}') as pbar:
                run_command(f"{sys.executable} -m pip install kaggle", check=False)
                pbar.update(1)
            import kaggle
        
        print(f"  Downloading {dataset_name}...")
        with tqdm(desc="  Downloading from Kaggle", bar_format='{desc}: {elapsed}') as pbar:
            kaggle.api.dataset_download_files(dataset_name, path=str(output_path), unzip=True)
            pbar.update(1)
        
        if not any(output_path.rglob("*")):
            print(f"  ✗ Warning: Download completed but no files found in {output_path}")
            return False
        
        print(f"  ✓ Downloaded to {output_path}")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        print(f"  ⚠ Note: Kaggle requires API credentials. Upload kaggle.json to /content/.kaggle/")
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
        
        with open(output_file, 'wb') as f:
            if total_size > 0:
                with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024, 
                      desc="  Downloading", bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                with tqdm(unit='B', unit_scale=True, unit_divisor=1024, 
                         desc="  Downloading", bar_format='{desc}: {n_fmt} [{elapsed}]') as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
        
        print(f"  ✓ Downloaded to {output_file}")
        return True
    except Exception as e:
        print(f"\n  ✗ Error: {e}")
        return False

def main():
    """Main function to download all datasets"""
    print("=" * 80)
    print("Downloading all datasets for Google Colab")
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
            "name": "Neural Style Transfer Synthetic Dataset",
            "type": "direct",
            "url": "https://arxiv.org/pdf/2504.03240.pdf",
            "output": "arxiv_2504.03240.pdf"
        },
    ]
    
    results = []
    
    with tqdm(total=len(datasets), desc="Overall Progress", unit="dataset", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
              position=0, leave=True) as overall_pbar:
        for idx, dataset in enumerate(datasets, 1):
            overall_pbar.set_description(f"Processing: {dataset['name'][:35]}")
            overall_pbar.set_postfix({"current": f"{idx}/{len(datasets)}"})
            print(f"\n{'='*80}")
            print(f"Processing: {dataset['name']} ({idx}/{len(datasets)})")
            print(f"{'='*80}")
            
            success = False
            if dataset['type'] == 'huggingface':
                success = download_huggingface_dataset(dataset['name'], dataset['output'])
            elif dataset['type'] == 'github':
                success = download_github_repo(dataset['url'], dataset['output'], dataset.get('subpath'))
            elif dataset['type'] == 'kaggle':
                success = download_kaggle_dataset(dataset['dataset_id'], dataset['output'])
            elif dataset['type'] == 'direct':
                success = download_file(dataset['url'], dataset['output'])
            
            results.append({
                "name": dataset['name'],
                "success": success
            })
            
            overall_pbar.update(1)
    
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
