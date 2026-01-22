"""
Download Helper - Automatically download large model files from external storage.

This module handles downloading large files (model weights, FAISS index, datasets)
from external sources (Google Drive, AWS S3, etc.) when they are missing locally.

This is necessary for Vercel deployment because:
- GitHub has a 100MB file size limit
- Vercel has a ~250MB total deployment size limit
- Our model files total ~1.5GB

Author: Graduation Project
Created: 2026-01-19
"""

import os
import sys
from pathlib import Path
from typing import Optional
import gdown  # pip install gdown


# =============================================================================
# FILE CONFIGURATIONS
# =============================================================================

# Google Drive file IDs (YOU NEED TO REPLACE THESE WITH YOUR OWN)
# How to get Google Drive file ID:
# 1. Upload file to Google Drive
# 2. Right-click → Share → Get link
# 3. URL looks like: https://drive.google.com/file/d/FILE_ID_HERE/view
# 4. Copy the FILE_ID_HERE part

FILE_CONFIGS = {
    "model": {
        "url": "https://drive.google.com/uc?id=1UN4sI6go4FlZ7xF_mNEZNA8PJcw3LGKr",  # best_model.pth
        "path": "../models/best_model.pth",
        "size_mb": 5,
        "description": "Trained PyTorch model weights"
    },
    "faiss_index": {
        "url": "https://drive.google.com/uc?id=1WCndvZhKEnAwGWZFAVb4dn4lLBEAN6QA",  # faiss_index.bin
        "path": "../models/faiss_index.bin",
        "size_mb": 135,
        "description": "FAISS similarity search index"
    },
    "faiss_mappings": {
        "url": "https://drive.google.com/uc?id=15MzEnaHwfXdQUsPkcxoYcdjYKkt4KYik",  # mappings.pkl
        "path": "../models/faiss_index.bin.mappings.pkl",
        "size_mb": 71,
        "description": "FAISS index metadata mappings"
    },
    # "train_data": {
    #     "url": "https://drive.google.com/uc?id=YOUR_DATA_FILE_ID",  # train.csv
    #     "path": "../data/processed/train.csv",
    #     "size_mb": 964,
    #     "description": "Song database (441K songs)"
    # },
    # NOTE: train.csv is OPTIONAL. If missing, app falls back to models/song_metadata.csv (29MB)
    # which is included in the Git repo. This saves ~1GB of download size.
    "audio_stats": {
        "url": "https://drive.google.com/uc?id=YOUR_STATS_FILE_ID",  # audio_stats.json
        "path": "../models/audio_stats.json",
        "size_mb": 0.001,
        "description": "Audio feature normalization statistics"
    }
}


def get_absolute_path(relative_path: str) -> Path:
    """
    Convert relative path to absolute path from this script's location.
    
    Args:
        relative_path: Path relative to web_app/ directory
        
    Returns:
        Absolute Path object
    """
    script_dir = Path(__file__).parent
    return (script_dir / relative_path).resolve()


def download_file(url: str, output_path: Path, description: str = "file") -> bool:
    """
    Download a file from Google Drive using gdown.
    
    Args:
        url: Google Drive direct download URL
        output_path: Where to save the downloaded file
        description: Human-readable description for logging
        
    Returns:
        True if download successful, False otherwise
    """
    try:
        print(f"\n{'='*60}")
        print(f"📥 Downloading {description}...")
        print(f"   Target: {output_path}")
        print(f"   URL: {url[:50]}...")
        print(f"{'='*60}")
        
        # Create parent directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download using gdown
        gdown.download(url, str(output_path), quiet=False)
        
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"✅ Successfully downloaded {description} ({size_mb:.1f} MB)")
            return True
        else:
            print(f"❌ Download failed: File not created")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading {description}: {e}")
        return False


def check_and_download_all() -> bool:
    """
    Check all required files and download missing ones.
    
    Returns:
        True if all files are available (either exist or successfully downloaded),
        False if any required file is missing and could not be downloaded.
    """
    print("\n" + "="*60)
    print("🔍 CHECKING REQUIRED FILES")
    print("="*60)
    
    all_success = True
    files_to_download = []
    
    # Check which files are missing
    for file_key, config in FILE_CONFIGS.items():
        file_path = get_absolute_path(config["path"])
        
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"✅ {config['description']}: Found ({size_mb:.1f} MB)")
        else:
            print(f"❌ {config['description']}: Missing")
            files_to_download.append((file_key, config))
    
    # Download missing files
    if files_to_download:
        total_size = sum(cfg["size_mb"] for _, cfg in files_to_download)
        print(f"\n⚠️  Need to download {len(files_to_download)} files (~{total_size:.0f} MB total)")
        print("⏱️  This may take 5-15 minutes depending on your internet speed...")
        
        for file_key, config in files_to_download:
            file_path = get_absolute_path(config["path"])
            
            # Check if URL is configured
            if "YOUR_" in config["url"]:
                print(f"\n❌ ERROR: {config['description']}")
                print(f"   Please update the Google Drive URL in download_helper.py")
                print(f"   File key: {file_key}")
                all_success = False
                continue
            
            # Download the file
            success = download_file(
                url=config["url"],
                output_path=file_path,
                description=config["description"]
            )
            
            if not success:
                all_success = False
    else:
        print("\n✅ All required files are present!")
    
    return all_success


def ensure_files_ready() -> None:
    """
    Ensure all required files are downloaded and ready.
    Exits the program if any file is missing and cannot be downloaded.
    
    This function should be called during application startup.
    """
    success = check_and_download_all()
    
    if not success:
        print("\n" + "="*60)
        print("❌ FATAL ERROR: Required files are missing")
        print("="*60)
        print("\nPlease follow these steps:")
        print("1. Upload the following files to Google Drive:")
        print("   - models/best_model.pth")
        print("   - models/faiss_index.bin")
        print("   - models/faiss_index.bin.mappings.pkl")
        print("   - data/processed/train.csv")
        print("   - models/audio_stats.json")
        print("\n2. Make files publicly accessible (Anyone with link can view)")
        print("\n3. Get the file IDs from the share links")
        print("\n4. Update FILE_CONFIGS in download_helper.py with your file IDs")
        print("\n5. Restart the application")
        print("="*60)
        sys.exit(1)
    
    print("\n" + "="*60)
    print("✅ ALL FILES READY - Application can start")
    print("="*60)


if __name__ == "__main__":
    """
    Test the download helper by running:
        python download_helper.py
    """
    ensure_files_ready()
