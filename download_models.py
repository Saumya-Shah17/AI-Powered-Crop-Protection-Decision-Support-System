#!/usr/bin/env python3
"""
Model Download Script for CropGuard
This script helps download YOLOv8 model files if they're missing.
"""

import os
import requests
import sys
from pathlib import Path

# Model URLs (replace with actual URLs when you host them)
MODEL_URLS = {
    "Step-1-yolov8.pt": "https://example.com/models/Step-1-yolov8.pt",
    "Step-2-yolov8.pt": "https://example.com/models/Step-2-yolov8.pt", 
    "Apple.pt": "https://example.com/models/Apple.pt",
    "coconut.pt": "https://example.com/models/coconut.pt",
    "Grape.pt": "https://example.com/models/Grape.pt",
    "jowar.pt": "https://example.com/models/jowar.pt",
    "Tomato.pt": "https://example.com/models/Tomato.pt"
}

def download_file(url, filename, model_dir):
    """Download a file with progress bar"""
    filepath = model_dir / filename
    try:
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}%", end='', flush=True)
        print(f"\n✅ Downloaded {filename}")
        return True
    except Exception as e:
        print(f"\n❌ Failed to download {filename}: {e}")
        return False

def main():
    """Main function to check and download missing models"""
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)
    
    print("🔍 Checking for missing model files...")
    
    missing_models = []
    for model_name in MODEL_URLS.keys():
        if not (model_dir / model_name).exists():
            missing_models.append(model_name)
            print(f"❌ Missing: {model_name}")
        else:
            print(f"✅ Found: {model_name}")
    
    if not missing_models:
        print("\n🎉 All model files are present!")
        return
    
    print(f"\n📥 Found {len(missing_models)} missing model(s)")
    print("Note: This script requires actual download URLs.")
    print("Please contact the maintainer for model files or download from releases.")
    
    # Uncomment and update URLs when you have them hosted
    # for model_name in missing_models:
    #     url = MODEL_URLS[model_name]
    #     download_file(url, model_name, model_dir)

if __name__ == "__main__":
    main()
