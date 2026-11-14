"""
Dataset inspection and pilot download utility script.
"""

import os
import argparse
import sys
from pipeline.data.dataset_loader import FFppDatasetLoader
from pipeline.utils.logging import get_logger

logger = get_logger("inspect_dataset")

def main():
    parser = argparse.ArgumentParser(description="Inspect FaceForensics++ dataset directory.")
    parser.add_argument("--data-dir", type=str, default="./data", help="Path to FF++ directory.")
    parser.add_argument("--download-pilot", action="store_true", help="Download pilot subset from EU2.")
    args = parser.parse_args()

    if args.download_pilot:
        logger.info("Downloading pilot subset from EU2 server...")
        # Example pilot download command
        print("To download the pilot subset, run:")
        print("python faceforensics_download_v4.py ./data -d Face2Face -c c23 -n 10 --server EU2")
        return

    loader = FFppDatasetLoader(args.data_dir)
    print(f"\n--- FaceForensics++ Dataset Summary in {args.data_dir} ---")
    print(f"Total videos discovered: {len(loader)}")
    
    categories = set(v.category for v in loader.videos)
    for cat in sorted(categories):
        subset = loader.filter(categories=[cat])
        print(f"  * {cat}: {len(subset)} videos")
        
    compressions = set(v.compression for v in loader.videos)
    for comp in sorted(compressions):
        subset = loader.filter(compressions=[comp])
        print(f"  * Compression {comp}: {len(subset)} videos")

if __name__ == "__main__":
    main()
