"""
Automated, non-blocking FaceForensics++ pilot subset downloader for Kaggle.
Downloads directly from the official EU2 server (TUM).
"""

import os
import sys
import argparse
import json
import urllib.request
from typing import List

# Ensure src directory is on sys.path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from pipeline.utils.logging import get_logger

logger = get_logger("download_ffpp")

BASE_URL = "https://kaldir.vc.in.tum.de/faceforensics/v3"
FILELIST_URL = f"{BASE_URL}/misc/filelist.json"

DATASET_PATHS = {
    "original": "original_sequences/youtube",
    "Deepfakes": "manipulated_sequences/Deepfakes",
    "Face2Face": "manipulated_sequences/Face2Face",
    "FaceSwap": "manipulated_sequences/FaceSwap",
    "NeuralTextures": "manipulated_sequences/NeuralTextures",
    "FaceShifter": "manipulated_sequences/FaceShifter"
}

def download_file(url: str, output_path: str) -> bool:
    """Download single file with overwrite guard."""
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return True
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(output_path, "wb") as f:
            f.write(resp.read())
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download FaceForensics++ pilot subset from EU2.")
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/data", help="Target dataset directory.")
    parser.add_argument("--categories", nargs="+", default=["original", "Face2Face", "Deepfakes", "FaceSwap"], 
                        help="Categories to download.")
    parser.add_argument("--compression", type=str, default="c23", choices=["c23", "c40", "raw"])
    parser.add_argument("--num-videos", type=int, default=5, help="Number of videos per category.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Connecting to official FaceForensics++ EU2 server ({BASE_URL})...")

    # Fetch official file pairs
    req = urllib.request.Request(FILELIST_URL, headers={"User-Agent": "Mozilla/5.0"})
    file_pairs = json.loads(urllib.request.urlopen(req).read().decode("utf-8"))
    logger.info(f"Retrieved {len(file_pairs)} official video pairs.")

    for cat in args.categories:
        if cat not in DATASET_PATHS:
            logger.warning(f"Unknown category '{cat}', skipping.")
            continue

        rel_path = DATASET_PATHS[cat]
        target_dir = os.path.join(args.output_dir, rel_path, args.compression, "videos")
        os.makedirs(target_dir, exist_ok=True)

        if cat == "original":
            # List of single video IDs: 585.mp4, 599.mp4, ...
            filenames = []
            for pair in file_pairs:
                for vid in pair:
                    if f"{vid}.mp4" not in filenames:
                        filenames.append(f"{vid}.mp4")
        else:
            # List of pair video IDs: 585_599.mp4, 599_585.mp4
            filenames = []
            for pair in file_pairs:
                filenames.append(f"{pair[0]}_{pair[1]}.mp4")
                filenames.append(f"{pair[1]}_{pair[0]}.mp4")

        selected = filenames[:args.num_videos]
        logger.info(f"Downloading {len(selected)} videos for {cat} ({args.compression})...")

        for fn in selected:
            url = f"{BASE_URL}/{rel_path}/{args.compression}/videos/{fn}"
            dest = os.path.join(target_dir, fn)
            success = download_file(url, dest)
            if success:
                size_mb = os.path.getsize(dest) / (1024 * 1024)
                print(f"  [+] Downloaded {fn} ({size_mb:.2f} MB)")

    logger.info(f"Pilot subset download complete. Files saved in {args.output_dir}")

if __name__ == "__main__":
    main()
