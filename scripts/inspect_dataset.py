"""
Dataset inspection and pilot download utility script.
Supports both local directory inspection and exhaustive remote server inspection.
"""

import os
import sys

# Ensure local 'src' directory takes priority in sys.path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import argparse
import json
import urllib.request
from pipeline.data.dataset_loader import FFppDatasetLoader
from pipeline.utils.logging import get_logger

logger = get_logger("inspect_dataset")

EU2_BASE_URL = "https://kaldir.vc.in.tum.de/faceforensics/v3"
FILELIST_URL = f"{EU2_BASE_URL}/misc/filelist.json"
DFDC_URL = f"{EU2_BASE_URL}/misc/deepfake_detection_filenames.json"

def inspect_remote_server():
    """Query the official EU2 server and print an exhaustive breakdown of FF++."""
    logger.info("Connecting to official FaceForensics++ EU2 server for remote inspection...")
    
    # 1. Standard FF++ Pairs
    req = urllib.request.Request(FILELIST_URL, headers={"User-Agent": "Mozilla/5.0"})
    pairs = json.loads(urllib.request.urlopen(req).read().decode("utf-8"))
    
    all_original_ids = set()
    for p in pairs:
        all_original_ids.update(p)

    # 2. Google / Jigsaw DeepFakeDetection data
    req_dfdc = urllib.request.Request(DFDC_URL, headers={"User-Agent": "Mozilla/5.0"})
    dfdc_data = json.loads(urllib.request.urlopen(req_dfdc).read().decode("utf-8"))
    actor_count = len(dfdc_data.get("actors", []))
    dfdc_count = len(dfdc_data.get("DeepFakesDetection", []))

    print("\n" + "="*70)
    print("      EXHAUSTIVE FACEFORENSICS++ (FF++) REMOTE SERVER INVENTORY")
    print("="*70)
    print(f"Official Mirror: {EU2_BASE_URL}")
    print(f"Available Compression Tiers: raw (c0 / lossless), c23 (high quality), c40 (low quality)")
    
    print("\n--- 1. Standard FaceForensics++ Benchmark ---")
    print(f"Unique Pristine YouTube Source Videos: {len(all_original_ids)} (IDs {min(int(x) for x in all_original_ids):03d} to {max(int(x) for x in all_original_ids):03d})")
    print(f"Official Target-Source Video Pairs:     {len(pairs)} pairs")
    print(f"Manipulated Sequences per Method:      1,000 videos (each pair generated as A_B and B_A)")
    
    categories = [
        ("original", "Pristine real YouTube talking-head videos", 1000),
        ("Deepfakes", "Autoencoder-based face replacement (FaceSwap style)", 1000),
        ("Face2Face", "Facial reenactment / expression transfer", 1000),
        ("FaceSwap", "Graphics-based 3D face swapping", 1000),
        ("NeuralTextures", "Photometric facial reenactment via neural rendering", 1000),
        ("FaceShifter", "High-fidelity two-stage face swapping (AEI-Net)", 1000),
    ]
    
    print("\nCategories available on server:")
    for cat, desc, count in categories:
        print(f"  * {cat:<16} | {count:>5} videos | {desc}")

    print("\n--- 2. DeepFake Detection (Google/Jigsaw Release) ---")
    print(f"  * Pristine Actor Videos:     {actor_count:>5} videos")
    print(f"  * Manipulated Actor Videos:  {dfdc_count:>5} videos")

    print("\n--- 3. Dataset Footprint Estimates ---")
    print("  * Pilot Subset (20-50 videos per category, c23):  ~200 MB - 600 MB (Recommended for Kaggle)")
    print("  * Full c23 Compressed Benchmark (~6,000 videos):   ~12 GB - 16 GB")
    print("  * Full c40 Compressed Benchmark (~6,000 videos):   ~4 GB - 6 GB")
    print("  * Raw / Uncompressed Benchmark (Lossless):         ~1.5 TB - 2.0 TB")
    print("="*70 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Inspect FaceForensics++ dataset.")
    parser.add_argument("--data-dir", type=str, default="./data", help="Path to local FF++ directory.")
    parser.add_argument("--remote", action="store_true", help="Perform exhaustive remote inspection of official EU2 server.")
    parser.add_argument("--download-pilot", action="store_true", help="Download pilot subset.")
    args = parser.parse_args()

    if args.remote:
        inspect_remote_server()
        return

    loader = FFppDatasetLoader(args.data_dir)
    print(f"\n--- FaceForensics++ Dataset Summary in {args.data_dir} ---")
    print(f"Total videos discovered: {len(loader)}")
    
    if len(loader) == 0:
        print(f"\n[!] No FaceForensics++ videos found in '{args.data_dir}'.")
        print("    1. To inspect the full official FF++ server without downloading, run:")
        print("       python scripts/inspect_dataset.py --remote")
        print("    2. To download a fast pilot subset into /kaggle/working/data, run:")
        print("       python scripts/download_ffpp.py --output-dir /kaggle/working/data --num-videos 5")
        return

    categories = set(v.category for v in loader.videos)
    for cat in sorted(categories):
        subset = loader.filter(categories=[cat])
        print(f"  * {cat:<16}: {len(subset):>4} videos")
        
    compressions = set(v.compression for v in loader.videos)
    for comp in sorted(compressions):
        subset = loader.filter(compressions=[comp])
        print(f"  * Compression {comp:<5}: {len(subset):>4} videos")

    print("\nSample discovered videos:")
    for v in loader.videos[:5]:
        pair_info = f"target={v.target_id}, donor={v.donor_id}" if v.is_manipulated else f"target={v.target_id}"
        print(f"  - [{v.category}] {v.video_id} ({pair_info}) -> {v.path}")

if __name__ == "__main__":
    main()
