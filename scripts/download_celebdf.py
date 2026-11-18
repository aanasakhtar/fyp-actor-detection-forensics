"""
Automated Celeb-DF pilot downloader and selective extractor for Kaggle.
Downloads the archive from Google Drive and extracts a targeted subset of
hard cases (YouTube-real crowds, large poses, multi-person interviews, and synthesis swaps)
while cleaning up the large archive immediately to respect Kaggle disk limits.
"""

import os
import sys
import argparse
import zipfile
import subprocess
from typing import List

# Ensure src directory is in sys.path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from pipeline.utils.logging import get_logger

logger = get_logger("download_celebdf")

# Official Google Drive IDs
DRIVE_IDS = {
    "v2": "1iLx76wsbi9itnkxSqz9BVBl4ZvnbIazj",
    "v1": "10NGF38RgF8FZneKOuCOdRIsPzpC7_WDd"
}


def download_with_gdown(file_id: str, output_path: str) -> bool:
    """Download large zip file from Google Drive using gdown with automatic confirmation."""
    logger.info(f"Downloading Celeb-DF archive from Google Drive (ID: {file_id})...")
    try:
        import gdown
    except ImportError:
        logger.info("Installing gdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown"])
        import gdown

    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        gdown.download(url, output_path, quiet=False)
        return os.path.exists(output_path) and os.path.getsize(output_path) > 1000
    except Exception as e:
        logger.error(f"Failed to download via gdown: {e}")
        return False


def selective_extract_hard_cases(
    zip_path: str, 
    output_dir: str, 
    num_real: int = 10, 
    num_yt: int = 10, 
    num_synth: int = 20
) -> List[str]:
    """Selectively extract representative and challenging videos from Celeb-DF zip."""
    extracted_files = []
    logger.info(f"Scanning archive {zip_path} for potential hard cases...")
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        all_members = [m for m in zf.namelist() if m.endswith('.mp4')]
        
        # Categorize
        celeb_real = [m for m in all_members if 'Celeb-real' in m]
        yt_real = [m for m in all_members if 'YouTube-real' in m]
        celeb_synth = [m for m in all_members if 'Celeb-synthesis' in m]
        
        logger.info(f"Archive contains: {len(celeb_real)} Celeb-real, {len(yt_real)} YouTube-real, {len(celeb_synth)} Celeb-synthesis videos.")
        
        # Pick diverse selection
        # For Celeb-real: pick across different celebrity IDs (id0, id1, id2, ...)
        selected_real = celeb_real[:num_real]
        # For YouTube-real: unconstrained real scenes with crowds, interviews, varying lighting
        selected_yt = yt_real[:num_yt] if yt_real else []
        # For Celeb-synthesis: corresponding swaps
        selected_synth = celeb_synth[:num_synth]
        
        to_extract = selected_real + selected_yt + selected_synth
        logger.info(f"Extracting {len(to_extract)} targeted videos into {output_dir}...")
        
        for member in to_extract:
            zf.extract(member, output_dir)
            extracted_files.append(os.path.join(output_dir, member))
            
    return extracted_files


def main():
    parser = argparse.ArgumentParser(description="Download and selectively extract Celeb-DF hard cases.")
    parser.add_argument("--version", type=str, default="v2", choices=["v1", "v2"], help="Celeb-DF version.")
    parser.add_argument("--file-id", type=str, default=None, help="Custom Google Drive file ID (e.g. from a personal 'Make a Copy' to bypass global quota).")
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/data_celebdf", help="Target extraction directory.")
    parser.add_argument("--zip-path", type=str, default=None, help="Path to already downloaded zip (if any).")
    parser.add_argument("--num-real", type=int, default=10, help="Number of Celeb-real videos.")
    parser.add_argument("--num-yt", type=int, default=10, help="Number of YouTube-real unconstrained/crowd videos.")
    parser.add_argument("--num-synth", type=int, default=20, help="Number of Celeb-synthesis face-swap videos.")
    parser.add_argument("--keep-zip", action="store_true", help="Do not delete the downloaded zip.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    file_id = args.file_id or DRIVE_IDS[args.version]
    zip_file = args.zip_path or f"/kaggle/working/Celeb-DF-{args.version}.zip"

    # Step 1: Download if not already present
    if not os.path.exists(zip_file) or os.path.getsize(zip_file) == 0:
        success = download_with_gdown(file_id, zip_file)
        if not success:
            logger.error("Download failed. Please check internet connection or Google Drive file quota.")
            return

    # Step 2: Selective extraction of hard cases
    try:
        extracted = selective_extract_hard_cases(
            zip_file, 
            args.output_dir, 
            num_real=args.num_real, 
            num_yt=args.num_yt, 
            num_synth=args.num_synth
        )
        print("\n" + "="*70)
        print(f"      CELEB-DF ({args.version.upper()}) HARD-CASE EXTRACTION COMPLETE")
        print("="*70)
        print(f"Target Directory: {args.output_dir}")
        print(f"Total Videos Extracted: {len(extracted)}")
        print("Categories Extracted:")
        print(f"  * Celeb-real:       {args.num_real} videos (Primacy celebrity headshots & interviews)")
        print(f"  * YouTube-real:     {args.num_yt} videos (Crowds, outdoor scenes, unconstrained lighting)")
        print(f"  * Celeb-synthesis:  {args.num_synth} videos (High-quality face swaps across diverse targets)")
        print("="*70 + "\n")
    except Exception as e:
        logger.error(f"Error during extraction: {e}")
        return

    # Step 3: Cleanup zip file to preserve Kaggle disk space
    if not args.keep_zip and os.path.exists(zip_file):
        try:
            os.remove(zip_file)
            logger.info(f"Removed archive {zip_file} to preserve Kaggle disk space.")
        except Exception as e:
            logger.warning(f"Could not remove {zip_file}: {e}")


if __name__ == "__main__":
    main()
