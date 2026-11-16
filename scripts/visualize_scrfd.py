"""
SCRFD Face Detection Visualization and Reporting Utility.
Generates supervisor-ready visual figures with bounding boxes, 5-point landmarks,
track IDs, and dual-crop (ArcFace vs GenD) comparisons.
"""

import os
import sys
import argparse
import json
from typing import List, Dict, Any, Optional

# Ensure src directory is in sys.path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    import cv2
except ImportError:
    cv2 = None

import numpy as np
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None
    plt = None

from pipeline.data.dataset_loader import FFppDatasetLoader
from pipeline.data.video_reader import VideoReader
from pipeline.detection.scrfd_detector import SCRFDDetector
from pipeline.tracking.face_tracking import IoUFaceTracker
from pipeline.preprocessing.face_alignment import align_face_5point
from pipeline.preprocessing.face_cropping import crop_context_face
from pipeline.visualization.detection_visualization import draw_detections
from pipeline.utils.logging import get_logger

logger = get_logger("visualize_scrfd")


def inspect_logged_detections(scrfd_results_dir: str):
    """Print an exhaustive statistical summary of saved SCRFD detection JSONs."""
    if not os.path.exists(scrfd_results_dir):
        logger.warning(f"Detection results directory '{scrfd_results_dir}' not found.")
        return

    json_files = [f for f in os.listdir(scrfd_results_dir) if f.endswith("_detections.json")]
    if not json_files:
        logger.warning(f"No *_detections.json files found in {scrfd_results_dir}")
        return

    total_videos = len(json_files)
    total_frames = 0
    total_faces = 0
    confs = []
    category_faces: Dict[str, int] = {}
    category_videos: Dict[str, int] = {}

    for jf in json_files:
        with open(os.path.join(scrfd_results_dir, jf), "r") as f:
            data = json.load(f)
        cat = data.get("category", "unknown")
        category_videos[cat] = category_videos.get(cat, 0) + 1
        dets = data.get("detections", [])
        
        frame_indices = set()
        for d in dets:
            total_faces += 1
            confs.append(d.get("confidence", 0.0))
            category_faces[cat] = category_faces.get(cat, 0) + 1
            frame_indices.add(d.get("frame_index"))
            
        total_frames += len(frame_indices)

    avg_conf = np.mean(confs) if confs else 0.0
    min_conf = np.min(confs) if confs else 0.0
    max_conf = np.max(confs) if confs else 0.0

    print("\n" + "="*70)
    print("       STAGE 1: SCRFD DETECTION & LOCALIZATION SUMMARY LOG")
    print("="*70)
    print(f"Total Videos Processed:    {total_videos}")
    print(f"Total Faces Extracted:     {total_faces}")
    print(f"Average Face Confidence:   {avg_conf:.4f} (Min: {min_conf:.3f}, Max: {max_conf:.3f})")
    print(f"Avg Faces Detected/Video:  {total_faces / total_videos:.1f}")
    
    print("\nBreakdown by Manipulation Category:")
    print(f"{'Category':<18} | {'Videos':>6} | {'Faces Detected':>14} | {'Faces/Video':>11}")
    print("-" * 57)
    for cat in sorted(category_videos.keys()):
        v_cnt = category_videos[cat]
        f_cnt = category_faces.get(cat, 0)
        print(f"{cat:<18} | {v_cnt:>6} | {f_cnt:>14} | {f_cnt/v_cnt:>11.1f}")
    print("="*70 + "\n")


def generate_visualizations(
    data_dir: str, 
    output_dir: str, 
    max_samples: int = 6,
    sample_index: int = 0,
    video_id: Optional[str] = None
):
    """Generate annotated frames and a multi-panel supervisor report figure."""
    os.makedirs(output_dir, exist_ok=True)
    loader = FFppDatasetLoader(data_dir)
    
    if len(loader) == 0:
        logger.error(f"No videos found in {data_dir} for visualization.")
        return

    detector = SCRFDDetector(conf_threshold=0.50)
    tracker = IoUFaceTracker()

    if video_id:
        selected_videos = [v for v in loader.videos if video_id in v.video_id][:max_samples]
        if not selected_videos:
            logger.warning(f"No video matching ID '{video_id}' found in {data_dir}.")
            return
    else:
        # Pick sample video from each category based on sample_index
        categories = sorted(list(set(v.category for v in loader.videos)))
        selected_videos = []
        for cat in categories:
            vids = [v for v in loader.videos if v.category == cat]
            if vids:
                pick_idx = sample_index % len(vids)
                selected_videos.append(vids[pick_idx])
                if len(selected_videos) >= max_samples:
                    break

    print(f"Generating visual overlays for {len(selected_videos)} representative videos (sample_index={sample_index})...")

    report_items = []

    for meta in selected_videos:
        try:
            reader = VideoReader(meta.video_path)
            frames = reader.sample_uniform(max_frames=16)
            if not frames:
                continue

            # Pick a middle frame with clear face
            target_frame = frames[len(frames) // 2]
            dets = detector.detect_frame(target_frame.image_bgr, target_frame.frame_index, target_frame.timestamp_sec)
            dets = tracker.update(dets, target_frame.frame_index)

            if not dets:
                continue

            primary_det = dets[0]
            annotated_frame = draw_detections(target_frame.image_bgr, dets)

            # Generate ArcFace 112x112 similarity-aligned crop
            aligned_112 = None
            if primary_det.landmarks_5 is not None:
                aligned_112 = align_face_5point(target_frame.image_bgr, primary_det.landmarks_5, output_size=112)

            # Generate GenD 224x224 contextual crop (1.3x margin)
            context_224 = crop_context_face(target_frame.image_bgr, primary_det.bbox, expansion_factor=1.30, target_size=(224, 224))

            # Save individual annotated frame
            save_name = f"{meta.category}_{meta.video_id}_annotated.jpg"
            save_path = os.path.join(output_dir, save_name)
            cv2.imwrite(save_path, annotated_frame)

            report_items.append({
                "category": meta.category,
                "video_id": meta.video_id,
                "confidence": primary_det.confidence,
                "annotated_bgr": annotated_frame,
                "aligned_112": aligned_112,
                "context_224": context_224
            })
            print(f"  [+] Saved {save_name} (conf={primary_det.confidence:.2f})")

        except Exception as e:
            logger.error(f"Error visualizing {meta.video_id}: {e}")

    # Create Supervisor Multi-Panel Figure
    if report_items:
        create_supervisor_summary_figure(report_items, os.path.join(output_dir, "scrfd_stage1_supervisor_report.png"))


def create_supervisor_summary_figure(items: List[Dict[str, Any]], output_filepath: str):
    """
    Creates a publication-quality composite figure showing:
    - Detected frames with Bounding Box, Confidence Score, and 5 Facial Landmarks
    - Stage 2 ArcFace Aligned 112x112 Face Crop
    - Stage 3 GenD Contextual 224x224 Face Crop (1.3x Margin)
    """
    n_cols = len(items)
    fig, axes = plt.subplots(3, n_cols, figsize=(4.5 * n_cols, 11), squeeze=False)

    plt.suptitle("FYP Stage 1: SCRFD Face Localization & Dual-Stream Crop Preprocessing", fontsize=16, fontweight="bold", y=0.98)

    for col_idx, item in enumerate(items):
        cat = item["category"]
        conf = item["confidence"]

        # Row 1: Full Frame with Bounding Box & 5 Landmarks
        ax1 = axes[0, col_idx]
        rgb_frame = cv2.cvtColor(item["annotated_bgr"], cv2.COLOR_BGR2RGB)
        ax1.imshow(rgb_frame)
        ax1.set_title(f"[{cat}]\nConf: {conf:.2f} | 5 Landmarks", fontsize=11, fontweight="semibold")
        ax1.axis("off")

        # Row 2: ArcFace 112x112 Aligned Crop
        ax2 = axes[1, col_idx]
        if item["aligned_112"] is not None:
            rgb_arc = cv2.cvtColor(item["aligned_112"], cv2.COLOR_BGR2RGB)
            ax2.imshow(rgb_arc)
            ax2.set_title("ArcFace Input\n(112x112 Similarity Aligned)", fontsize=10)
        else:
            ax2.text(0.5, 0.5, "Alignment N/A", ha="center", va="center")
        ax2.axis("off")

        # Row 3: GenD 224x224 Contextual Crop
        ax3 = axes[2, col_idx]
        if item["context_224"] is not None:
            rgb_gend = cv2.cvtColor(item["context_224"], cv2.COLOR_BGR2RGB)
            ax3.imshow(rgb_gend)
            ax3.set_title("GenD Input\n(224x224 Contextual 1.3x)", fontsize=10)
        else:
            ax3.text(0.5, 0.5, "Crop N/A", ha="center", va="center")
        ax3.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig(output_filepath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n[SUCCESS] Composite Supervisor Report Figure saved to:")
    print(f"  {output_filepath}")


def main():
    parser = argparse.ArgumentParser(description="Visualize SCRFD face detections and export supervisor summary figures.")
    parser.add_argument("--data-dir", type=str, default="/kaggle/working/data", help="Directory of FF++ videos.")
    parser.add_argument("--scrfd-dir", type=str, default="/kaggle/working/results/scrfd", help="Directory with detection JSONs.")
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/results/visualizations", help="Output directory for images.")
    parser.add_argument("--num-samples", type=int, default=6, help="Number of representative videos to visualize.")
    parser.add_argument("--sample-index", type=int, default=0, help="0-based index to pick different sample videos per category.")
    parser.add_argument("--video-id", type=str, default=None, help="Target specific video ID (e.g. '183' or '585_599').")
    parser.add_argument("--list-videos", action="store_true", help="List all available video IDs and exit.")
    args = parser.parse_args()

    if args.list_videos:
        loader = FFppDatasetLoader(args.data_dir)
        print(f"\nDiscovered {len(loader)} total videos in {args.data_dir}:")
        categories = sorted(list(set(v.category for v in loader.videos)))
        for cat in categories:
            vids = [v for v in loader.videos if v.category == cat]
            ids = [v.video_id for v in vids]
            print(f"  * {cat:<16}: {', '.join(ids)}")
        return

    # 1. Inspect logged detection data
    inspect_logged_detections(args.scrfd_dir)

    # 2. Generate visualization figures
    generate_visualizations(
        args.data_dir, 
        args.output_dir, 
        max_samples=args.num_samples,
        sample_index=args.sample_index,
        video_id=args.video_id
    )


if __name__ == "__main__":
    main()
