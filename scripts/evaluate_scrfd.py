"""
Comprehensive evaluation script to isolate and benchmark SCRFD face detection
across FaceForensics++ and Celeb-DF datasets.
Evaluates:
  1. Face Acquisition Rate (Coverage % overall and per manipulation category)
  2. Detection Confidence Distribution (p_det)
  3. Landmark Extraction & Geometric Plausibility
  4. Face Resolution / Scale (Inter-Pupillary Distance - IPD)
  5. Multi-Face Scene Prevalence (background vs primary faces)
  6. Temporal Bounding Box Stability (Consecutive frame IoU)
  7. Inference Throughput (Frames Per Second)
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, List, Any
import numpy as np

# Ensure local 'src' directory takes priority in sys.path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from pipeline.data.dataset_loader import FFppDatasetLoader
from pipeline.data.video_reader import VideoReader
from pipeline.detection.scrfd_detector import SCRFDDetector
from pipeline.evaluation.detection_metrics import (
    compute_detection_coverage,
    compute_detection_confidence_summary,
    compute_bbox_iou,
    compute_inter_pupillary_distance,
    compute_landmark_geometric_validity
)
from pipeline.visualization.evaluation_plots import plot_scrfd_cross_dataset_evaluation
from pipeline.utils.logging import get_logger

logger = get_logger("evaluate_scrfd")


def evaluate_dataset(
    data_dir: str, 
    detector: SCRFDDetector, 
    dataset_name: str, 
    max_frames: int = 16,
    max_videos: int = None
) -> Dict[str, Any]:
    """Isolate face detection metrics for a single dataset directory."""
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        logger.warning(f"Dataset path '{data_dir}' does not exist or is empty. Skipping.")
        return {}

    loader = FFppDatasetLoader(data_dir)
    videos = loader.videos
    if max_videos and len(videos) > max_videos:
        reals = [v for v in loader.videos if not v.is_fake]
        fakes = [v for v in loader.videos if v.is_fake]
        n_half = max_videos // 2
        rng = np.random.RandomState(42)
        sample_reals = list(rng.choice(reals, min(len(reals), n_half), replace=False)) if reals else []
        sample_fakes = list(rng.choice(fakes, min(len(fakes), max_videos - len(sample_reals)), replace=False)) if fakes else []
        videos = sample_reals + sample_fakes

    logger.info(f"Evaluating SCRFD on {dataset_name} ({len(videos)} videos, max {max_frames} frames/video)...")

    total_frames = 0
    frames_with_face = 0
    multi_face_frames = 0
    confidences = []
    ipds = []
    ious = []
    landmark_valid_count = 0
    total_faces_detected = 0

    # Per-category metrics
    category_frames: Dict[str, int] = {}
    category_covered: Dict[str, int] = {}

    start_time = time.time()

    for idx, meta in enumerate(videos):
        cat = meta.category
        category_frames.setdefault(cat, 0)
        category_covered.setdefault(cat, 0)

        try:
            reader = VideoReader(meta.video_path)
            frames = reader.sample_uniform(max_frames=max_frames)
            if not frames:
                continue

            prev_primary_bbox = None

            for frame in frames:
                total_frames += 1
                category_frames[cat] += 1

                dets = detector.detect_frame(frame.image_bgr, frame.frame_index, frame.timestamp_sec)
                
                if dets:
                    frames_with_face += 1
                    category_covered[cat] += 1
                    if len(dets) > 1:
                        multi_face_frames += 1

                    # Primary face is the largest bounding box (foreground actor)
                    primary_det = max(dets, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))

                    # Temporal tracking IoU stability against previous frame
                    if prev_primary_bbox is not None:
                        iou_val = compute_bbox_iou(prev_primary_bbox, primary_det.bbox)
                        ious.append(iou_val)
                    prev_primary_bbox = primary_det.bbox

                    for d in dets:
                        total_faces_detected += 1
                        confidences.append(float(d.confidence))
                        if d.landmarks_5 is not None:
                            ipd = compute_inter_pupillary_distance(d.landmarks_5)
                            ipds.append(ipd)
                            if compute_landmark_geometric_validity(d.landmarks_5, d.bbox):
                                landmark_valid_count += 1
                else:
                    prev_primary_bbox = None

        except Exception as e:
            logger.warning(f"Error processing video {meta.video_id}: {e}")

    elapsed = time.time() - start_time
    fps = total_frames / elapsed if elapsed > 0 else 0.0

    overall_coverage = frames_with_face / total_frames if total_frames > 0 else 0.0
    multi_face_rate = multi_face_frames / total_frames if total_frames > 0 else 0.0
    landmark_rate = landmark_valid_count / total_faces_detected if total_faces_detected > 0 else 0.0

    category_coverage_rates = {
        c: (category_covered[c] / category_frames[c]) if category_frames[c] > 0 else 0.0
        for c in category_frames
    }

    return {
        "dataset_name": dataset_name,
        "videos_evaluated": len(videos),
        "total_frames": total_frames,
        "total_faces_detected": total_faces_detected,
        "face_coverage_rate": round(overall_coverage, 4),
        "multi_face_scene_rate": round(multi_face_rate, 4),
        "landmark_validity_rate": round(landmark_rate, 4),
        "mean_confidence": round(float(np.mean(confidences)), 4) if confidences else 0.0,
        "std_confidence": round(float(np.std(confidences)), 4) if confidences else 0.0,
        "mean_ipd_pixels": round(float(np.mean(ipds)), 2) if ipds else 0.0,
        "mean_temporal_iou": round(float(np.mean(ious)), 4) if ious else 0.0,
        "inference_fps": round(fps, 1),
        "category_coverage": category_coverage_rates,
        "confidences": confidences,
        "ipds": ipds,
        "ious": ious
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate SCRFD Face Detector across datasets.")
    parser.add_argument("--data-dir", type=str, default=None, help="Generic dataset directory.")
    parser.add_argument("--ffpp-dir", type=str, default="/kaggle/working/data", help="FaceForensics++ dataset path.")
    parser.add_argument("--celebdf-dir", type=str, default="/kaggle/working/data_celebdf", help="Celeb-DF dataset path.")
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/results/scrfd_evaluation", help="Output directory for reports.")
    parser.add_argument("--max-frames", type=int, default=16, help="Frames sampled per video.")
    parser.add_argument("--max-videos", type=int, default=None, help="Max videos per dataset.")
    parser.add_argument("--conf-thresh", type=float, default=0.50, help="Confidence threshold.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    detector = SCRFDDetector(conf_threshold=args.conf_thresh)

    datasets_to_run = {}

    # Check explicitly passed single directory
    if args.data_dir and os.path.exists(args.data_dir):
        datasets_to_run["Dataset"] = args.data_dir
    else:
        # Check standard paths
        ffpp_candidates = [args.ffpp-dir if hasattr(args, "ffpp-dir") else None, args.ffpp_dir, "/kaggle/working/data", "/kaggle/input/faceforensics-pilot", "./data"]
        for p in ffpp_candidates:
            if p and os.path.exists(p) and os.listdir(p):
                datasets_to_run["FaceForensics++"] = p
                break

        celebdf_candidates = [args.celebdf_dir, "/kaggle/working/data_celebdf", "/kaggle/input/celeb-df-v2", "./data_celebdf"]
        for p in celebdf_candidates:
            if p and os.path.exists(p) and os.listdir(p):
                datasets_to_run["Celeb-DF"] = p
                break

    if not datasets_to_run:
        logger.error("No valid dataset directory found! Please provide --data-dir, --ffpp-dir, or --celebdf-dir.")
        return

    benchmark_results = {}

    for d_name, d_path in datasets_to_run.items():
        res = evaluate_dataset(
            data_dir=d_path,
            detector=detector,
            dataset_name=d_name,
            max_frames=args.max_frames,
            max_videos=args.max_videos
        )
        if res:
            benchmark_results[d_name] = res

    if not benchmark_results:
        logger.error("No evaluations completed.")
        return

    # Print publication-grade comparison table
    print("\n" + "="*84)
    print("      STAGE 1: SCRFD FACE DETECTION CROSS-DATASET BENCHMARK RESULTS")
    print("="*84)
    print(f"{'Metric':<38} " + " ".join([f"{name:<20}" for name in benchmark_results.keys()]))
    print("-" * 84)

    metrics_to_show = [
        ("Videos Evaluated", lambda r: f"{r['videos_evaluated']}"),
        ("Total Frames Evaluated", lambda r: f"{r['total_frames']}"),
        ("Total Faces Localized", lambda r: f"{r['total_faces_detected']}"),
        ("Face Acquisition / Coverage Rate", lambda r: f"{r['face_coverage_rate']*100:.2f}%"),
        ("Mean Detection Confidence (p_det)", lambda r: f"{r['mean_confidence']:.3f} ± {r['std_confidence']:.3f}"),
        ("Landmark Geometric Sanity Rate", lambda r: f"{r['landmark_validity_rate']*100:.2f}%"),
        ("Face Scale (Mean IPD pixels)", lambda r: f"{r['mean_ipd_pixels']:.1f} px"),
        ("Temporal Bounding Box IoU Stability", lambda r: f"{r['mean_temporal_iou']:.3f}"),
        ("Multi-Face Scene Prevalence Rate", lambda r: f"{r['multi_face_scene_rate']*100:.2f}%"),
        ("Inference Speed (Throughput)", lambda r: f"{r['inference_fps']:.1f} FPS"),
    ]

    for label, fn in metrics_to_show:
        vals = [f"{fn(r):<20}" for r in benchmark_results.values()]
        print(f"{label:<38} " + " ".join(vals))

    print("="*84)

    # Print Category Coverage breakdown
    print("\n[+] Detailed Face Coverage Rate by Category:")
    for d_name, r in benchmark_results.items():
        print(f"  --- {d_name} ---")
        for cat, cov in r["category_coverage"].items():
            print(f"    * {cat:<24}: {cov*100:.2f}%")
    print("="*84 + "\n")

    # Save summary JSON (excluding raw array lists for lightweight export)
    summary_export = {}
    for d_name, r in benchmark_results.items():
        summary_export[d_name] = {k: v for k, v in r.items() if k not in ["confidences", "ipds", "ious"]}

    metrics_file = os.path.join(args.output_dir, "scrfd_benchmark_summary.json")
    with open(metrics_file, "w") as f:
        json.dump(summary_export, f, indent=2)
    print(f"[+] Saved SCRFD benchmark metrics to: {metrics_file}")

    # Generate 4-panel diagnostic figure
    plot_file = os.path.join(args.output_dir, "scrfd_cross_dataset_evaluation.png")
    try:
        plot_scrfd_cross_dataset_evaluation(
            dataset_metrics=benchmark_results,
            title="Stage 1: SCRFD Cross-Dataset Face Detection Forensic Benchmark",
            save_path=plot_file
        )
        print(f"[+] Saved 4-panel evaluation figure to: {plot_file}")
    except Exception as e:
        logger.warning(f"Could not render evaluation figure: {e}")


if __name__ == "__main__":
    main()
