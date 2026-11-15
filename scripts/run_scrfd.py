"""
Standalone script to run SCRFD face detection on a dataset directory.
"""

import os
import sys

# Ensure local 'src' directory takes priority in sys.path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import argparse
import json
import time
from pipeline.data.dataset_loader import FFppDatasetLoader
from pipeline.data.video_reader import VideoReader
from pipeline.detection.scrfd_detector import SCRFDDetector
from pipeline.tracking.face_tracking import IoUFaceTracker
from pipeline.utils.logging import get_logger

logger = get_logger("run_scrfd")

def main():
    parser = argparse.ArgumentParser(description="Run SCRFD face detection on videos.")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./results/scrfd")
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--conf-thresh", type=float, default=0.50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    loader = FFppDatasetLoader(args.data_dir)
    detector = SCRFDDetector(conf_threshold=args.conf_thresh)
    tracker = IoUFaceTracker()

    logger.info(f"Processing {len(loader)} videos for face detection...")
    
    total_frames = 0
    total_faces = 0
    start_time = time.time()

    for idx, meta in enumerate(loader.videos):
        cache_file = os.path.join(args.output_dir, f"{meta.video_id}_detections.json")
        if os.path.exists(cache_file):
            continue

        try:
            reader = VideoReader(meta.video_path)
            frames = reader.sample_uniform(max_frames=args.max_frames)
            video_dets = []

            for frame in frames:
                dets = detector.detect_frame(frame.image_bgr, frame.frame_index, frame.timestamp_sec)
                dets = tracker.update(dets, frame.frame_index)
                for d in dets:
                    video_dets.append({
                        "frame_index": d.frame_index,
                        "timestamp_sec": d.timestamp_sec,
                        "bbox": d.bbox.tolist(),
                        "confidence": float(d.confidence),
                        "track_id": d.track_id,
                        "landmarks_5": d.landmarks_5.tolist() if d.landmarks_5 is not None else None
                    })
                total_faces += len(dets)
            total_frames += len(frames)

            with open(cache_file, "w") as f:
                json.dump({
                    "video_id": meta.video_id,
                    "category": meta.category,
                    "compression": meta.compression,
                    "detections": video_dets
                }, f, indent=2)

        except Exception as e:
            logger.error(f"Failed processing {meta.video_id}: {e}")

    elapsed = time.time() - start_time
    fps = total_frames / elapsed if elapsed > 0 else 0.0
    logger.info(f"Completed SCRFD processing. Extracted {total_faces} faces across {total_frames} frames ({fps:.1f} FPS).")

if __name__ == "__main__":
    main()
