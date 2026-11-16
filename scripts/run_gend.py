"""
Standalone script to run GenD deepfake manipulation detection across candidate backbones.
"""

import os
import sys

# Ensure local 'src' directory takes priority in sys.path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import argparse
import json
from pipeline.data.dataset_loader import FFppDatasetLoader
from pipeline.data.video_reader import VideoReader
from pipeline.detection.scrfd_detector import SCRFDDetector
from pipeline.preprocessing.face_cropping import crop_context_face
from pipeline.forensics.clip_detector import GenDCLIPDetector
from pipeline.forensics.pe_detector import GenDPEDetector
from pipeline.forensics.dinov3_detector import GenDDINOv3Detector
from pipeline.evaluation.forensic_metrics import compute_forensic_metrics
from pipeline.utils.logging import get_logger

logger = get_logger("run_gend")

def main():
    parser = argparse.ArgumentParser(description="Evaluate GenD deepfake detection backbones.")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--backbone", type=str, default="clip", choices=["clip", "pe", "dinov3", "all"])
    parser.add_argument("--backbones", nargs="+", default=None, help="List of backbones to evaluate.")
    parser.add_argument("--scrfd-dir", type=str, default=None, help="Directory containing SCRFD detections.")
    parser.add_argument("--output-dir", type=str, default="./results/gend")
    args = parser.parse_args()

    # Consolidate backbone choices
    selected_backbones = []
    if args.backbones:
        selected_backbones = [b.lower() for b in args.backbones]
    elif args.backbone == "all":
        selected_backbones = ["clip", "pe", "dinov3"]
    else:
        selected_backbones = [args.backbone.lower()]

    os.makedirs(args.output_dir, exist_ok=True)
    loader = FFppDatasetLoader(args.data_dir)
    detector = SCRFDDetector(conf_threshold=0.50)

    backbones = {}
    if "clip" in selected_backbones:
        backbones["clip_vit_l14"] = GenDCLIPDetector()
    if "pe" in selected_backbones:
        backbones["pe_large"] = GenDPEDetector()
    if "dinov3" in selected_backbones:
        backbones["dinov3_vit_l16"] = GenDDINOv3Detector()

    for name, model in backbones.items():
        logger.info(f"Evaluating GenD backbone: {name}")
        y_true = []
        y_scores = []

        for meta in loader.videos:
            reader = VideoReader(meta.video_path)
            frames = reader.sample_uniform(max_frames=16)
            crops = []
            for f in frames:
                dets = detector.detect_frame(f.image_bgr)
                if dets:
                    c = crop_context_face(f.image_bgr, dets[0].bbox)
                    if c is not None:
                        crops.append(c)

            if crops:
                frame_scores = model.predict_crops(crops)
                vid_score = model.aggregate_scores(frame_scores)
                y_true.append(1 if meta.is_fake else 0)
                y_scores.append(vid_score)

        metrics = compute_forensic_metrics(y_true, y_scores)
        logger.info(f"Results for {name}: {metrics}")
        with open(os.path.join(args.output_dir, f"{name}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
