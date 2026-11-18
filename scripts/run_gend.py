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
import numpy as np
from pipeline.data.dataset_loader import FFppDatasetLoader
from pipeline.data.video_reader import VideoReader
from pipeline.detection.scrfd_detector import SCRFDDetector
from pipeline.preprocessing.face_cropping import crop_context_face
from pipeline.forensics.clip_detector import GenDCLIPDetector
from pipeline.forensics.pe_detector import GenDPEDetector
from pipeline.forensics.dinov3_detector import GenDDINOv3Detector
from pipeline.evaluation.forensic_metrics import compute_forensic_metrics
from pipeline.visualization.evaluation_plots import plot_gend_evaluation
from pipeline.utils.logging import get_logger

logger = get_logger("run_gend")

def resolve_data_dir(requested_path: str) -> str:
    """Resolve data directory with fallbacks for Kaggle and local paths."""
    if os.path.exists(requested_path) and os.listdir(requested_path):
        return requested_path
    fallbacks = [
        "/kaggle/working/data",
        "/kaggle/input/faceforensics-pilot",
        "/kaggle/working/data_celebdf",
        "./data"
    ]
    for fb in fallbacks:
        if os.path.exists(fb) and os.listdir(fb):
            logger.warning(f"Requested path '{requested_path}' unavailable. Falling back to: {fb}")
            return fb
    return requested_path

def main():
    parser = argparse.ArgumentParser(description="Evaluate GenD deepfake detection backbones.")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--backbone", type=str, default="clip", choices=["clip", "pe", "dinov3", "all"])
    parser.add_argument("--backbones", nargs="+", default=None, help="List of backbones to evaluate.")
    parser.add_argument("--scrfd-dir", type=str, default=None, help="Directory containing SCRFD detections.")
    parser.add_argument("--output-dir", type=str, default="./results/gend")
    parser.add_argument("--max-frames", type=int, default=16, help="Frames sampled per video.")
    args = parser.parse_args()

    actual_data_dir = resolve_data_dir(args.data_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    loader = FFppDatasetLoader(actual_data_dir)
    detector = SCRFDDetector(conf_threshold=0.50)

    # Consolidate backbone choices
    selected_backbones = []
    if args.backbones:
        selected_backbones = [b.lower() for b in args.backbones]
    elif args.backbone == "all":
        selected_backbones = ["clip", "pe", "dinov3"]
    else:
        selected_backbones = [args.backbone.lower()]

    backbones = {}
    if "clip" in selected_backbones:
        backbones["clip_vit_l14"] = GenDCLIPDetector()
    if "pe" in selected_backbones:
        backbones["pe_large"] = GenDPEDetector()
    if "dinov3" in selected_backbones:
        backbones["dinov3_vit_l16"] = GenDDINOv3Detector()

    logger.info(f"Loaded dataset: {len(loader)} total videos from '{actual_data_dir}'")

    for name, model in backbones.items():
        logger.info(f"Evaluating GenD backbone: {name} across {len(loader)} videos...")
        y_true = []
        y_scores = []
        video_results = []

        for meta in loader.videos:
            reader = VideoReader(meta.video_path)
            frames = reader.sample_uniform(max_frames=args.max_frames)
            crops = []
            for f in frames:
                dets = detector.detect_frame(f.image_bgr, f.frame_index, f.timestamp_sec)
                if dets:
                    # Select largest foreground face (avoids background studio monitors)
                    best_det = max(dets, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
                    c = crop_context_face(f.image_bgr, best_det.bbox)
                    if c is not None:
                        crops.append(c)

            if crops:
                frame_scores = model.predict_crops(crops)
                vid_score = model.aggregate_scores(frame_scores)
                label = 1 if meta.is_fake else 0
                y_true.append(label)
                y_scores.append(vid_score)
                video_results.append({
                    "video_id": meta.video_id,
                    "category": meta.category,
                    "is_fake": meta.is_fake,
                    "score": round(vid_score, 4),
                    "prediction": "Fake" if vid_score >= 0.50 else "Real"
                })

        metrics = compute_forensic_metrics(y_true, y_scores)
        cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
        tn, fp = cm[0][0], cm[0][1]
        fn, tp = cm[1][0], cm[1][1]

        num_real = int(np.sum(np.array(y_true) == 0))
        num_fake = int(np.sum(np.array(y_true) == 1))

        print("\n" + "="*70)
        print(f"      STAGE 3: GEND DEEPFAKE DETECTION EVALUATION ({name.upper()})")
        print("="*70)
        print(f"Total Videos Evaluated:       {len(y_true)}")
        print(f"  * Real / Pristine Videos:   {num_real}")
        print(f"  * Fake / Manipulated Videos:{num_fake}")
        print("-" * 70)
        print(f"ROC-AUC:                      {metrics['roc_auc']:.4f}")
        print(f"PR-AUC (Average Precision):   {metrics['pr_auc']:.4f}")
        print(f"Accuracy:                     {metrics['accuracy'] * 100:.2f}%")
        print(f"Precision:                    {metrics['precision'] * 100:.2f}%")
        print(f"Recall (Sensitivity):         {metrics['recall'] * 100:.2f}%")
        print(f"F1-Score:                     {metrics['f1']:.4f}")
        print(f"Confusion Matrix:             TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
        print("="*70 + "\n")

        # Save metrics JSON
        metrics_file = os.path.join(args.output_dir, f"{name}_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[+] Saved metrics to: {metrics_file}")

        # Save per-video predictions
        preds_file = os.path.join(args.output_dir, f"{name}_predictions.json")
        with open(preds_file, "w") as f:
            json.dump(video_results, f, indent=2)
        print(f"[+] Saved predictions to: {preds_file}")

        # Save supervisor evaluation plot
        plot_file = os.path.join(args.output_dir, f"{name}_evaluation.png")
        try:
            plot_gend_evaluation(
                y_true=y_true,
                y_scores=y_scores,
                title=f"Stage 3: GenD Manipulation Detection ({name})",
                threshold=0.50,
                save_path=plot_file
            )
            print(f"[+] Saved evaluation plot to: {plot_file}")
        except Exception as e:
            logger.warning(f"Could not generate evaluation plot: {e}")

if __name__ == "__main__":
    main()
