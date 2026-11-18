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
    parser.add_argument("--max-videos", type=int, default=None, help="Maximum videos to evaluate (balanced real/fake subsampling).")
    args = parser.parse_args()

    actual_data_dir = resolve_data_dir(args.data_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    loader = FFppDatasetLoader(actual_data_dir)
    detector = SCRFDDetector(conf_threshold=0.50)

    # Subsample if max_videos requested
    eval_videos = loader.videos
    if args.max_videos and len(eval_videos) > args.max_videos:
        reals = [v for v in loader.videos if not v.is_fake]
        fakes = [v for v in loader.videos if v.is_fake]
        n_half = args.max_videos // 2
        rng = np.random.RandomState(42)
        sample_reals = list(rng.choice(reals, min(len(reals), n_half), replace=False)) if reals else []
        sample_fakes = list(rng.choice(fakes, min(len(fakes), args.max_videos - len(sample_reals)), replace=False)) if fakes else []
        eval_videos = sample_reals + sample_fakes
        logger.info(f"Subsampled {len(eval_videos)} videos ({len(sample_reals)} Real, {len(sample_fakes)} Fake) for pilot evaluation.")

    # Consolidate backbone choices
    selected_backbones = []
    if args.backbones:
        selected_backbones = [b.lower() for b in args.backbones]
    elif args.backbone == "all":
        selected_backbones = ["clip", "pe", "dinov3"]
    else:
        selected_backbones = [args.backbone.lower()]

    backbones = {}
    for b_key in selected_backbones:
        try:
            if b_key == "clip":
                backbones["clip_vit_l14"] = GenDCLIPDetector()
            elif b_key == "pe":
                backbones["pe_large"] = GenDPEDetector()
            elif b_key == "dinov3":
                backbones["dinov3_vit_l16"] = GenDDINOv3Detector()
        except Exception as e_init:
            logger.error(f"Failed to initialize backbone '{b_key}': {e_init}. Skipping.")

    logger.info(f"Loaded dataset: {len(loader)} total videos from '{actual_data_dir}'")
    all_metrics = {}

    for name, model in backbones.items():
        if getattr(model, "model", None) is None:
            logger.error(f"Backbone '{name}' weights could not be loaded. Skipping this backbone to prevent invalid benchmark metrics.")
            continue

        logger.info(f"Evaluating GenD backbone: {name} across {len(eval_videos)} videos...")
        y_true = []
        y_scores = []
        video_results = []

        for meta in eval_videos:
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
        all_metrics[name] = metrics
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

    if len(all_metrics) > 1:
        print("\n" + "="*84)
        print("         STAGE 3: GEND MULTI-BACKBONE FORENSIC BENCHMARK SUMMARY")
        print("="*84)
        print(f"{'Backbone':<20} {'ROC-AUC':<10} {'PR-AUC':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<10} {'F1':<8}")
        print("-" * 84)
        for b_name, m in all_metrics.items():
            print(f"{b_name:<20} {m['roc_auc']:<10.4f} {m['pr_auc']:<10.4f} {m['accuracy']*100:<11.2f}% {m['precision']*100:<11.2f}% {m['recall']*100:<9.2f}% {m['f1']:<8.4f}")
        print("="*84 + "\n")
        summary_file = os.path.join(args.output_dir, "multi_backbone_summary.json")
        with open(summary_file, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"[+] Saved multi-backbone comparison to: {summary_file}")

if __name__ == "__main__":
    main()
