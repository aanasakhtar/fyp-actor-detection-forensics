"""
Comprehensive Forensic Benchmark & Latency Profiler for Talent Marketplace Monitoring.

Evaluates forensic detectors (ViT, SBI EfficientNet-B4, GenD) across large-coverage datasets (Celeb-DF & FF++).
Measures:
  1. Diagnostic Detection Metrics: ROC-AUC, PR-AUC, Accuracy, Precision, Recall, F1, FPR on Real.
  2. Latency Benchmarking: Detection (SCRFD), Biometric Embedding (ArcFace), and Forensic Inference (ms/frame, FPS).
  3. Executive Visual Comparison: 4-panel diagnostic card and JSON summary.
"""

import os
import sys
import time
import argparse
import json
from typing import Dict, List, Any, Optional
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Ensure local 'src' directory takes priority in sys.path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from pipeline.data.dataset_loader import FFppDatasetLoader
from pipeline.data.video_reader import VideoReader
from pipeline.detection.scrfd_detector import SCRFDDetector
from pipeline.preprocessing.face_alignment import align_face_5point, align_face_gend
from pipeline.preprocessing.face_cropping import crop_context_face
from pipeline.recognition.arcface_encoder import ArcFaceEncoder
from pipeline.forensics.vit_detector import ViTDeepfakeDetector
from pipeline.forensics.sbi_detector import SBIEfficientNetDetector
from pipeline.forensics.clip_detector import GenDCLIPDetector
from pipeline.forensics.pe_detector import GenDPEDetector
from pipeline.forensics.dinov3_detector import GenDDINOv3Detector
from pipeline.evaluation.forensic_metrics import compute_forensic_metrics
from pipeline.utils.logging import get_logger

logger = get_logger("benchmark_forensics")


def init_forensic_models(model_names: List[str]) -> Dict[str, Any]:
    """Instantiate requested forensic detection models."""
    models = {}
    for name in model_names:
        key = name.lower().strip()
        try:
            if key in ["vit", "vit-base"]:
                logger.info("Initializing ViT Deepfake Detector ('dima806/deepfake_vs_real_image_detection')...")
                models["ViT"] = ViTDeepfakeDetector()
            elif key in ["efficientnet-b4", "sbi", "efficientnet"]:
                logger.info("Initializing SBI EfficientNet-B4 ('nikokons/contrastive-deepfake-detector')...")
                models["EfficientNet-B4 (SBI)"] = SBIEfficientNetDetector()
            elif key in ["gend-pe", "pe"]:
                logger.info("Initializing GenD Perception Encoder ('yermandy/GenD_PE_L')...")
                models["GenD (PE-L)"] = GenDPEDetector()
            elif key in ["gend-clip", "clip"]:
                logger.info("Initializing GenD CLIP ('yermandy/GenD_CLIP_L_14')...")
                models["GenD (CLIP-L)"] = GenDCLIPDetector()
            elif key in ["gend-dinov3", "dinov3"]:
                logger.info("Initializing GenD DINOv3 ('yermandy/GenD_DINOv3_L')...")
                models["GenD (DINOv3-L)"] = GenDDINOv3Detector()
            else:
                logger.warning(f"Unknown forensic model '{name}'. Skipping.")
        except Exception as e:
            logger.error(f"Failed to load model '{name}': {e}")

    # Keep only successfully loaded models
    active_models = {k: v for k, v in models.items() if getattr(v, "model", None) is not None}
    return active_models


def find_optimal_threshold(y_true: List[int], scores: List[float]) -> Dict[str, Any]:
    """Find optimal decision threshold maximizing Youden's J statistic (TPR - FPR)."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    opt_thresh = float(thresholds[best_idx])
    
    # Clip to valid probability bounds [0.10, 0.92]
    opt_thresh = max(0.10, min(0.92, opt_thresh))
    
    pred = (np.array(scores) >= opt_thresh).astype(int)
    y_arr = np.array(y_true)
    
    acc = float(np.mean(pred == y_arr))
    tp = int(np.sum((pred == 1) & (y_arr == 1)))
    fp = int(np.sum((pred == 1) & (y_arr == 0)))
    tn = int(np.sum((pred == 0) & (y_arr == 0)))
    fn = int(np.sum((pred == 0) & (y_arr == 1)))
    
    rec = float(tp / max(1, (tp + fn)))
    prec = float(tp / max(1, (tp + fp)))
    fpr_real = float(fp / max(1, (fp + tn)))
    f1 = float(2 * prec * rec / max(1e-6, (prec + rec)))
    
    return {
        "optimal_threshold": round(opt_thresh, 3),
        "calibrated_accuracy": round(acc, 4),
        "calibrated_recall": round(rec, 4),
        "calibrated_precision": round(prec, 4),
        "calibrated_fpr_on_real": round(fpr_real, 4),
        "calibrated_f1": round(f1, 4)
    }


def plot_benchmark_dashboard(
    metrics_by_model: Dict[str, Dict[str, Any]],
    timing_stats: Dict[str, Dict[str, float]],
    labels_true: List[int],
    scores_by_model: Dict[str, List[float]],
    save_path: str,
    calibrated_by_model: Optional[Dict[str, Dict[str, Any]]] = None
):
    """Render 4-panel executive diagnostic figure."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
    models = list(metrics_by_model.keys())
    colors = ["#1976D2", "#388E3C", "#D32F2F", "#7B1FA2", "#F57C00"][:len(models)]

    # 1. Panel 1: Performance Metrics Comparison (Bar Chart)
    ax1 = axes[0, 0]
    metric_keys = ["accuracy", "roc_auc", "f1", "recall"]
    metric_labels = ["Accuracy", "ROC-AUC", "F1 Score", "Recall (TPR)"]
    x = np.arange(len(metric_labels))
    width = 0.8 / max(1, len(models))

    for i, (m_name, m_data) in enumerate(metrics_by_model.items()):
        vals = [m_data.get(k, 0.0) for k in metric_keys]
        offset = (i - len(models) / 2 + 0.5) * width
        rects = ax1.bar(x + offset, vals, width, label=m_name, color=colors[i], alpha=0.88)
        for r in rects:
            h = r.get_height()
            ax1.annotate(f"{h:.2f}", xy=(r.get_x() + r.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels, fontweight="bold")
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel("Score (0.0 - 1.0)", fontweight="bold")
    ax1.set_title("FORENSIC ACCURACY & DETECTION POWER", fontsize=12, fontweight="bold", pad=10)
    ax1.legend(loc="lower right", framealpha=0.9)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # 2. Panel 2: ROC Curves
    ax2 = axes[0, 1]
    y_true = np.array(labels_true)
    for i, (m_name, scores) in enumerate(scores_by_model.items()):
        if len(set(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, scores)
            roc_auc = metrics_by_model[m_name].get("roc_auc", 0.0)
            ax2.plot(fpr, tpr, color=colors[i], lw=2.2, label=f"{m_name} (AUC = {roc_auc:.4f})")
    ax2.plot([0, 1], [0, 1], color="grey", lw=1.2, linestyle="--")
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("False Positive Rate (FPR)", fontweight="bold")
    ax2.set_ylabel("True Positive Rate (TPR / Recall)", fontweight="bold")
    ax2.set_title("RECEIVER OPERATING CHARACTERISTIC (ROC)", fontsize=12, fontweight="bold", pad=10)
    ax2.legend(loc="lower right", framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # 3. Panel 3: Score Distribution (Real vs Fake)
    ax3 = axes[1, 0]
    box_positions = []
    box_data = []
    box_colors = []
    tick_labels = []

    pos = 1
    for i, (m_name, scores) in enumerate(scores_by_model.items()):
        s_arr = np.array(scores)
        real_s = s_arr[y_true == 0]
        fake_s = s_arr[y_true == 1]
        
        box_data.extend([real_s, fake_s])
        box_positions.extend([pos, pos + 0.8])
        box_colors.extend(["#4CAF50", "#E53935"])  # Green for Real, Red for Fake
        tick_labels.append(f"{m_name}\n(Real / Fake)")
        pos += 2.2

    if box_data:
        bp = ax3.boxplot(box_data, positions=box_positions, widths=0.6, patch_artist=True)
        for patch, col in zip(bp["boxes"], box_colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.75)
        ax3.set_xticks([1.4 + i * 2.2 for i in range(len(models))])
        ax3.set_xticklabels([m for m in models], fontweight="bold")
        ax3.axhline(0.50, color="black", linestyle=":", lw=1.5, label="Default Threshold (0.50)")
        if calibrated_by_model:
            # Highlight optimal threshold of highest-AUC model (typically GenD PE-L)
            best_m = max(metrics_by_model.keys(), key=lambda k: metrics_by_model[k].get("roc_auc", 0.0))
            best_t = calibrated_by_model.get(best_m, {}).get("optimal_threshold")
            if best_t:
                ax3.axhline(best_t, color="#1976D2", linestyle="--", lw=2.0, label=f"Calibrated T* ({best_m}: {best_t:.2f})")
        ax3.set_ylabel("Manipulation Score p(Fake)", fontweight="bold")
        ax3.set_title("PREDICTION DISTRIBUTIONS: REAL (Green) vs FAKE (Red)", fontsize=12, fontweight="bold", pad=10)
        ax3.grid(axis="y", linestyle="--", alpha=0.5)
        ax3.legend(loc="upper left")

    # 4. Panel 4: Latency & Throughput Benchmark (ms/frame)
    ax4 = axes[1, 1]
    det_time = timing_stats.get("avg_detection_ms", 0.0)
    arc_time = timing_stats.get("avg_embedding_ms", 0.0)
    
    inf_times = [timing_stats.get("models", {}).get(m, {}).get("avg_inference_ms", 0.0) for m in models]
    y_pos = np.arange(len(models))
    
    # Stacked horizontal bar: Stage 1 + Stage 2 + Stage 3
    ax4.barh(y_pos, [det_time] * len(models), height=0.45, label=f"Stage 1: SCRFD Detection ({det_time:.1f} ms)", color="#607D8B", alpha=0.85)
    ax4.barh(y_pos, [arc_time] * len(models), left=[det_time] * len(models), height=0.45, label=f"Stage 2: ArcFace Biometrics ({arc_time:.1f} ms)", color="#009688", alpha=0.85)
    ax4.barh(y_pos, inf_times, left=[det_time + arc_time] * len(models), height=0.45, label="Stage 3: Deepfake Forensics", color="#FF9800", alpha=0.85)

    max_tot = 0.0
    for i, m in enumerate(models):
        tot = det_time + arc_time + inf_times[i]
        if tot > max_tot:
            max_tot = tot
        fps = 1000.0 / tot if tot > 0 else 0.0
        ax4.text(tot + 5, i, f"Total: {tot:.1f} ms  ({fps:.1f} FPS)\n[S3: {inf_times[i]:.1f} ms]", va="center", fontweight="bold", fontsize=8.5)

    ax4.set_xlim(0, max(100.0, max_tot * 1.35))
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(models, fontweight="bold")
    ax4.set_xlabel("Latency per Frame (Milliseconds - Lower is Faster)", fontweight="bold")
    ax4.set_title("END-TO-END PIPELINE LATENCY BREAKDOWN (Stages 1, 2, 3)", fontsize=12, fontweight="bold", pad=10)
    ax4.legend(loc="lower right", framealpha=0.92)
    ax4.grid(axis="x", linestyle="--", alpha=0.5)

    plt.suptitle("TALENT MARKETPLACE: COMPREHENSIVE AI FORENSICS & LATENCY BENCHMARK", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved Executive Benchmark Report Card to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate forensic detectors across multiple videos and frames.")
    parser.add_argument("--data-dir", type=str, default="/kaggle/working/data_celebdf", help="Dataset directory (Celeb-DF or FF++).")
    parser.add_argument("--models", nargs="+", default=["vit", "efficientnet-b4"], help="Forensic models to benchmark ('vit', 'efficientnet-b4', 'gend-clip', 'gend-pe', 'gend-dinov3').")
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/results/marketplace_benchmark", help="Output directory.")
    parser.add_argument("--max-videos", type=int, default=None, help="Maximum number of videos to evaluate (balanced real/fake).")
    parser.add_argument("--max-frames", type=int, default=12, help="Uniform sampled frames per video.")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face User Access Token (optional if HF_TOKEN env is set).")
    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*78)
    print("      TALENT MARKETPLACE: MULTI-MODEL FORENSIC EVALUATION & LATENCY BENCHMARK")
    print("="*78)

    # 1. Initialize Dataset
    target_data_dir = args.data_dir
    if not os.path.exists(target_data_dir):
        logger.warning(f"Data root directory does not exist: '{target_data_dir}'. Searching for mounted datasets...")
        import glob
        found = (
            glob.glob("/kaggle/input/**/Celeb-synthesis", recursive=True) +
            glob.glob("/kaggle/input/**/celeb-synthesis", recursive=True) +
            glob.glob("/kaggle/input/**/Celeb-real", recursive=True) +
            glob.glob("/kaggle/input/**/original_sequences", recursive=True)
        )
        if found:
            target_data_dir = os.path.dirname(found[0])
            logger.info(f"Auto-discovered dataset root at: '{target_data_dir}'")
        elif os.path.exists("/kaggle/input"):
            subdirs = os.listdir("/kaggle/input")
            logger.info(f"Available directories in /kaggle/input: {subdirs}")
            for sd in subdirs:
                cand = os.path.join("/kaggle/input", sd)
                if os.path.isdir(cand) and "celeb" in sd.lower():
                    target_data_dir = cand
                    logger.info(f"Attempting candidate Celeb directory: '{target_data_dir}'")
                    break

    loader = FFppDatasetLoader(target_data_dir)
    if len(loader) == 0:
        logger.error(f"No valid MP4 videos found in '{target_data_dir}'.")
        if os.path.exists("/kaggle/input"):
            logger.error(f"Currently mounted in /kaggle/input: {os.listdir('/kaggle/input')}")
        logger.error("Please check if the dataset has finished mounting in Kaggle's 'Input' pane.")
        return

    # Select balanced, identity-diverse sample if max_videos is specified
    eval_videos = loader.videos
    if args.max_videos and len(eval_videos) > args.max_videos:
        reals = [v for v in loader.videos if not v.is_fake]
        fakes = [v for v in loader.videos if v.is_fake]
        n_half = args.max_videos // 2
        rng = np.random.RandomState(42)

        # Diverse real sampling across distinct actor IDs
        reals_by_id: Dict[str, List[Any]] = {}
        for r in reals:
            reals_by_id.setdefault(r.target_id, []).append(r)
        distinct_real_ids = sorted(list(reals_by_id.keys()))
        rng.shuffle(distinct_real_ids)
        
        sample_reals = []
        idx = 0
        while len(sample_reals) < min(len(reals), n_half) and distinct_real_ids:
            cur_id = distinct_real_ids[idx % len(distinct_real_ids)]
            if reals_by_id[cur_id]:
                sample_reals.append(reals_by_id[cur_id].pop(0))
            idx += 1
            if all(len(v) == 0 for v in reals_by_id.values()):
                break

        # Diverse fake sampling across distinct actor combinations
        fakes_by_id: Dict[str, List[Any]] = {}
        for f in fakes:
            fakes_by_id.setdefault(f.target_id, []).append(f)
        distinct_fake_ids = sorted(list(fakes_by_id.keys()))
        rng.shuffle(distinct_fake_ids)
        
        sample_fakes = []
        idx = 0
        n_fakes_target = args.max_videos - len(sample_reals)
        while len(sample_fakes) < min(len(fakes), n_fakes_target) and distinct_fake_ids:
            cur_id = distinct_fake_ids[idx % len(distinct_fake_ids)]
            if fakes_by_id[cur_id]:
                sample_fakes.append(fakes_by_id[cur_id].pop(0))
            idx += 1
            if all(len(v) == 0 for v in fakes_by_id.values()):
                break

        eval_videos = sample_reals + sample_fakes

    n_real = sum(1 for v in eval_videos if not v.is_fake)
    n_fake = sum(1 for v in eval_videos if v.is_fake)
    logger.info(f"Target Evaluation Set: {len(eval_videos)} videos ({n_real} Real, {n_fake} Deepfake)")

    # 2. Initialize Engines
    detector = SCRFDDetector(conf_threshold=0.50)
    encoder = ArcFaceEncoder()
    forensic_models = init_forensic_models(args.models)

    if not forensic_models:
        logger.error("No forensic models loaded. Exiting.")
        return

    # Tracking storage
    scores_by_model: Dict[str, List[float]] = {m: [] for m in forensic_models}
    labels_true: List[int] = []
    
    # Timing accumulation (seconds)
    total_det_time = 0.0
    total_arc_time = 0.0
    total_model_times = {m: 0.0 for m in forensic_models}
    total_frames_processed = 0
    total_faces_processed = 0
    video_records = []

    print(f"\nProcessing {len(eval_videos)} videos with {len(forensic_models)} forensic models...")

    for v_idx, meta in enumerate(eval_videos):
        t_vid_start = time.time()
        reader = VideoReader(meta.video_path)
        frames = reader.sample_uniform(max_frames=args.max_frames)
        
        if not frames:
            continue

        vid_crops_by_model = {m: [] for m in forensic_models}
        vid_embs = []

        # Stage 1: Face Detection & Stage 2: Embedding
        for f in frames:
            t0 = time.time()
            dets = detector.detect_frame(f.image_bgr, f.frame_index, f.timestamp_sec)
            total_det_time += (time.time() - t0)
            total_frames_processed += 1

            if dets:
                primary = max(dets, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
                total_faces_processed += 1
                
                # ArcFace embedding
                if primary.landmarks_5 is not None:
                    aligned = align_face_5point(f.image_bgr, primary.landmarks_5)
                    if aligned is not None:
                        t0 = time.time()
                        emb = encoder.extract_embedding(aligned)
                        total_arc_time += (time.time() - t0)
                        if emb is not None:
                            vid_embs.append(emb)

                # Prepare model-tailored crops
                for m_name in forensic_models:
                    if "vit" in m_name.lower():
                        c = crop_context_face(f.image_bgr, primary.bbox, expansion_factor=1.20, target_size=(224, 224))
                    elif "efficientnet" in m_name.lower() or "sbi" in m_name.lower():
                        c = crop_context_face(f.image_bgr, primary.bbox, expansion_factor=1.15, target_size=(380, 380))
                    else:  # GenD models
                        c = align_face_gend(f.image_bgr, primary.landmarks_5, target_size=(224, 224), scale=1.30)
                        if c is None:
                            c = crop_context_face(f.image_bgr, primary.bbox, target_size=(224, 224))
                    if c is not None:
                        vid_crops_by_model[m_name].append(c)

        # Stage 3: Inference across forensic models
        vid_scores = {}
        for m_name, model in forensic_models.items():
            crops = vid_crops_by_model[m_name]
            if crops:
                t0 = time.time()
                f_scores = model.predict_crops(crops)
                total_model_times[m_name] += (time.time() - t0)
                agg_score = model.aggregate_scores(f_scores)
            else:
                agg_score = 0.0
            vid_scores[m_name] = agg_score
            scores_by_model[m_name].append(agg_score)

        is_fake_int = 1 if meta.is_fake else 0
        labels_true.append(is_fake_int)

        t_vid_elapsed = time.time() - t_vid_start
        video_records.append({
            "video_id": meta.video_id,
            "category": meta.category,
            "is_fake": meta.is_fake,
            "scores": vid_scores,
            "latency_sec": round(t_vid_elapsed, 3)
        })

        log_step = 10 if len(eval_videos) > 60 else 5
        if (v_idx + 1) % log_step == 0 or (v_idx + 1) == len(eval_videos):
            print(f"  [{v_idx+1:03d}/{len(eval_videos):03d}] {meta.category:16s} | "
                  f"True: {'FAKE' if meta.is_fake else 'REAL':4s} | " + 
                  " | ".join([f"{m[:12]}: {vid_scores.get(m, 0.0):.4f}" for m in forensic_models]))

    # 3. Compute Metrics (Uncalibrated @ 0.50 and Calibrated @ Optimal T*)
    metrics_by_model = {}
    calibrated_by_model = {}
    for m_name in forensic_models:
        m_dict = compute_forensic_metrics(labels_true, scores_by_model[m_name], threshold=0.50)
        # Compute False Positive Rate (FPR) on Real videos
        cm = m_dict.get("confusion_matrix", [[0, 0], [0, 0]])
        tn, fp = cm[0][0], cm[0][1]
        fpr = fp / max(1, (fp + tn))
        m_dict["fpr_on_real"] = round(fpr, 4)
        metrics_by_model[m_name] = m_dict
        calibrated_by_model[m_name] = find_optimal_threshold(labels_true, scores_by_model[m_name])

    # 4. Latency Statistics
    avg_det_ms = (total_det_time / max(1, total_frames_processed)) * 1000.0
    avg_arc_ms = (total_arc_time / max(1, total_faces_processed)) * 1000.0
    timing_stats = {
        "avg_detection_ms": round(avg_det_ms, 2),
        "avg_embedding_ms": round(avg_arc_ms, 2),
        "total_frames": total_frames_processed,
        "total_faces": total_faces_processed,
        "models": {}
    }

    for m_name in forensic_models:
        avg_inf_ms = (total_model_times[m_name] / max(1, total_faces_processed)) * 1000.0
        tot_ms = avg_det_ms + avg_arc_ms + avg_inf_ms
        fps = 1000.0 / tot_ms if tot_ms > 0 else 0.0
        timing_stats["models"][m_name] = {
            "stage_1_detection_ms": round(avg_det_ms, 2),
            "stage_2_biometrics_ms": round(avg_arc_ms, 2),
            "stage_3_forensic_ms": round(avg_inf_ms, 2),
            "avg_inference_ms": round(avg_inf_ms, 2),
            "total_frame_ms": round(tot_ms, 2),
            "effective_fps": round(fps, 1)
        }

    # 5. Print Detailed Markdown Tables
    print("\n" + "="*112)
    print("                     TABLE 1: STAGE-BY-STAGE PIPELINE LATENCY & THROUGHPUT")
    print("="*112)
    print(f"{'Forensic Detector':24s} | {'Stage 1: SCRFD':14s} | {'Stage 2: ArcFace':16s} | {'Stage 3: Forensic':17s} | {'Total Frame':12s} | {'Throughput':10s}")
    print("-" * 112)
    for m_name in forensic_models:
        t_data = timing_stats["models"].get(m_name, {})
        s1 = f"{t_data.get('stage_1_detection_ms', 0):.1f} ms"
        s2 = f"{t_data.get('stage_2_biometrics_ms', 0):.1f} ms"
        s3 = f"{t_data.get('stage_3_forensic_ms', 0):.1f} ms"
        tot = f"{t_data.get('total_frame_ms', 0):.1f} ms"
        fps = f"{t_data.get('effective_fps', 0):.1f} FPS"
        print(f"{m_name:24s} | {s1:>14s} | {s2:>16s} | {s3:>17s} | {tot:>12s} | {fps:>10s}")
    print("="*112)

    print("\n" + "="*104)
    print("                     TABLE 2: DEFAULT FORENSIC PERFORMANCE (THRESHOLD = 0.50)")
    print("="*104)
    print(f"{'Forensic Detector':24s} | {'ROC-AUC':7s} | {'Accuracy':8s} | {'Recall':6s} | {'Precision':9s} | {'FPR (Real)':10s} | {'F1 Score':8s}")
    print("-" * 104)
    for m_name, m_data in metrics_by_model.items():
        print(f"{m_name:24s} | {m_data['roc_auc']:7.4f} | {m_data['accuracy']*100:6.1f} % | {m_data['recall']*100:5.1f}% | {m_data['precision']*100:7.1f}% | {m_data['fpr_on_real']*100:7.1f} % | {m_data.get('f1', 0.0):7.4f}")
    print("="*104)

    print("\n" + "="*124)
    print("                     TABLE 3: CALIBRATED FORENSIC PERFORMANCE (OPTIMAL THRESHOLD T*)")
    print("="*124)
    print(f"{'Forensic Detector':24s} | {'Optimal T*':11s} | {'Calibrated Acc':15s} | {'Recall':8s} | {'Precision':10s} | {'FPR (Real)':11s} | {'F1 Score':8s}")
    print("-" * 124)
    for m_name, c_data in calibrated_by_model.items():
        print(f"{m_name:24s} | {c_data['optimal_threshold']:11.3f} | {c_data['calibrated_accuracy']*100:13.1f} % | {c_data['calibrated_recall']*100:6.1f} % | {c_data['calibrated_precision']*100:8.1f} % | {c_data['calibrated_fpr_on_real']*100:9.1f} % | {c_data['calibrated_f1']:8.4f}")
    print("="*124 + "\n")

    # 6. Generate 4-Panel Executive Diagnostic Report Card
    report_img_path = os.path.join(args.output_dir, "marketplace_forensic_benchmark.png")
    plot_benchmark_dashboard(
        metrics_by_model=metrics_by_model,
        timing_stats=timing_stats,
        labels_true=labels_true,
        scores_by_model=scores_by_model,
        save_path=report_img_path,
        calibrated_by_model=calibrated_by_model
    )

    # 7. Save JSON Export
    json_path = os.path.join(args.output_dir, "marketplace_benchmark_results.json")
    with open(json_path, "w") as jf:
        json.dump({
            "dataset_dir": args.data_dir,
            "videos_evaluated": len(eval_videos),
            "real_videos": n_real,
            "fake_videos": n_fake,
            "metrics_at_0_50": metrics_by_model,
            "calibrated_metrics": calibrated_by_model,
            "timing_benchmarks": timing_stats,
            "per_video_records": video_records
        }, jf, indent=2)
    print(f"[+] Saved Machine-Readable Benchmark Results to: {json_path}")
    print(f"[+] Saved Executive Benchmark Report Card to: {report_img_path}\n")


if __name__ == "__main__":
    main()
