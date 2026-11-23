"""
Benchmark Script for 'mrxaman19/Indian-deepfake-face' Hugging Face Dataset.

Evaluates forensic detectors on South Asian / Indian real vs. deepfake face images.
Measures:
  1. Diagnostic Detection Metrics: ROC-AUC, PR-AUC, Accuracy, Precision, Recall, F1, FPR on Real.
  2. Latency Benchmarking: Pure neural forward inference latency (ms/face) and throughput (faces/sec).
  3. Visual Diagnostic Report Card and JSON export.
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

logger = get_logger("benchmark_indian_faces")


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


def plot_indian_dashboard(
    metrics_by_model: Dict[str, Dict[str, Any]],
    timing_stats: Dict[str, Any],
    labels_true: List[int],
    scores_by_model: Dict[str, List[float]],
    save_path: str,
    calibrated_by_model: Optional[Dict[str, Dict[str, Any]]] = None
):
    """Render 4-panel executive diagnostic figure including Real vs. Fake boxplots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
    models = list(metrics_by_model.keys())
    colors = ["#1976D2", "#388E3C", "#D32F2F", "#7B1FA2"][:len(models)]

    # 1. Panel 1: Performance Bar Chart
    ax1 = axes[0, 0]
    metric_keys = ["accuracy", "roc_auc", "f1", "recall"]
    metric_labels = ["Accuracy", "ROC-AUC", "F1 Score", "Recall (TPR)"]
    x = np.arange(len(metric_labels))
    w = 0.8 / max(1, len(models))

    for i, (m_name, m_data) in enumerate(metrics_by_model.items()):
        vals = [m_data.get(k, 0.0) for k in metric_keys]
        offset = (i - len(models) / 2 + 0.5) * w
        rects = ax1.bar(x + offset, vals, w, label=m_name, color=colors[i], alpha=0.88)
        for r in rects:
            h = r.get_height()
            ax1.annotate(f"{h:.2f}", xy=(r.get_x() + r.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels, fontweight="bold")
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel("Score (0.0 - 1.0)", fontweight="bold")
    ax1.set_title("SOUTH ASIAN FACIAL DEEPFAKE DETECTION ACCURACY", fontsize=12, fontweight="bold", pad=10)
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

    # 3. Panel 3: Prediction Distributions (Real vs Fake Boxplots)
    ax3 = axes[1, 0]
    box_positions = []
    box_data = []
    box_colors = []
    pos = 1
    for i, (m_name, scores) in enumerate(scores_by_model.items()):
        s_arr = np.array(scores)
        real_s = s_arr[y_true == 0]
        fake_s = s_arr[y_true == 1]
        box_data.extend([real_s, fake_s])
        box_positions.extend([pos, pos + 0.8])
        box_colors.extend(["#4CAF50", "#E53935"])  # Green for Real, Red for Fake
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
            labeled = False
            for i, m_name in enumerate(models):
                c_data = calibrated_by_model.get(m_name)
                if c_data and "optimal_threshold" in c_data:
                    t_val = c_data["optimal_threshold"]
                    cx = 1.4 + i * 2.2
                    lbl = "Calibrated Optimal T*" if not labeled else None
                    labeled = True
                    ax3.hlines(t_val, cx - 0.70, cx + 0.70, colors="#1976D2", linestyles="--", lw=2.2, label=lbl)
                    y_text = t_val - 0.045 if t_val > 0.85 else t_val + 0.025
                    va = "top" if t_val > 0.85 else "bottom"
                    ax3.text(cx, y_text, f"T*={t_val:.2f}", color="#0D47A1", ha="center", va=va, fontsize=8, fontweight="bold",
                             bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="#1976D2", alpha=0.88, lw=0.8))
        ax3.set_ylabel("Manipulation Score p(Fake)", fontweight="bold")
        ax3.set_title("PREDICTION DISTRIBUTIONS: REAL (Green) vs FAKE (Red)", fontsize=12, fontweight="bold", pad=10)
        ax3.grid(axis="y", linestyle="--", alpha=0.5)
        ax3.legend(loc="upper left")

    # 4. Panel 4: Latency & Throughput Benchmark
    ax4 = axes[1, 1]
    det_time = timing_stats.get("avg_detection_ms", 0.0)
    arc_time = timing_stats.get("avg_embedding_ms", 0.0)
    inf_times = [timing_stats.get("models", {}).get(m, {}).get("stage_3_forensic_ms", 0.0) for m in models]
    y_pos = np.arange(len(models))

    ax4.barh(y_pos, [det_time] * len(models), height=0.45, label=f"Stage 1: SCRFD Detection ({det_time:.1f} ms)", color="#607D8B", alpha=0.85)
    ax4.barh(y_pos, [arc_time] * len(models), left=[det_time] * len(models), height=0.45, label=f"Stage 2: ArcFace Biometrics ({arc_time:.1f} ms)", color="#009688", alpha=0.85)
    ax4.barh(y_pos, inf_times, left=[det_time + arc_time] * len(models), height=0.45, label="Stage 3: Deepfake Forensics", color="#FF9800", alpha=0.85)

    max_tot = 0.0
    for i, m in enumerate(models):
        tot = det_time + arc_time + inf_times[i]
        if tot > max_tot:
            max_tot = tot
        fps = 1000.0 / tot if tot > 0 else 0.0
        ax4.text(tot + 5, i, f"Total: {tot:.1f} ms ({fps:.1f} FPS)\n[S3: {inf_times[i]:.1f} ms]", va="center", fontweight="bold", fontsize=8.5)

    ax4.set_xlim(0, max(100.0, max_tot * 1.35))
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(models, fontweight="bold")
    ax4.set_xlabel("Latency per Image (Milliseconds - Lower is Faster)", fontweight="bold")
    ax4.set_title("END-TO-END PIPELINE LATENCY BREAKDOWN (Stages 1, 2, 3)", fontsize=12, fontweight="bold", pad=10)
    ax4.legend(loc="lower right", framealpha=0.92)
    ax4.grid(axis="x", linestyle="--", alpha=0.5)

    plt.suptitle("BENCHMARK: INDIAN & SOUTH ASIAN DEEPFAKE DETECTION (mrxaman19)", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


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

    return {k: v for k, v in models.items() if getattr(v, "model", None) is not None}


def fetch_dataset_files(
    repo_id: str = "mrxaman19/Indian-deepfake-face",
    num_real: int = 150,
    num_fake: int = 150,
    hf_token: str = None
) -> List[Dict[str, Any]]:
    """List and sample balanced image paths from the Hugging Face repository."""
    from huggingface_hub import HfApi, hf_hub_download
    token = hf_token or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    logger.info(f"Connecting to Hugging Face Hub dataset '{repo_id}'...")
    try:
        files = api.list_repo_files(repo_id, repo_type="dataset")
    except Exception as e:
        logger.error(f"Failed to access repository '{repo_id}': {e}")
        logger.info("Hint: If gated, pass --hf-token or set environment variable HF_TOKEN=<your_token>")
        return []

    real_files = sorted([f for f in files if f.startswith("real/") and f.lower().endswith((".png", ".jpg", ".jpeg"))])
    fake_files = sorted([f for f in files if f.startswith("fake/") and f.lower().endswith((".png", ".jpg", ".jpeg"))])

    rng = np.random.RandomState(42)
    sample_reals = list(rng.choice(real_files, min(len(real_files), num_real), replace=False)) if real_files else []
    sample_fakes = list(rng.choice(fake_files, min(len(fake_files), num_fake), replace=False)) if fake_files else []

    logger.info(f"Discovered in repository: {len(real_files)} Real, {len(fake_files)} Fake images.")
    logger.info(f"Sampled for benchmark:   {len(sample_reals)} Real, {len(sample_fakes)} Fake images.")

    records = []
    for f in sample_reals:
        records.append({"repo_path": f, "is_fake": False, "repo_id": repo_id})
    for f in sample_fakes:
        records.append({"repo_path": f, "is_fake": True, "repo_id": repo_id})

    # Shuffle evaluation order
    rng.shuffle(records)
    return records


def main():
    parser = argparse.ArgumentParser(description="Evaluate deepfake detectors on mrxaman19/Indian-deepfake-face dataset.")
    parser.add_argument("--repo-id", type=str, default="mrxaman19/Indian-deepfake-face", help="Hugging Face dataset repository ID.")
    parser.add_argument("--models", nargs="+", default=["pe", "clip", "vit", "efficientnet-b4"], help="Models to benchmark.")
    parser.add_argument("--num-real", type=int, default=100, help="Number of Real Indian face images to evaluate.")
    parser.add_argument("--num-fake", type=int, default=100, help="Number of Deepfake Indian face images to evaluate.")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face User Access Token (optional if HF_TOKEN env is set).")
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/results/indian_deepfake_benchmark", help="Output directory.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    token = args.hf_token or os.environ.get("HF_TOKEN")

    print("\n" + "="*80)
    print("      BENCHMARK: INDIAN & SOUTH ASIAN DEEPFAKE DETECTION (mrxaman19)")
    print("="*80)

    # 1. Fetch & Sample Image Records
    records = fetch_dataset_files(
        repo_id=args.repo_id,
        num_real=args.num_real,
        num_fake=args.num_fake,
        hf_token=token
    )

    if not records:
        logger.error("No image records found. Exiting.")
        return

    # 2. Initialize Engines (Stages 1, 2, 3)
    detector = SCRFDDetector(conf_threshold=0.35)
    encoder = ArcFaceEncoder()
    forensic_models = init_forensic_models(args.models)
    if not forensic_models:
        logger.error("No active forensic models loaded. Exiting.")
        return

    # 3. Evaluation Loop
    from huggingface_hub import hf_hub_download
    from PIL import Image

    scores_by_model: Dict[str, List[float]] = {m: [] for m in forensic_models}
    labels_true: List[int] = []
    
    total_det_time = 0.0
    total_arc_time = 0.0
    total_model_times = {m: 0.0 for m in forensic_models}
    total_images_processed = 0

    print(f"\nEvaluating {len(records)} Indian face images across full 3-stage pipeline ({len(forensic_models)} forensic models)...")

    for idx, item in enumerate(records):
        try:
            local_path = hf_hub_download(
                repo_id=item["repo_id"],
                filename=item["repo_path"],
                repo_type="dataset",
                token=token
            )
            pil_img = Image.open(local_path).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.warning(f"Could not load image {item['repo_path']}: {e}")
            continue

        total_images_processed += 1
        labels_true.append(1 if item["is_fake"] else 0)

        # Stage 1: SCRFD Face & Landmark Detection
        t0 = time.time()
        dets = detector.detect_frame(img_bgr)
        total_det_time += (time.time() - t0)

        # Stage 2: ArcFace Biometrics & Canonical Alignment
        aligned_face = None
        primary_det = None
        if dets:
            primary_det = max(dets, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
            if primary_det.landmarks_5 is not None:
                aligned_face = align_face_5point(img_bgr, primary_det.landmarks_5)
                if aligned_face is not None:
                    t0 = time.time()
                    encoder.extract_embedding(aligned_face)
                    total_arc_time += (time.time() - t0)

        # Stage 3: Deepfake Forensic Evaluation
        step_scores = {}
        for m_name, model in forensic_models.items():
            # Tailor input crop according to model architecture
            if primary_det is not None:
                if "vit" in m_name.lower():
                    c = crop_context_face(img_bgr, primary_det.bbox, expansion_factor=1.20, target_size=(224, 224))
                elif "efficientnet" in m_name.lower() or "sbi" in m_name.lower():
                    c = crop_context_face(img_bgr, primary_det.bbox, expansion_factor=1.15, target_size=(380, 380))
                else:  # GenD models
                    c = align_face_gend(img_bgr, primary_det.landmarks_5, target_size=(224, 224), scale=1.30)
                    if c is None:
                        c = crop_context_face(img_bgr, primary_det.bbox, target_size=(224, 224))
            else:
                c = cv2.resize(img_bgr, (224, 224))

            t0 = time.time()
            score_list = model.predict_crops([c])
            elapsed = time.time() - t0
            total_model_times[m_name] += elapsed

            score = score_list[0] if score_list else 0.0
            scores_by_model[m_name].append(score)
            step_scores[m_name] = score

        if (idx + 1) % 20 == 0 or (idx + 1) == len(records):
            print(f"  [{idx+1:03d}/{len(records):03d}] True: {'FAKE' if item['is_fake'] else 'REAL':4s} | " +
                  " | ".join([f"{m[:12]}: {step_scores.get(m, 0.0):.4f}" for m in forensic_models]))

    # 4. Compute Metrics (Uncalibrated @ 0.50 and Calibrated @ Optimal T*)
    metrics_by_model = {}
    calibrated_by_model = {}
    for m_name in forensic_models:
        m_dict = compute_forensic_metrics(labels_true, scores_by_model[m_name], threshold=0.50)
        cm = m_dict.get("confusion_matrix", [[0, 0], [0, 0]])
        tn, fp = cm[0][0], cm[0][1]
        fpr = fp / max(1, (fp + tn))
        m_dict["fpr_on_real"] = round(fpr, 4)
        metrics_by_model[m_name] = m_dict
        calibrated_by_model[m_name] = find_optimal_threshold(labels_true, scores_by_model[m_name])

    # 5. Timing Statistics
    avg_det_ms = (total_det_time / max(1, total_images_processed)) * 1000.0
    avg_arc_ms = (total_arc_time / max(1, total_images_processed)) * 1000.0
    timing_stats = {
        "avg_detection_ms": round(avg_det_ms, 2),
        "avg_embedding_ms": round(avg_arc_ms, 2),
        "models": {}
    }

    for m_name in forensic_models:
        avg_inf_ms = (total_model_times[m_name] / max(1, total_images_processed)) * 1000.0
        tot_ms = avg_det_ms + avg_arc_ms + avg_inf_ms
        fps = 1000.0 / tot_ms if tot_ms > 0 else 0.0
        timing_stats["models"][m_name] = {
            "stage_1_detection_ms": round(avg_det_ms, 2),
            "stage_2_biometrics_ms": round(avg_arc_ms, 2),
            "stage_3_forensic_ms": round(avg_inf_ms, 2),
            "total_frame_ms": round(tot_ms, 2),
            "effective_fps": round(fps, 1)
        }

    # 6. Print Console Tables
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
    print("                TABLE 2: DEFAULT FORENSIC PERFORMANCE (THRESHOLD = 0.50)")
    print("="*104)
    print(f"{'Forensic Detector':24s} | {'ROC-AUC':7s} | {'Accuracy':8s} | {'Recall':6s} | {'Precision':9s} | {'FPR (Real)':10s} | {'F1 Score':8s}")
    print("-" * 104)
    for m_name, m_data in metrics_by_model.items():
        print(f"{m_name:24s} | {m_data['roc_auc']:7.4f} | {m_data['accuracy']*100:6.1f} % | {m_data['recall']*100:5.1f}% | {m_data['precision']*100:7.1f}% | {m_data['fpr_on_real']*100:7.1f} % | {m_data.get('f1', 0.0):7.4f}")
    print("="*104)

    print("\n" + "="*124)
    print("                TABLE 3: CALIBRATED FORENSIC PERFORMANCE (OPTIMAL THRESHOLD T*)")
    print("="*124)
    print(f"{'Forensic Detector':24s} | {'Optimal T*':11s} | {'Calibrated Acc':15s} | {'Recall':8s} | {'Precision':10s} | {'FPR (Real)':11s} | {'F1 Score':8s}")
    print("-" * 124)
    for m_name, c_data in calibrated_by_model.items():
        print(f"{m_name:24s} | {c_data['optimal_threshold']:11.3f} | {c_data['calibrated_accuracy']*100:13.1f} % | {c_data['calibrated_recall']*100:6.1f} % | {c_data['calibrated_precision']*100:8.1f} % | {c_data['calibrated_fpr_on_real']*100:9.1f} % | {c_data['calibrated_f1']:8.4f}")
    print("="*124 + "\n")

    # 7. Render 4-Panel Diagnostic Card
    chart_path = os.path.join(args.output_dir, "indian_deepfake_benchmark.png")
    plot_indian_dashboard(
        metrics_by_model=metrics_by_model,
        timing_stats=timing_stats,
        labels_true=labels_true,
        scores_by_model=scores_by_model,
        save_path=chart_path,
        calibrated_by_model=calibrated_by_model
    )

    # 8. Save JSON
    json_path = os.path.join(args.output_dir, "indian_deepfake_results.json")
    with open(json_path, "w") as jf:
        json.dump({
            "dataset": args.repo_id,
            "images_evaluated": total_images_processed,
            "real_images": sum(1 for y in labels_true if y == 0),
            "fake_images": sum(1 for y in labels_true if y == 1),
            "metrics_at_0_50": metrics_by_model,
            "calibrated_metrics": calibrated_by_model,
            "timing": timing_stats
        }, jf, indent=2)

    print(f"[+] Saved 4-Panel Benchmark Report Card to: {chart_path}")
    print(f"[+] Saved Machine-Readable Results to: {json_path}\n")


if __name__ == "__main__":
    main()
