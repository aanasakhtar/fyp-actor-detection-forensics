"""
Comprehensive South Asian Benchmark on FakeAVCeleb (NeurIPS 2021).

Evaluates the multi-stage pipeline on South Asian celebrities across:
  1. Multiple Manipulation Types: FaceSwap (Deepfakes), FSGAN (Reenactment), and Wav2Lip (Lip sync).
  2. Stage-by-Stage Latency: SCRFD Detection -> ArcFace Biometrics -> Stage 3 Forensics.
  3. Diagnostic Comparison: ViT, EfficientNet-B4, GenD CLIP-L, and GenD PE-L.
  4. Decision Boundary Analysis: Real vs Fake Boxplots & Threshold Calibration (T*).
"""

import os
import sys
import time
import argparse
import json
import glob
from typing import Dict, List, Any, Optional
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Ensure local 'src' directory takes priority in sys.path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

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

logger = get_logger("benchmark_fakeavceleb")


def init_forensic_models(model_names: List[str]) -> Dict[str, Any]:
    """Instantiate requested forensic detection models."""
    models = {}
    for name in model_names:
        key = name.lower().strip()
        try:
            if key in ["vit", "vit-base"]:
                models["ViT"] = ViTDeepfakeDetector()
            elif key in ["efficientnet-b4", "sbi", "efficientnet"]:
                models["EfficientNet-B4 (SBI)"] = SBIEfficientNetDetector()
            elif key in ["gend-pe", "pe"]:
                models["GenD (PE-L)"] = GenDPEDetector()
            elif key in ["gend-clip", "clip"]:
                models["GenD (CLIP-L)"] = GenDCLIPDetector()
            elif key in ["gend-dinov3", "dinov3"]:
                models["GenD (DINOv3-L)"] = GenDDINOv3Detector()
        except Exception as e:
            logger.error(f"Failed to load model '{name}': {e}")

    return {k: v for k, v in models.items() if getattr(v, "model", None) is not None}


def find_optimal_threshold(y_true: List[int], scores: List[float]) -> Dict[str, Any]:
    """Find optimal decision threshold maximizing Youden's J statistic (TPR - FPR)."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    opt_thresh = float(thresholds[best_idx])
    
    # Clip to valid bounds [0.10, 0.92]
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


def plot_fakeavceleb_dashboard(
    metrics_by_model: Dict[str, Dict[str, Any]],
    method_accuracies: Dict[str, Dict[str, float]],
    timing_stats: Dict[str, Any],
    labels_true: List[int],
    scores_by_model: Dict[str, List[float]],
    unique_methods: List[str],
    save_path: str,
    calibrated_by_model: Optional[Dict[str, Dict[str, Any]]] = None
):
    """Render 4-panel executive diagnostic dashboard with boxplots and latency breakdown."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
    models = list(metrics_by_model.keys())
    colors = ["#1976D2", "#388E3C", "#D32F2F", "#7B1FA2"][:len(models)]

    # 1. Panel 1: Accuracy by Manipulation Method
    ax1 = axes[0, 0]
    x = np.arange(len(unique_methods))
    w = 0.8 / max(1, len(models))
    for i, m in enumerate(models):
        accs = [method_accuracies[m].get(meth, 0.0) for meth in unique_methods]
        rects = ax1.bar(x + (i - len(models)/2 + 0.5) * w, accs, w, label=m, color=colors[i], alpha=0.88)
        for r in rects:
            h = r.get_height()
            ax1.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.upper() for m in unique_methods], fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.set_ylabel("Accuracy (%)", fontweight="bold")
    ax1.set_title("ACCURACY BY DEEPFAKE MANIPULATION METHOD", fontsize=12, fontweight="bold", pad=10)
    ax1.legend(loc="lower right", framealpha=0.9)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # 2. Panel 2: ROC Curves
    ax2 = axes[0, 1]
    y_true = np.array(labels_true)
    for i, (m, sc) in enumerate(scores_by_model.items()):
        if len(set(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, sc)
            auc_val = metrics_by_model[m].get("roc_auc", 0.0)
            ax2.plot(fpr, tpr, color=colors[i], lw=2.2, label=f"{m} (AUC = {auc_val:.4f})")
    ax2.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.7)
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
    ax4.set_xlabel("Latency per Frame (Milliseconds - Lower is Faster)", fontweight="bold")
    ax4.set_title("END-TO-END PIPELINE LATENCY BREAKDOWN (Stages 1, 2, 3)", fontsize=12, fontweight="bold", pad=10)
    ax4.legend(loc="lower right", framealpha=0.92)
    ax4.grid(axis="x", linestyle="--", alpha=0.5)

    plt.suptitle("BENCHMARK: FAKEAVCELEB SOUTH ASIAN MULTIMODAL DEEPFAKE EVALUATION", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def parse_fakeavceleb_filename(filepath: str) -> Dict[str, Any]:
    """
    Parse FakeAVCeleb filename conventions:
    Real: '00001.mp4' or 'RealVideo-RealAudio/...'
    Fake: '00005_id00018_sap-FjURHlc_id00184_wavtolip.mp4'
          'id00018_id00184_faceswap.mp4'
          'id00018_fsgan.mp4'
    """
    base = os.path.basename(filepath)
    lower = filepath.lower()

    # Determine method
    if "wav2lip" in lower or "wavtolip" in lower:
        method = "wav2lip"
    elif "faceswap" in lower:
        method = "faceswap"
    elif "fsgan" in lower:
        method = "fsgan"
    elif "real" in lower:
        method = "real"
    else:
        method = "unknown"

    is_fake = method != "real" and "realvideo" not in lower
    return {
        "video_path": filepath,
        "video_id": os.path.splitext(base)[0],
        "method": method,
        "is_fake": is_fake
    }


def discover_fakeavceleb_videos(data_dir: str, max_videos: Optional[int] = None) -> List[Dict[str, Any]]:
    """Scan data directory for FakeAVCeleb video files."""
    if not os.path.exists(data_dir):
        logger.warning(f"Directory '{data_dir}' does not exist. Searching /kaggle/input...")
        matches = glob.glob("/kaggle/input/**/FakeVideo*", recursive=True) + \
                  glob.glob("/kaggle/input/**/RealVideo*", recursive=True) + \
                  glob.glob("/kaggle/input/**/fakeavceleb*", recursive=True)
        if matches:
            data_dir = os.path.dirname(matches[0])
            logger.info(f"Discovered FakeAVCeleb dataset root at: '{data_dir}'")
        else:
            if os.path.exists("/kaggle/input"):
                mounted = os.listdir("/kaggle/input")
                logger.error(f"Datasets currently mounted in /kaggle/input: {mounted}")
            else:
                logger.error("Path /kaggle/input does not exist.")

    mp4_files = glob.glob(os.path.join(data_dir, "**", "*.mp4"), recursive=True)
    if not mp4_files:
        logger.error(f"No MP4 files found in '{data_dir}'")
        return []

    records = [parse_fakeavceleb_filename(f) for f in mp4_files]
    reals = [r for r in records if not r["is_fake"]]
    fakes = [r for r in records if r["is_fake"]]

    logger.info(f"Discovered in directory: {len(records)} total ({len(reals)} Real, {len(fakes)} Deepfake)")

    if max_videos and len(records) > max_videos:
        n_half = max_videos // 2
        rng = np.random.RandomState(42)
        sample_reals = list(rng.choice(reals, min(len(reals), n_half), replace=False)) if reals else []
        sample_fakes = list(rng.choice(fakes, min(len(fakes), max_videos - len(sample_reals)), replace=False)) if fakes else []
        records = sample_reals + sample_fakes
        rng.shuffle(records)

    return records


def main():
    parser = argparse.ArgumentParser(description="Evaluate deepfake pipeline on FakeAVCeleb South Asian benchmarks.")
    parser.add_argument("--data-dir", type=str, default="/kaggle/input/fakeavceleb", help="Path to FakeAVCeleb videos.")
    parser.add_argument("--models", nargs="+", default=["pe", "clip", "vit", "efficientnet-b4"], help="Forensic models to benchmark.")
    parser.add_argument("--max-videos", type=int, default=80, help="Maximum number of videos to evaluate.")
    parser.add_argument("--max-frames", type=int, default=10, help="Uniform sampled frames per video.")
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/results/fakeavceleb_benchmark", help="Output directory.")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face User Access Token (optional if HF_TOKEN env is set).")
    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*84)
    print("      BENCHMARK: FAKEAVCELEB SOUTH ASIAN MULTIMODAL DEEPFAKE EVALUATION")
    print("="*84)

    # 1. Discover Videos
    video_records = discover_fakeavceleb_videos(args.data_dir, max_videos=args.max_videos)
    if not video_records:
        logger.error("No valid video records discovered. Please check --data-dir.")
        return

    n_real = sum(1 for v in video_records if not v["is_fake"])
    n_fake = sum(1 for v in video_records if v["is_fake"])
    logger.info(f"Target Evaluation Set: {len(video_records)} videos ({n_real} Real, {n_fake} Deepfake)")

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
    methods_list: List[str] = []

    total_det_time = 0.0
    total_arc_time = 0.0
    total_model_times = {m: 0.0 for m in forensic_models}
    total_frames_processed = 0
    total_faces_processed = 0

    print(f"\nProcessing {len(video_records)} videos with {len(forensic_models)} forensic models...")

    for v_idx, meta in enumerate(video_records):
        reader = VideoReader(meta["video_path"])
        frames = reader.sample_uniform(max_frames=args.max_frames)
        if not frames:
            continue

        vid_crops_by_model = {m: [] for m in forensic_models}

        # Stage 1: SCRFD & Stage 2: ArcFace
        for f in frames:
            t0 = time.time()
            dets = detector.detect_frame(f.image_bgr, f.frame_index, f.timestamp_sec)
            total_det_time += (time.time() - t0)
            total_frames_processed += 1

            if dets:
                primary = max(dets, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
                total_faces_processed += 1

                if primary.landmarks_5 is not None:
                    aligned = align_face_5point(f.image_bgr, primary.landmarks_5)
                    if aligned is not None:
                        t0 = time.time()
                        encoder.extract_embedding(aligned)
                        total_arc_time += (time.time() - t0)

                for m_name in forensic_models:
                    if "vit" in m_name.lower():
                        c = crop_context_face(f.image_bgr, primary.bbox, expansion_factor=1.20, target_size=(224, 224))
                    elif "efficientnet" in m_name.lower() or "sbi" in m_name.lower():
                        c = crop_context_face(f.image_bgr, primary.bbox, expansion_factor=1.15, target_size=(380, 380))
                    else:
                        c = align_face_gend(f.image_bgr, primary.landmarks_5, target_size=(224, 224), scale=1.30)
                        if c is None:
                            c = crop_context_face(f.image_bgr, primary.bbox, target_size=(224, 224))
                    if c is not None:
                        vid_crops_by_model[m_name].append(c)

        # Stage 3: Inference
        vid_scores = {}
        for m_name, model in forensic_models.items():
            crops = vid_crops_by_model[m_name]
            if crops:
                t0 = time.time()
                f_scores = model.predict_crops(crops)
                total_model_times[m_name] += (time.time() - t0)
                agg = model.aggregate_scores(f_scores)
            else:
                agg = 0.0
            vid_scores[m_name] = agg
            scores_by_model[m_name].append(agg)

        labels_true.append(1 if meta["is_fake"] else 0)
        methods_list.append(meta["method"])

        if (v_idx + 1) % 10 == 0 or (v_idx + 1) == len(video_records):
            print(f"  [{v_idx+1:03d}/{len(video_records):03d}] {meta['method']:12s} | "
                  f"True: {'FAKE' if meta['is_fake'] else 'REAL':4s} | " + 
                  " | ".join([f"{m[:12]}: {vid_scores.get(m, 0.0):.4f}" for m in forensic_models]))

    # 3. Compute Metrics (Default Threshold = 0.50 and Calibrated T*)
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
            "total_frame_ms": round(tot_ms, 2),
            "effective_fps": round(fps, 1)
        }

    # 5. Method-by-Method Breakdown
    unique_methods = sorted(list(set(methods_list)))
    method_accuracies: Dict[str, Dict[str, float]] = {m: {} for m in forensic_models}
    for m_name in forensic_models:
        sc_arr = np.array(scores_by_model[m_name])
        y_arr = np.array(labels_true)
        pred_arr = (sc_arr >= 0.50).astype(int)
        for meth in unique_methods:
            mask = np.array([m == meth for m in methods_list])
            if np.sum(mask) > 0:
                acc = np.mean(pred_arr[mask] == y_arr[mask])
                method_accuracies[m_name][meth] = round(float(acc) * 100, 1)

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
    print("="*124)

    print("\n" + "="*84)
    print("               TABLE 4: ACCURACY BY MANIPULATION METHOD (%)")
    print("="*84)
    header = f"{'Forensic Detector':24s} | " + " | ".join([f"{meth:12s}" for meth in unique_methods])
    print(header)
    print("-" * len(header))
    for m_name in forensic_models:
        row = f"{m_name:24s} | " + " | ".join([f"{method_accuracies[m_name].get(meth, 0.0):10.1f} %" for meth in unique_methods])
        print(row)
    print("="*84 + "\n")

    # 7. Render 4-Panel Diagnostic Dashboard
    chart_path = os.path.join(args.output_dir, "fakeavceleb_benchmark.png")
    plot_fakeavceleb_dashboard(
        metrics_by_model=metrics_by_model,
        method_accuracies=method_accuracies,
        timing_stats=timing_stats,
        labels_true=labels_true,
        scores_by_model=scores_by_model,
        unique_methods=unique_methods,
        save_path=chart_path,
        calibrated_by_model=calibrated_by_model
    )

    # 8. Save JSON Export
    json_path = os.path.join(args.output_dir, "fakeavceleb_benchmark_results.json")
    with open(json_path, "w") as jf:
        json.dump({
            "dataset_dir": args.data_dir,
            "videos_evaluated": len(video_records),
            "real_videos": n_real,
            "fake_videos": n_fake,
            "metrics_at_0_50": metrics_by_model,
            "calibrated_metrics": calibrated_by_model,
            "timing_benchmarks": timing_stats,
            "method_breakdown": method_accuracies
        }, jf, indent=2)

    print(f"[+] Saved 4-Panel Benchmark Report Card to: {chart_path}")
    print(f"[+] Saved Machine-Readable Results to: {json_path}\n")


if __name__ == "__main__":
    main()
