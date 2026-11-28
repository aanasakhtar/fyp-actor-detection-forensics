"""
Full-Scale Celeb-DF v2 Benchmark (~6,529 Videos) with Leakage-Free Train/Test Split.

Implements strict experimental separation across all 3 Celeb-DF v2 subsets
(Celeb-real, YouTube-real, Celeb-synthesis):
  1. Partition all discovered videos into identity-disjoint Train/Calibration (70%)
     and Held-Out Test (30%) sets (or official List_of_testing_videos.txt split).
  2. Batched Stage 1 (SCRFD) + Stage 3 (Forensic) inference with incremental JSONL
     checkpointing so full 6,500+ video runs can resume cleanly if interrupted.
  3. Optimize decision threshold T*_train strictly on the Train/Calibration split via
     Youden's J statistic (TPR - FPR).
  4. Evaluate frozen T*_train blindly on the unseen Held-Out Test split, reporting
     Accuracy, Balanced Accuracy, ROC-AUC, PR-AUC, Recall, Precision, FPR, F1,
     and per-folder breakdown (Celeb-real, YouTube-real, Celeb-synthesis).
"""

import os
import sys
import time
import argparse
import json
import glob
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

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

logger = get_logger("benchmark_celebdf_full")


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
            else:
                logger.warning(f"Unknown forensic model '{name}'. Skipping.")
        except Exception as e:
            logger.error(f"Failed to load model '{name}': {e}")

    return {k: v for k, v in models.items() if getattr(v, "model", None) is not None}


def find_optimal_threshold_on_train(y_true: List[int], scores: List[float]) -> float:
    """Optimize threshold strictly on the Training/Calibration split via Youden's J."""
    if len(set(y_true)) < 2:
        return 0.50
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    opt_thresh = float(thresholds[best_idx])
    return round(max(0.10, min(0.95, opt_thresh)), 4)


def evaluate_at_threshold(y_true: List[int], scores: List[float], threshold: float) -> Dict[str, Any]:
    """Compute comprehensive metrics including Balanced Accuracy at a fixed threshold."""
    m_dict = compute_forensic_metrics(y_true, scores, threshold=threshold)
    cm = m_dict.get("confusion_matrix", [[0, 0], [0, 0]])
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]
    fpr = fp / max(1, (fp + tn))
    tpr = tp / max(1, (tp + fn))
    tnr = tn / max(1, (tn + fp))
    bal_acc = 0.5 * (tpr + tnr)
    m_dict["fpr_on_real"] = round(float(fpr), 4)
    m_dict["specificity"] = round(float(tnr), 4)
    m_dict["balanced_accuracy"] = round(float(bal_acc), 4)
    m_dict["threshold"] = round(float(threshold), 4)
    return m_dict


def discover_celebdf_root(data_dir: str) -> str:
    """Resolve Celeb-DF v2 root containing Celeb-real, Celeb-synthesis, and YouTube-real."""
    if os.path.exists(data_dir) and glob.glob(os.path.join(data_dir, "**", "*.mp4"), recursive=True):
        return data_dir

    logger.warning(f"Directory '{data_dir}' missing or empty. Searching /kaggle/input...")
    matches = (
        glob.glob("/kaggle/input/**/Celeb-synthesis", recursive=True)
        + glob.glob("/kaggle/input/**/Celeb-real", recursive=True)
    )
    if matches:
        resolved = os.path.dirname(matches[0])
        logger.info(f"Auto-discovered Celeb-DF v2 root at: '{resolved}'")
        return resolved
    return data_dir


def partition_train_test(
    loader: FFppDatasetLoader,
    split_mode: str = "identity",
    train_ratio: float = 0.70,
    seed: int = 42,
    data_dir: str = ""
) -> Tuple[List[Any], List[Any]]:
    """
    Partition dataset into Train/Calibration and Held-Out Test sets.
    - 'identity': Identity-disjoint split grouped by target_id (e.g. 70% Train / 30% Test).
    - 'official': Uses Celeb-DF's List_of_testing_videos.txt as Test and remainder as Train.
    """
    if split_mode == "official":
        txt_candidates = glob.glob(os.path.join(data_dir, "**", "*List_of_testing_videos*.txt"), recursive=True)
        if txt_candidates:
            with open(txt_candidates[0], "r", encoding="utf-8") as f:
                test_rel_paths = set()
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        test_rel_paths.add(os.path.basename(parts[1]))
            test_vids = [v for v in loader.videos if os.path.basename(v.video_path) in test_rel_paths]
            train_vids = [v for v in loader.videos if os.path.basename(v.video_path) not in test_rel_paths]
            if test_vids and train_vids:
                logger.info(f"Official Celeb-DF split loaded: Train/Calib={len(train_vids)}, Test={len(test_vids)}")
                return train_vids, test_vids
        logger.warning("Official test list txt not found; falling back to identity-disjoint 70/30 split.")

    train_vids, _, test_vids = loader.get_leakage_free_splits(
        train_ratio=train_ratio,
        val_ratio=0.0,
        test_ratio=round(1.0 - train_ratio, 4),
        seed=seed
    )
    train_ids = set(v.target_id for v in train_vids)
    test_ids = set(v.target_id for v in test_vids)
    assert train_ids.isdisjoint(test_ids), "Identity leakage detected between Train and Test splits!"
    return train_vids, test_vids


def load_checkpoint(checkpoint_path: str) -> Dict[str, Dict[str, Any]]:
    """Load already evaluated video records from incremental JSONL checkpoint."""
    completed: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(checkpoint_path):
        return completed
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = f"{rec['category']}/{rec['video_id']}"
                completed[key] = rec
            except Exception:
                continue
    return completed


def plot_full_celebdf_dashboard(
    test_metrics_05: Dict[str, Dict[str, Any]],
    test_metrics_cal: Dict[str, Dict[str, Any]],
    train_thresholds: Dict[str, float],
    test_labels: List[int],
    test_scores: Dict[str, List[float]],
    test_categories: List[str],
    save_path: str
):
    """Render 4-panel Held-Out Test executive dashboard with Train-Calibrated thresholds."""
    fig, axes = plt.subplots(2, 2, figsize=(17, 12), dpi=150)
    models = list(test_metrics_cal.keys())
    colors = ["#1976D2", "#388E3C", "#D32F2F", "#7B1FA2", "#F57C00"][:len(models)]
    y_true = np.array(test_labels)

    # Panel 1: Held-Out Test Performance (Train-Optimized T*)
    ax1 = axes[0, 0]
    m_keys = ["accuracy", "balanced_accuracy", "roc_auc", "f1"]
    m_labels = ["Accuracy", "Balanced Acc", "ROC-AUC", "F1 Score"]
    x = np.arange(len(m_labels))
    w = 0.8 / max(1, len(models))

    for i, m_name in enumerate(models):
        vals = [test_metrics_cal[m_name].get(k, 0.0) for k in m_keys]
        rects = ax1.bar(x + (i - len(models)/2 + 0.5) * w, vals, w, label=m_name, color=colors[i], alpha=0.88)
        for r in rects:
            h = r.get_height()
            ax1.annotate(f"{h:.2f}", xy=(r.get_x() + r.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(m_labels, fontweight="bold")
    ax1.set_ylim([0.0, 1.15])
    ax1.set_ylabel("Score (0.0 - 1.0)", fontweight="bold")
    ax1.set_title("HELD-OUT TEST METRICS (TRAIN-OPTIMIZED THRESHOLD T*)", fontsize=11.5, fontweight="bold", pad=10)
    ax1.legend(loc="lower right", framealpha=0.9)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel 2: Held-Out Test ROC Curves with Operating Points
    ax2 = axes[0, 1]
    for i, m_name in enumerate(models):
        sc = test_scores[m_name]
        if len(set(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, sc)
            auc_val = test_metrics_cal[m_name].get("roc_auc", 0.0)
            ax2.plot(fpr, tpr, color=colors[i], lw=2.2, label=f"{m_name} (AUC = {auc_val:.4f})")
            op_fpr = test_metrics_cal[m_name].get("fpr_on_real", 0.0)
            op_tpr = test_metrics_cal[m_name].get("recall", 0.0)
            ax2.scatter([op_fpr], [op_tpr], color=colors[i], s=70, zorder=5, edgecolor="black")
    ax2.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.7)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("False Positive Rate (FPR on Real)", fontweight="bold")
    ax2.set_ylabel("True Positive Rate (TPR / Recall)", fontweight="bold")
    ax2.set_title("HELD-OUT TEST ROC CURVES (Stars = Frozen T* Operating Point)", fontsize=11.5, fontweight="bold", pad=10)
    ax2.legend(loc="lower right", framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Panel 3: Held-Out Test Prediction Boxplots (Default 0.50 & Frozen Train T*)
    ax3 = axes[1, 0]
    box_data, box_positions, box_colors = [], [], []
    pos = 1
    for m_name in models:
        s_arr = np.array(test_scores[m_name])
        box_data.extend([s_arr[y_true == 0], s_arr[y_true == 1]])
        box_positions.extend([pos, pos + 0.8])
        box_colors.extend(["#4CAF50", "#E53935"])
        pos += 2.2

    if box_data:
        bp = ax3.boxplot(box_data, positions=box_positions, widths=0.6, patch_artist=True)
        for patch, col in zip(bp["boxes"], box_colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.75)
        ax3.set_xticks([1.4 + i * 2.2 for i in range(len(models))])
        ax3.set_xticklabels(models, fontweight="bold")
        ax3.axhline(0.50, color="black", linestyle=":", lw=1.5, label="Default Threshold (0.50)")
        labeled = False
        for i, m_name in enumerate(models):
            t_val = train_thresholds[m_name]
            cx = 1.4 + i * 2.2
            lbl = "Frozen Train T*" if not labeled else None
            labeled = True
            ax3.hlines(t_val, cx - 0.70, cx + 0.70, colors="#1976D2", linestyles="--", lw=2.2, label=lbl)
            y_text = t_val - 0.045 if t_val > 0.85 else t_val + 0.025
            va = "top" if t_val > 0.85 else "bottom"
            ax3.text(cx, y_text, f"T*={t_val:.2f}", color="#0D47A1", ha="center", va=va, fontsize=8, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="#1976D2", alpha=0.88, lw=0.8))
        ax3.set_ylabel("Manipulation Score p(Fake)", fontweight="bold")
        ax3.set_title("TEST DISTRIBUTION: REAL (Green) vs FAKE (Red)", fontsize=11.5, fontweight="bold", pad=10)
        ax3.grid(axis="y", linestyle="--", alpha=0.5)
        ax3.legend(loc="upper left")

    # Panel 4: Per-Folder Accuracy Breakdown on Held-Out Test Split
    ax4 = axes[1, 1]
    cats = sorted(list(set(test_categories)))
    x4 = np.arange(len(cats))
    w4 = 0.8 / max(1, len(models))
    for i, m_name in enumerate(models):
        t_val = train_thresholds[m_name]
        sc_arr = np.array(test_scores[m_name])
        pred_arr = (sc_arr >= t_val).astype(int)
        cat_accs = []
        for c in cats:
            mask = np.array([tc == c for tc in test_categories])
            cat_accs.append(float(np.mean(pred_arr[mask] == y_true[mask])) * 100.0 if np.any(mask) else 0.0)
        rects = ax4.bar(x4 + (i - len(models)/2 + 0.5) * w4, cat_accs, w4, label=m_name, color=colors[i], alpha=0.88)
        for r in rects:
            h = r.get_height()
            ax4.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7.5)
    ax4.set_xticks(x4)
    ax4.set_xticklabels(cats, fontweight="bold")
    ax4.set_ylim([0.0, 115.0])
    ax4.set_ylabel("Classification Accuracy (%)", fontweight="bold")
    ax4.set_title("HELD-OUT TEST ACCURACY BY CELEB-DF SUBSET (@ FROZEN T*)", fontsize=11.5, fontweight="bold", pad=10)
    ax4.legend(loc="lower right", framealpha=0.9)
    ax4.grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("FULL CELEB-DF v2 BENCHMARK: TRAIN CALIBRATION -> HELD-OUT TEST EVALUATION", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Exhaustive full Celeb-DF v2 benchmark with identity-disjoint Train/Test split.")
    parser.add_argument("--data-dir", type=str, default="/kaggle/input/celeb-df-v2", help="Celeb-DF v2 dataset directory.")
    parser.add_argument("--models", nargs="+", default=["pe", "clip", "vit", "efficientnet-b4"], help="Models to evaluate.")
    parser.add_argument("--split-mode", type=str, default="identity", choices=["identity", "official"], help="Train/Test partitioning strategy.")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Proportion of identity groups for threshold calibration (default: 0.70).")
    parser.add_argument("--max-frames", type=int, default=8, help="Uniform sampled frames per video (default: 8).")
    parser.add_argument("--batch-videos", type=int, default=16, help="Videos to batch together before Stage 3 GPU forward pass.")
    parser.add_argument("--max-videos", type=int, default=0, help="0 = exhaust ALL videos (default); >0 caps total videos for quick debug.")
    parser.add_argument("--with-arcface", action="store_true", help="Also run Stage 2 ArcFace embedding extraction on every frame.")
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/results/celebdf_full_benchmark", help="Output directory.")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face Token.")
    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    os.makedirs(args.output_dir, exist_ok=True)
    data_dir = discover_celebdf_root(args.data_dir)
    loader = FFppDatasetLoader(data_dir)

    if len(loader.videos) == 0:
        logger.error(f"No MP4 videos found in '{data_dir}'. Exiting.")
        return

    # Category summary
    cat_counts: Dict[str, int] = {}
    for v in loader.videos:
        cat_counts[v.category] = cat_counts.get(v.category, 0) + 1

    print("\n" + "="*100)
    print("        FULL CELEB-DF v2 BENCHMARK (IDENTITY-DISJOINT TRAIN CALIBRATION -> TEST EVALUATION)")
    print("="*100)
    print(f"Discovered Dataset Root: {data_dir}")
    print(f"Total Videos Available:  {len(loader.videos)} -> {cat_counts}")

    # Partition into leakage-free Train/Calibration vs Held-Out Test
    train_videos, test_videos = partition_train_test(
        loader, split_mode=args.split_mode, train_ratio=args.train_ratio, seed=42, data_dir=data_dir
    )

    if args.max_videos > 0:
        rng = np.random.RandomState(42)
        n_tr = max(1, int(args.max_videos * args.train_ratio))
        n_te = max(1, args.max_videos - n_tr)
        train_videos = list(rng.choice(train_videos, min(len(train_videos), n_tr), replace=False))
        test_videos = list(rng.choice(test_videos, min(len(test_videos), n_te), replace=False))

    split_assignments = {}
    for v in train_videos:
        split_assignments[f"{v.category}/{v.video_id}"] = ("train", v)
    for v in test_videos:
        split_assignments[f"{v.category}/{v.video_id}"] = ("test", v)

    all_eval_videos = [("train", v) for v in train_videos] + [("test", v) for v in test_videos]
    print(f"Train/Calibration Split: {len(train_videos)} videos ({sum(1 for v in train_videos if not v.is_fake)} Real, {sum(1 for v in train_videos if v.is_fake)} Fake)")
    print(f"Held-Out Test Split:     {len(test_videos)} videos ({sum(1 for v in test_videos if not v.is_fake)} Real, {sum(1 for v in test_videos if v.is_fake)} Fake)")

    # Initialize engines
    detector = SCRFDDetector(conf_threshold=0.50)
    encoder = ArcFaceEncoder() if args.with_arcface else None
    forensic_models = init_forensic_models(args.models)
    if not forensic_models:
        logger.error("No forensic models loaded. Exiting.")
        return

    model_keys = list(forensic_models.keys())
    checkpoint_path = os.path.join(args.output_dir, "celebdf_full_checkpoint.jsonl")
    completed_map = load_checkpoint(checkpoint_path)
    # Re-evaluate only missing videos OR videos missing any requested model
    pending = [
        (sp, v) for (sp, v) in all_eval_videos
        if f"{v.category}/{v.video_id}" not in completed_map
        or not all(m in completed_map[f"{v.category}/{v.video_id}"]["scores"] for m in model_keys)
    ]

    print(f"Checkpoint Status: {len(all_eval_videos) - len(pending)}/{len(all_eval_videos)} already cached; {len(pending)} remaining.\n")

    t_start = time.time()
    batch_size = max(1, args.batch_videos)

    with open(checkpoint_path, "a", encoding="utf-8") as ckpt_f:
        for b_start in range(0, len(pending), batch_size):
            batch_items = pending[b_start:b_start + batch_size]
            batch_crops: Dict[str, List[np.ndarray]] = {m: [] for m in model_keys}
            batch_meta: List[Tuple[str, Any, Dict[str, int]]] = []  # (split, meta, {m: num_crops})

            for split_name, meta in batch_items:
                reader = VideoReader(meta.video_path)
                frames = reader.sample_uniform(max_frames=args.max_frames)
                crop_counts = {m: 0 for m in model_keys}

                for f in frames:
                    dets = detector.detect_frame(f.image_bgr, f.frame_index, f.timestamp_sec)
                    if not dets:
                        continue
                    primary = max(dets, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))

                    if encoder is not None and primary.landmarks_5 is not None:
                        aligned = align_face_5point(f.image_bgr, primary.landmarks_5)
                        if aligned is not None:
                            encoder.extract_embedding(aligned)

                    for m_name, model in forensic_models.items():
                        t_sz = model.get_target_size() if hasattr(model, "get_target_size") else (224, 224)
                        if "vit" in m_name.lower():
                            c = crop_context_face(f.image_bgr, primary.bbox, expansion_factor=1.20, target_size=t_sz)
                        elif "efficientnet" in m_name.lower() or "sbi" in m_name.lower():
                            c = crop_context_face(f.image_bgr, primary.bbox, expansion_factor=1.15, target_size=t_sz)
                        else:
                            c = align_face_gend(f.image_bgr, primary.landmarks_5, target_size=t_sz, scale=1.30)
                            if c is None:
                                c = crop_context_face(f.image_bgr, primary.bbox, expansion_factor=1.30, target_size=t_sz)
                        if c is not None:
                            batch_crops[m_name].append(c)
                            crop_counts[m_name] += 1

                batch_meta.append((split_name, meta, crop_counts))

            # Batched Stage 3 GPU Forward Pass across all videos in the batch
            batch_preds: Dict[str, List[float]] = {}
            for m_name, model in forensic_models.items():
                crops = batch_crops[m_name]
                batch_preds[m_name] = model.predict_crops(crops, batch_size=64) if crops else []

            cursors = {m: 0 for m in model_keys}
            for split_name, meta, crop_counts in batch_meta:
                key = f"{meta.category}/{meta.video_id}"
                existing = completed_map.get(key, {}).get("scores", {})
                vid_scores = dict(existing)

                for m_name, model in forensic_models.items():
                    cnt = crop_counts[m_name]
                    f_scores = batch_preds[m_name][cursors[m_name]:cursors[m_name] + cnt]
                    cursors[m_name] += cnt
                    vid_scores[m_name] = round(float(model.aggregate_scores(f_scores)), 5) if f_scores else 0.0

                record = {
                    "split": split_name,
                    "video_id": meta.video_id,
                    "category": meta.category,
                    "target_id": meta.target_id,
                    "is_fake": bool(meta.is_fake),
                    "scores": vid_scores
                }
                completed_map[key] = record
                ckpt_f.write(json.dumps(record) + "\n")
            ckpt_f.flush()

            done_cnt = min(b_start + batch_size, len(pending))
            elapsed = time.time() - t_start
            rate = done_cnt / max(1e-3, elapsed)
            eta_min = ((len(pending) - done_cnt) / max(1e-3, rate)) / 60.0
            if done_cnt % (batch_size * 5) == 0 or done_cnt == len(pending):
                print(f"  Progress: {done_cnt:4d}/{len(pending):4d} remaining ({rate:.2f} vid/s | ETA: {eta_min:.1f} min)")

    # Assemble Train & Test Arrays cleanly from completed records
    train_records = [completed_map[f"{v.category}/{v.video_id}"] for v in train_videos if f"{v.category}/{v.video_id}" in completed_map]
    test_records = [completed_map[f"{v.category}/{v.video_id}"] for v in test_videos if f"{v.category}/{v.video_id}" in completed_map]

    train_y = [1 if r["is_fake"] else 0 for r in train_records]
    test_y = [1 if r["is_fake"] else 0 for r in test_records]
    test_cats = [r["category"] for r in test_records]

    train_scores = {m: [r["scores"].get(m, 0.0) for r in train_records] for m in model_keys}
    test_scores = {m: [r["scores"].get(m, 0.0) for r in test_records] for m in model_keys}

    # Optimize threshold strictly on Train/Calibration split
    train_thresholds: Dict[str, float] = {}
    train_metrics_cal: Dict[str, Dict[str, Any]] = {}
    test_metrics_05: Dict[str, Dict[str, Any]] = {}
    test_metrics_cal: Dict[str, Dict[str, Any]] = {}

    for m in model_keys:
        t_opt = find_optimal_threshold_on_train(train_y, train_scores[m])
        train_thresholds[m] = t_opt
        train_metrics_cal[m] = evaluate_at_threshold(train_y, train_scores[m], t_opt)
        test_metrics_05[m] = evaluate_at_threshold(test_y, test_scores[m], 0.50)
        test_metrics_cal[m] = evaluate_at_threshold(test_y, test_scores[m], t_opt)

    # Print Tables
    print("\n" + "="*128)
    print(f"      TABLE 1: CALIBRATION / TRAIN SPLIT THRESHOLD OPTIMIZATION (N = {len(train_y)} VIDEOS)")
    print("="*128)
    print(f"{'Forensic Detector':24s} | {'Learned T*':10s} | {'ROC-AUC':7s} | {'Accuracy':8s} | {'Bal Acc':8s} | {'Recall':7s} | {'FPR (Real)':10s} | {'F1':7s}")
    print("-" * 128)
    for m, d in train_metrics_cal.items():
        print(f"{m:24s} | {d['threshold']:10.3f} | {d['roc_auc']:7.4f} | {d['accuracy']*100:6.2f}% | {d['balanced_accuracy']*100:6.2f}% | {d['recall']*100:5.2f}% | {d['fpr_on_real']*100:8.2f}% | {d['f1']:7.4f}")

    print("\n" + "="*128)
    print(f"      TABLE 2: HELD-OUT TEST SPLIT PERFORMANCE @ DEFAULT THRESHOLD = 0.50 (N = {len(test_y)} VIDEOS)")
    print("="*128)
    print(f"{'Forensic Detector':24s} | {'Threshold':10s} | {'ROC-AUC':7s} | {'Accuracy':8s} | {'Bal Acc':8s} | {'Recall':7s} | {'FPR (Real)':10s} | {'F1':7s}")
    print("-" * 128)
    for m, d in test_metrics_05.items():
        print(f"{m:24s} | {0.50:10.3f} | {d['roc_auc']:7.4f} | {d['accuracy']*100:6.2f}% | {d['balanced_accuracy']*100:6.2f}% | {d['recall']*100:5.2f}% | {d['fpr_on_real']*100:8.2f}% | {d['f1']:7.4f}")

    print("\n" + "="*128)
    print(f"      TABLE 3: HELD-OUT TEST SPLIT PERFORMANCE @ FROZEN TRAIN-OPTIMIZED T* (N = {len(test_y)} VIDEOS)")
    print("="*128)
    print(f"{'Forensic Detector':24s} | {'Frozen T*':10s} | {'ROC-AUC':7s} | {'Accuracy':8s} | {'Bal Acc':8s} | {'Recall':7s} | {'FPR (Real)':10s} | {'F1':7s}")
    print("-" * 128)
    for m, d in test_metrics_cal.items():
        print(f"{m:24s} | {d['threshold']:10.3f} | {d['roc_auc']:7.4f} | {d['accuracy']*100:6.2f}% | {d['balanced_accuracy']*100:6.2f}% | {d['recall']*100:5.2f}% | {d['fpr_on_real']*100:8.2f}% | {d['f1']:7.4f}")
    print("="*128 + "\n")

    # Category-level breakdown on Held-Out Test Split
    cats = sorted(list(set(test_cats)))
    print("=" * 96)
    print("      TABLE 4: HELD-OUT TEST ACCURACY BY CELEB-DF FOLDER (%) @ FROZEN T*")
    print("=" * 96)
    cat_hdr = f"{'Forensic Detector':24s} | " + " | ".join([f"{c:18s}" for c in cats])
    print(cat_hdr)
    print("-" * len(cat_hdr))
    y_arr = np.array(test_y)
    cat_breakdown = {m: {} for m in model_keys}
    for m in model_keys:
        pred_arr = (np.array(test_scores[m]) >= train_thresholds[m]).astype(int)
        vals = []
        for c in cats:
            mask = np.array([tc == c for tc in test_cats])
            acc_c = float(np.mean(pred_arr[mask] == y_arr[mask])) * 100.0 if np.any(mask) else 0.0
            cat_breakdown[m][c] = round(acc_c, 2)
            vals.append(f"{acc_c:16.2f} %")
        print(f"{m:24s} | " + " | ".join(vals))
    print("=" * 96 + "\n")

    # Export JSON & Dashboard
    chart_path = os.path.join(args.output_dir, "celebdf_full_benchmark_dashboard.png")
    plot_full_celebdf_dashboard(
        test_metrics_05=test_metrics_05,
        test_metrics_cal=test_metrics_cal,
        train_thresholds=train_thresholds,
        test_labels=test_y,
        test_scores=test_scores,
        test_categories=test_cats,
        save_path=chart_path
    )

    json_path = os.path.join(args.output_dir, "celebdf_full_benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump({
            "dataset_dir": data_dir,
            "split_mode": args.split_mode,
            "train_videos": len(train_records),
            "test_videos": len(test_records),
            "learned_train_thresholds": train_thresholds,
            "train_metrics_calibrated": train_metrics_cal,
            "test_metrics_default_0_50": test_metrics_05,
            "test_metrics_train_calibrated": test_metrics_cal,
            "test_category_breakdown": cat_breakdown,
        }, jf, indent=2)

    print(f"[+] Saved Full Celeb-DF Results JSON: {json_path}")
    print(f"[+] Saved Full Celeb-DF Dashboard:    {chart_path}\n")


if __name__ == "__main__":
    main()
