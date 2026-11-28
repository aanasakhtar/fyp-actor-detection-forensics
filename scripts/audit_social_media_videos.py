"""
In-the-Wild Social Media Forensic Audit Script (Baseline vs. Augmented TTA Pipeline).

Evaluates user-downloaded social media videos (YouTube, Instagram, Facebook, TikTok)
across the full 3-Stage Pipeline:
  Stage 1: SCRFD Face Detection & 5-point landmark extraction.
  Stage 2: ArcFace Biometrics & Temporal Identity Consistency analysis.
  Stage 3: Multi-Model Deepfake Forensics: GenD (PE-L), GenD (CLIP-L), and ViT.

Compares:
  - Baseline Mode: Single crop, no quality gating, trimmed-mean aggregation.
  - Augmented Mode (Enhanced): Landmark Geometric Sanity + IPD/Conf Gating +
    4-View Multi-Scale Test-Time Augmentation (TTA) + Native Resolution + Median Aggregation.

Applies a constant user-defined decision threshold (default: 0.75) across all models.
"""

import os
import sys
import argparse
import json
import glob
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Ensure local 'src' directory takes priority in sys.path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from pipeline.data.video_reader import VideoReader
from pipeline.detection.scrfd_detector import SCRFDDetector
from pipeline.evaluation.detection_metrics import (
    compute_inter_pupillary_distance,
    compute_landmark_geometric_validity,
)
from pipeline.evaluation.forensic_metrics import compute_forensic_metrics
from pipeline.preprocessing.face_alignment import align_face_5point, align_face_gend
from pipeline.preprocessing.face_cropping import crop_context_face, extract_tta_face_views
from pipeline.recognition.arcface_encoder import ArcFaceEncoder
from pipeline.forensics.vit_detector import ViTDeepfakeDetector
from pipeline.forensics.clip_detector import GenDCLIPDetector
from pipeline.forensics.pe_detector import GenDPEDetector
from pipeline.forensics.sbi_detector import SBIEfficientNetDetector
from pipeline.utils.logging import get_logger

logger = get_logger("audit_social_media")


def init_forensic_models(model_names: List[str]) -> Dict[str, Any]:
    """Instantiate requested forensic detection models."""
    models = {}
    for name in model_names:
        key = name.lower().strip()
        try:
            if key in ["vit", "vit-base"]:
                logger.info("Initializing ViT Deepfake Detector ('dima806/deepfake_vs_real_image_detection')...")
                models["ViT"] = ViTDeepfakeDetector()
            elif key in ["gend-clip", "clip"]:
                logger.info("Initializing GenD CLIP-L/14 ('yermandy/GenD_CLIP_L_14')...")
                models["GenD (CLIP-L)"] = GenDCLIPDetector()
            elif key in ["gend-pe", "pe"]:
                logger.info("Initializing GenD PE-L ('yermandy/GenD_PE_L')...")
                models["GenD (PE-L)"] = GenDPEDetector()
            elif key in ["efficientnet-b4", "sbi", "efficientnet"]:
                logger.info("Initializing SBI EfficientNet-B4 ('nikokons/contrastive-deepfake-detector')...")
                models["EfficientNet-B4 (SBI)"] = SBIEfficientNetDetector()
            else:
                logger.warning(f"Unknown forensic model '{name}'. Skipping.")
        except Exception as e:
            logger.error(f"Failed to load model '{name}': {e}")

    return {k: v for k, v in models.items() if getattr(v, "model", None) is not None}


def parse_video_metadata(video_path: str) -> Dict[str, Any]:
    """Extract platform and video identifier from file naming convention."""
    base = os.path.basename(video_path)
    stem, _ = os.path.splitext(base)
    lower = stem.lower()

    if "facebook" in lower or "fb_" in lower:
        platform = "Facebook"
    elif "instagram" in lower or "insta" in lower or "ig_" in lower:
        platform = "Instagram"
    elif "youtube" in lower or "yt_" in lower:
        platform = "YouTube"
    elif "tiktok" in lower:
        platform = "TikTok"
    else:
        platform = "Social Media"

    return {
        "video_path": video_path,
        "video_name": base,
        "video_id": stem,
        "platform": platform,
        "file_size_mb": round(os.path.getsize(video_path) / (1024 * 1024), 2)
    }


def discover_input_videos(input_path: str) -> List[Dict[str, Any]]:
    """Discover all valid video files from a file path or directory."""
    valid_extensions = ("*.mp4", "*.mov", "*.mkv", "*.webm", "*.avi")

    if os.path.isfile(input_path):
        return [parse_video_metadata(input_path)]

    search_dirs = [
        input_path,
        os.path.abspath(input_path),
        os.path.join(os.path.dirname(__file__), "..", "video_storage"),
        "/kaggle/working/fyp-actor-detection-forensics/video_storage",
        "/kaggle/working/video_storage"
    ]

    found_files = []
    target_dir = None
    for d in search_dirs:
        if os.path.exists(d) and os.path.isdir(d):
            for ext in valid_extensions:
                found_files.extend(glob.glob(os.path.join(d, "**", ext), recursive=True))
                found_files.extend(glob.glob(os.path.join(d, ext)))
            if found_files:
                target_dir = d
                break

    if not found_files and os.path.exists("/kaggle/input"):
        logger.warning("Searching /kaggle/input for social media video files...")
        for ext in valid_extensions:
            found_files.extend(glob.glob(f"/kaggle/input/**/{ext}", recursive=True))

    found_files = sorted(list(set(found_files)))
    if not found_files:
        logger.error(f"No video files found in '{input_path}' or fallback search locations.")
        return []

    logger.info(f"Discovered {len(found_files)} social media videos from: '{target_dir or input_path}'")
    return [parse_video_metadata(f) for f in found_files]


def parse_ground_truth_labels(label_args: Optional[List[str]]) -> Dict[str, int]:
    """Parse optional CLI ground truth pairs like 'filename.mp4=0' (0=Real, 1=Fake)."""
    labels = {}
    if not label_args:
        return labels
    for item in label_args:
        if "=" in item:
            k, v = item.split("=", 1)
            k = k.strip()
            v_int = 1 if v.strip().lower() in ["1", "fake", "deepfake", "pos", "true"] else 0
            labels[k] = v_int
            labels[os.path.splitext(k)[0]] = v_int
    return labels


def compute_temporal_identity_consistency(embeddings: List[np.ndarray]) -> Tuple[float, float]:
    """Compute consecutive pairwise ArcFace cosine similarity and centroid similarity."""
    if not embeddings or len(embeddings) < 2:
        return 1.0, 1.0

    consec_sims = []
    for i in range(len(embeddings) - 1):
        e1, e2 = embeddings[i], embeddings[i + 1]
        dot = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8))
        consec_sims.append(dot)

    centroid = np.mean(embeddings, axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
    centroid_sims = [float(np.dot(e, centroid) / (np.linalg.norm(e) + 1e-8)) for e in embeddings]
    return round(float(np.mean(consec_sims)), 4), round(float(np.mean(centroid_sims)), 4)


def is_high_quality_face(det, min_conf: float = 0.65, min_ipd: float = 28.0) -> bool:
    """Gate out blurry, extreme side-profile, or geometrically invalid detections."""
    if det.confidence < min_conf:
        return False
    if det.landmarks_5 is None:
        return False
    if not compute_landmark_geometric_validity(det.landmarks_5, det.bbox):
        return False
    ipd = compute_inter_pupillary_distance(det.landmarks_5)
    return ipd >= min_ipd


def determine_constant_threshold_verdict(scores: Dict[str, float], threshold: float) -> Tuple[str, float]:
    """Compute GenD ensemble average and verdict against a constant threshold."""
    gend_scores = [v for k, v in scores.items() if "gend" in k.lower()]
    ensemble_score = float(np.mean(gend_scores)) if gend_scores else float(np.mean(list(scores.values())))
    pe_score = scores.get("GenD (PE-L)", ensemble_score)
    clip_score = scores.get("GenD (CLIP-L)", ensemble_score)

    if pe_score >= threshold and clip_score >= threshold:
        return "FAKE (DUAL GEND)", round(ensemble_score, 4)
    elif ensemble_score >= threshold:
        return "FAKE (ENSEMBLE)", round(ensemble_score, 4)
    return "REAL (AUTHENTIC)", round(ensemble_score, 4)


def plot_social_media_audit_dashboard(
    audit_results: List[Dict[str, Any]],
    model_names: List[str],
    threshold: float,
    save_path: str
):
    """Generate 4-panel executive dashboard comparing Baseline vs Augmented TTA at constant threshold."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), dpi=150)
    video_labels = [r["video_name"][:20] for r in audit_results]
    n_vids = len(audit_results)
    x = np.arange(n_vids)

    # Panel 1: Augmented (TTA + Quality Gated) Model Scores
    ax1 = axes[0, 0]
    w = 0.8 / max(1, len(model_names))
    model_colors = {"GenD (PE-L)": "#D32F2F", "GenD (CLIP-L)": "#388E3C", "ViT": "#1976D2", "EfficientNet-B4 (SBI)": "#7B1FA2"}

    for i, m_name in enumerate(model_names):
        scores = [r["enhanced_scores"].get(m_name, 0.0) for r in audit_results]
        col = model_colors.get(m_name, "#546E7A")
        rects = ax1.bar(x + (i - len(model_names)/2 + 0.5) * w, scores, w, label=m_name, color=col, alpha=0.88)
        for r_bar in rects:
            h = r_bar.get_height()
            ax1.annotate(f"{h:.2f}", xy=(r_bar.get_x() + r_bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7.5)

    ax1.axhline(threshold, color="black", linestyle="--", lw=2.0, label=f"Constant Threshold ({threshold:.2f})")
    ax1.set_xticks(x)
    ax1.set_xticklabels(video_labels, rotation=20, ha="right", fontsize=9, fontweight="bold")
    ax1.set_ylim([0.0, 1.15])
    ax1.set_ylabel("Manipulation Score p(Fake)", fontweight="bold")
    ax1.set_title(f"AUGMENTED (4-VIEW TTA + QUALITY GATE) SCORES @ T={threshold:.2f}", fontsize=11, fontweight="bold", pad=10)
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=8.5)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel 2: Baseline vs Augmented GenD (PE-L) Comparison
    ax2 = axes[0, 1]
    target_model = "GenD (PE-L)" if "GenD (PE-L)" in model_names else model_names[0]
    base_vals = [r["baseline_scores"].get(target_model, 0.0) for r in audit_results]
    enh_vals = [r["enhanced_scores"].get(target_model, 0.0) for r in audit_results]
    w2 = 0.35
    ax2.bar(x - w2/2, base_vals, w2, label=f"Baseline ({target_model})", color="#90A4AE", alpha=0.9)
    ax2.bar(x + w2/2, enh_vals, w2, label=f"Augmented TTA ({target_model})", color="#1976D2", alpha=0.9)
    ax2.axhline(threshold, color="black", linestyle="--", lw=2.0, label=f"Decision Cutoff ({threshold:.2f})")
    for xi, (bv, ev) in enumerate(zip(base_vals, enh_vals)):
        ax2.text(xi - w2/2, bv + 0.02, f"{bv:.2f}", ha="center", fontsize=8)
        ax2.text(xi + w2/2, ev + 0.02, f"{ev:.2f}", ha="center", fontsize=8, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(video_labels, rotation=20, ha="right", fontsize=9, fontweight="bold")
    ax2.set_ylim([0.0, 1.15])
    ax2.set_ylabel("Manipulation Score p(Fake)", fontweight="bold")
    ax2.set_title(f"ABLATION: BASELINE vs TTA+QUALITY GATING ({target_model})", fontsize=11, fontweight="bold", pad=10)
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=8.5)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel 3: Representative Thumbnails
    ax3 = axes[1, 0]
    ax3.axis("off")
    ax3.set_title("STAGE 1 & 2: VISUAL INSPECTION SAMPLES (QUALITY GATED)", fontsize=11, fontweight="bold", pad=10)
    n_display = min(5, n_vids)
    if n_display > 0:
        sub_grid = np.zeros((240, 240 * n_display, 3), dtype=np.uint8)
        for i in range(n_display):
            thumb = audit_results[i].get("representative_thumb")
            if thumb is not None:
                resized = cv2.resize(thumb, (240, 240))
                col_bgr = (0, 0, 255) if "FAKE" in audit_results[i]["enhanced_verdict"] else (0, 200, 0)
                cv2.rectangle(resized, (0, 0), (239, 239), col_bgr, 6)
                sub_grid[:, i*240:(i+1)*240] = resized
        ax3.imshow(cv2.cvtColor(sub_grid, cv2.COLOR_BGR2RGB))
        for i in range(n_display):
            v_name = audit_results[i]["video_name"][:14]
            gt_tag = ""
            if audit_results[i].get("ground_truth") is not None:
                gt_tag = f"\nGT: {'FAKE' if audit_results[i]['ground_truth'] == 1 else 'REAL'}"
            ax3.text(i*240 + 120, 255, f"{v_name}\nPred: {audit_results[i]['enhanced_verdict']}{gt_tag}",
                     ha="center", va="top", fontsize=7.5, fontweight="bold")

    # Panel 4: Temporal Identity Consistency (ArcFace)
    ax4 = axes[1, 1]
    pair_sims = [r["arcface_pairwise_sim"] for r in audit_results]
    gated_ratios = [r["quality_pass_rate"] * 100.0 for r in audit_results]
    ax4.bar(x - w2/2, [s * 100.0 for s in pair_sims], w2, label="ArcFace Id Consistency (%)", color="#009688", alpha=0.85)
    ax4.bar(x + w2/2, gated_ratios, w2, label="Quality-Gated Frames (%)", color="#FF9800", alpha=0.85)
    ax4.set_xticks(x)
    ax4.set_xticklabels(video_labels, rotation=20, ha="right", fontsize=9, fontweight="bold")
    ax4.set_ylim([0.0, 115.0])
    ax4.set_ylabel("Percentage (%)", fontweight="bold")
    ax4.set_title("BIOMETRIC IDENTITY STABILITY & QUALITY GATE COVERAGE", fontsize=11, fontweight="bold", pad=10)
    ax4.legend(loc="lower right", framealpha=0.9)
    ax4.grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle(f"IN-THE-WILD SOCIAL MEDIA FORENSIC AUDIT (CONSTANT THRESHOLD = {threshold:.2f})", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def print_metrics_comparison_table(
    audit_results: List[Dict[str, Any]],
    model_names: List[str],
    threshold: float
):
    """If ground truth labels are supplied, compute and print exact accuracy comparison."""
    labeled = [r for r in audit_results if r.get("ground_truth") is not None]
    if not labeled or len(set(r["ground_truth"] for r in labeled)) < 1:
        return

    y_true = [r["ground_truth"] for r in labeled]
    print("\n" + "="*118)
    print(f"       TABLE 3: ACCURACY ABLATION AT CONSTANT THRESHOLD T = {threshold:.2f} (N = {len(labeled)} LABELED VIDEOS)")
    print("="*118)
    print(f"{'Pipeline Variant / Detector':34s} | {'Accuracy':8s} | {'Recall':7s} | {'Precision':9s} | {'FPR (Real)':10s} | {'F1 Score':8s} | {'ROC-AUC':7s}")
    print("-" * 118)

    for mode_label, key_name in [("Baseline (Single Crop)", "baseline_scores"), ("Enhanced (4-TTA + Quality Gate)", "enhanced_scores")]:
        for m_name in model_names:
            scores = [r[key_name].get(m_name, 0.0) for r in labeled]
            m_dict = compute_forensic_metrics(y_true, scores, threshold=threshold)
            cm = m_dict.get("confusion_matrix", [[0, 0], [0, 0]])
            tn, fp = cm[0][0], cm[0][1]
            fpr = fp / max(1, (fp + tn))
            label = f"{mode_label[:13]} | {m_name}"
            print(f"{label:34s} | {m_dict['accuracy']*100:6.1f} % | {m_dict['recall']*100:5.1f}% | {m_dict['precision']*100:7.1f}% | {fpr*100:7.1f} % | {m_dict.get('f1', 0.0):8.4f} | {m_dict.get('roc_auc', 0.0):7.4f}")

        # Also evaluate Dual-GenD Ensemble if >1 GenD model active
        gend_keys = [m for m in model_names if "gend" in m.lower()]
        if len(gend_keys) >= 2:
            ens_scores = [float(np.mean([r[key_name].get(gk, 0.0) for gk in gend_keys])) for r in labeled]
            m_dict = compute_forensic_metrics(y_true, ens_scores, threshold=threshold)
            cm = m_dict.get("confusion_matrix", [[0, 0], [0, 0]])
            tn, fp = cm[0][0], cm[0][1]
            fpr = fp / max(1, (fp + tn))
            label = f"{mode_label[:13]} | Dual-GenD Ensemble"
            print(f"{label:34s} | {m_dict['accuracy']*100:6.1f} % | {m_dict['recall']*100:5.1f}% | {m_dict['precision']*100:7.1f}% | {fpr*100:7.1f} % | {m_dict.get('f1', 0.0):8.4f} | {m_dict.get('roc_auc', 0.0):7.4f}")
        print("-" * 118)
    print("="*118 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Audit social media videos with Baseline vs Enhanced TTA pipeline at fixed threshold.")
    parser.add_argument("--input-dir", type=str, default="video_storage", help="Directory containing videos or single video file.")
    parser.add_argument("--models", nargs="+", default=["pe", "clip", "vit"], help="Forensic models to evaluate.")
    parser.add_argument("--threshold", type=float, default=0.75, help="Constant decision threshold (default: 0.75).")
    parser.add_argument("--max-frames", type=int, default=16, help="Uniform frames to sample per video.")
    parser.add_argument("--labels", nargs="*", default=None, help="Optional ground truth pairs e.g. video.mp4=0 (0=Real, 1=Fake).")
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/results/social_media_audit", help="Output directory.")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face User Access Token.")
    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    os.makedirs(args.output_dir, exist_ok=True)
    gt_map = parse_ground_truth_labels(args.labels)

    print("\n" + "="*108)
    print(f"   IN-THE-WILD SOCIAL MEDIA FORENSIC AUDIT (CONSTANT THRESHOLD = {args.threshold:.2f})")
    print("="*108)

    video_records = discover_input_videos(args.input_dir)
    if not video_records:
        return

    detector = SCRFDDetector(conf_threshold=0.50)
    encoder = ArcFaceEncoder()
    forensic_models = init_forensic_models(args.models)
    if not forensic_models:
        logger.error("No forensic models loaded. Exiting.")
        return

    audit_results = []
    for v_idx, meta in enumerate(video_records):
        v_path = meta["video_path"]
        reader = VideoReader(v_path)
        frames = reader.sample_uniform(max_frames=args.max_frames)
        if not frames:
            continue

        base_crops = {m: [] for m in forensic_models}
        enh_frame_views = {m: [] for m in forensic_models}  # List of TTA view lists per quality-gated frame
        all_confs = []
        high_q_count = 0
        face_embeddings = []
        representative_thumb = None

        for f in frames:
            dets = detector.detect_frame(f.image_bgr, f.frame_index, f.timestamp_sec)
            if not dets:
                continue

            primary = max(dets, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
            all_confs.append(float(primary.confidence))
            hq = is_high_quality_face(primary)
            if hq:
                high_q_count += 1

            if primary.landmarks_5 is not None:
                aligned = align_face_5point(f.image_bgr, primary.landmarks_5)
                if aligned is not None:
                    emb = encoder.extract_embedding(aligned)
                    if emb is not None:
                        face_embeddings.append(emb)

            if representative_thumb is None or (hq and high_q_count == 1):
                thumb_copy = f.image_bgr.copy()
                x1, y1, x2, y2 = [int(v) for v in primary.bbox]
                cv2.rectangle(thumb_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if primary.landmarks_5 is not None:
                    for lx, ly in primary.landmarks_5:
                        cv2.circle(thumb_copy, (int(lx), int(ly)), 3, (0, 0, 255), -1)
                representative_thumb = thumb_copy

            for m_name, model in forensic_models.items():
                t_sz = model.get_target_size() if hasattr(model, "get_target_size") else (224, 224)
                is_vit = "vit" in m_name.lower()
                is_sbi = "efficientnet" in m_name.lower() or "sbi" in m_name.lower()

                # 1. Baseline crop
                if is_vit:
                    c_base = crop_context_face(f.image_bgr, primary.bbox, expansion_factor=1.20, target_size=t_sz)
                elif is_sbi:
                    c_base = crop_context_face(f.image_bgr, primary.bbox, expansion_factor=1.15, target_size=t_sz)
                else:
                    c_base = align_face_gend(f.image_bgr, primary.landmarks_5, target_size=(224, 224), scale=1.30)
                    if c_base is None:
                        c_base = crop_context_face(f.image_bgr, primary.bbox, target_size=(224, 224))
                if c_base is not None:
                    base_crops[m_name].append(c_base)

                # 2. Enhanced 4-View TTA crops (prefer quality-gated frames)
                if hq:
                    views = extract_tta_face_views(
                        f.image_bgr,
                        primary.bbox,
                        landmarks_5=primary.landmarks_5,
                        target_size=t_sz,
                        base_scale=1.20 if is_vit else 1.30,
                        use_alignment=(not is_vit and not is_sbi),
                    )
                    if views:
                        enh_frame_views[m_name].append(views)

        # Fallback if strict quality gate filtered all frames in an extreme video
        for m_name, model in forensic_models.items():
            if not enh_frame_views[m_name] and base_crops[m_name]:
                t_sz = model.get_target_size() if hasattr(model, "get_target_size") else (224, 224)
                for c in base_crops[m_name]:
                    c_res = cv2.resize(c, t_sz) if (c.shape[1], c.shape[0]) != t_sz else c
                    enh_frame_views[m_name].append([c_res, np.ascontiguousarray(c_res[:, ::-1, :])])

        pair_sim, cent_sim = compute_temporal_identity_consistency(face_embeddings)

        # Evaluate Baseline vs Enhanced
        baseline_scores = {}
        enhanced_scores = {}
        for m_name, model in forensic_models.items():
            # Baseline: single crop + trimmed_mean
            b_crops = base_crops[m_name]
            if b_crops:
                b_probs = model.predict_crops(b_crops)
                baseline_scores[m_name] = round(float(model.aggregate_scores(b_probs, method="trimmed_mean")), 4)
            else:
                baseline_scores[m_name] = 0.0

            # Enhanced: multi-view TTA average per frame -> median across frames
            frame_view_groups = enh_frame_views[m_name]
            if frame_view_groups:
                flat_views = [v for grp in frame_view_groups for v in grp]
                flat_probs = model.predict_crops(flat_views)
                frame_means = []
                cursor = 0
                for grp in frame_view_groups:
                    n_v = len(grp)
                    frame_means.append(float(np.mean(flat_probs[cursor:cursor + n_v])))
                    cursor += n_v
                enhanced_scores[m_name] = round(float(model.aggregate_scores(frame_means, method="median")), 4)
            else:
                enhanced_scores[m_name] = 0.0

        base_verdict, base_ens = determine_constant_threshold_verdict(baseline_scores, args.threshold)
        enh_verdict, enh_ens = determine_constant_threshold_verdict(enhanced_scores, args.threshold)
        gt_val = gt_map.get(meta["video_name"], gt_map.get(meta["video_id"]))

        audit_results.append({
            "video_name": meta["video_name"],
            "platform": meta["platform"],
            "ground_truth": gt_val,
            "frames_evaluated": len(frames),
            "quality_pass_rate": round(high_q_count / max(1, len(all_confs)), 4),
            "mean_det_confidence": round(float(np.mean(all_confs)), 4) if all_confs else 0.0,
            "arcface_pairwise_sim": pair_sim,
            "baseline_scores": baseline_scores,
            "enhanced_scores": enhanced_scores,
            "baseline_verdict": base_verdict,
            "enhanced_verdict": enh_verdict,
            "representative_thumb": representative_thumb
        })
        print(f"[{v_idx+1}/{len(video_records)}] {meta['video_name']:30s} | HQ: {high_q_count}/{len(all_confs)} | "
              f"PE Base: {baseline_scores.get('GenD (PE-L)', 0):.4f} -> Enh: {enhanced_scores.get('GenD (PE-L)', 0):.4f}")

    # Print Table 1: Baseline Scores at Constant Threshold
    model_keys = list(forensic_models.keys())
    print("\n" + "="*128)
    print(f"         TABLE 1: BASELINE PIPELINE SCORES (THRESHOLD = {args.threshold:.2f})")
    print("="*128)
    hdr = f"{'Video Identifier':32s} | {'Platform':9s} | {'GT':4s} | " + " | ".join([f"{m[:12]:12s}" for m in model_keys]) + f" | {'Verdict @ '+str(args.threshold):18s}"
    print(hdr)
    print("-" * len(hdr))
    for r in audit_results:
        gt_str = "FAKE" if r["ground_truth"] == 1 else ("REAL" if r["ground_truth"] == 0 else "N/A")
        m_str = " | ".join([f"{r['baseline_scores'].get(m, 0.0):12.4f}" for m in model_keys])
        print(f"{r['video_name'][:32]:32s} | {r['platform'][:9]:9s} | {gt_str:4s} | {m_str} | {r['baseline_verdict']:18s}")
    print("="*128)

    # Print Table 2: Augmented TTA + Quality Gate Scores at Constant Threshold
    print("\n" + "="*128)
    print(f"         TABLE 2: AUGMENTED PIPELINE (4-VIEW TTA + QUALITY GATE + BORDER FIX) @ THRESHOLD = {args.threshold:.2f}")
    print("="*128)
    print(hdr)
    print("-" * len(hdr))
    for r in audit_results:
        gt_str = "FAKE" if r["ground_truth"] == 1 else ("REAL" if r["ground_truth"] == 0 else "N/A")
        m_str = " | ".join([f"{r['enhanced_scores'].get(m, 0.0):12.4f}" for m in model_keys])
        print(f"{r['video_name'][:32]:32s} | {r['platform'][:9]:9s} | {gt_str:4s} | {m_str} | {r['enhanced_verdict']:18s}")
    print("="*128)

    # Print Table 3 if ground truth provided
    print_metrics_comparison_table(audit_results, model_keys, args.threshold)

    # Save JSON and Dashboard
    json_path = os.path.join(args.output_dir, "social_media_audit_results.json")
    export_items = []
    for r in audit_results:
        d = dict(r)
        d.pop("representative_thumb", None)
        export_items.append(d)
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump({"threshold": args.threshold, "results": export_items}, jf, indent=2)

    dashboard_path = os.path.join(args.output_dir, "social_media_audit_dashboard.png")
    plot_social_media_audit_dashboard(audit_results, model_keys, args.threshold, dashboard_path)
    print(f"[+] Saved Audit JSON: {json_path}")
    print(f"[+] Saved Audit Dashboard: {dashboard_path}\n")


if __name__ == "__main__":
    main()
