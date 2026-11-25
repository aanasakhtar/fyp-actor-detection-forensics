"""
In-the-Wild Social Media Forensic Audit Script.

Evaluates arbitrary user-downloaded social media videos (YouTube, Instagram, Facebook, TikTok)
across the full 3-Stage Pipeline:
  Stage 1: SCRFD Face Detection & 5-point landmark extraction.
  Stage 2: ArcFace Biometrics & Temporal Identity Consistency analysis.
  Stage 3: Multi-Model Deepfake Forensics: GenD (PE-L), GenD (CLIP-L), and ViT.

Outputs:
  1. Console Audit Table with per-model scores and consensus verdicts.
  2. Multi-panel Executive Diagnostic Dashboard (social_media_audit_dashboard.png).
  3. Machine-readable JSON summary (social_media_audit_results.json).
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
from pipeline.forensics.clip_detector import GenDCLIPDetector
from pipeline.forensics.pe_detector import GenDPEDetector
from pipeline.forensics.sbi_detector import SBIEfficientNetDetector
from pipeline.utils.logging import get_logger

logger = get_logger("audit_social_media")

# Calibrated decision thresholds (Youden's J optimal cutoffs)
CALIBRATED_THRESHOLDS = {
    "ViT": 0.612,
    "GenD (CLIP-L)": 0.741,
    "GenD (PE-L)": 0.782,
    "EfficientNet-B4 (SBI)": 0.285
}


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
    stem, ext = os.path.splitext(base)
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
        platform = "In-the-Wild Social"

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

    # If still empty, search /kaggle/input
    if not found_files and os.path.exists("/kaggle/input"):
        logger.warning(f"Searching /kaggle/input for social media video files...")
        for ext in valid_extensions:
            found_files.extend(glob.glob(f"/kaggle/input/**/{ext}", recursive=True))

    found_files = sorted(list(set(found_files)))
    if not found_files:
        logger.error(f"No video files found in '{input_path}' or fallback search locations.")
        return []

    logger.info(f"Discovered {len(found_files)} social media videos for forensic auditing from: '{target_dir or input_path}'")
    return [parse_video_metadata(f) for f in found_files]


def compute_temporal_identity_consistency(embeddings: List[np.ndarray]) -> Tuple[float, float]:
    """
    Compute pairwise consecutive ArcFace cosine similarity and centroid similarity
    to verify identity stability across video frames.
    """
    if not embeddings or len(embeddings) < 2:
        return 1.0, 1.0

    # Consecutive pairwise cosine similarity
    consec_sims = []
    for i in range(len(embeddings) - 1):
        e1, e2 = embeddings[i], embeddings[i + 1]
        dot = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8))
        consec_sims.append(dot)

    # Centroid consistency
    centroid = np.mean(embeddings, axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
    centroid_sims = [float(np.dot(e, centroid) / (np.linalg.norm(e) + 1e-8)) for e in embeddings]

    return round(float(np.mean(consec_sims)), 4), round(float(np.mean(centroid_sims)), 4)


def determine_consensus_verdict(scores_dict: Dict[str, float]) -> Tuple[str, str, str]:
    """
    Determine forensic verdict combining naive 0.50 baseline and calibrated thresholds.
    Returns: (verdict_label, severity_color, reasoning)
    """
    pe_score = scores_dict.get("GenD (PE-L)", 0.0)
    clip_score = scores_dict.get("GenD (CLIP-L)", 0.0)
    vit_score = scores_dict.get("ViT", 0.0)

    pe_fake = pe_score >= CALIBRATED_THRESHOLDS.get("GenD (PE-L)", 0.782)
    clip_fake = clip_score >= CALIBRATED_THRESHOLDS.get("GenD (CLIP-L)", 0.741)
    vit_fake = vit_score >= CALIBRATED_THRESHOLDS.get("ViT", 0.612)

    fake_votes = sum([pe_fake, clip_fake, vit_fake])

    if pe_fake and clip_fake:
        verdict = "UNAUTHORIZED DEEPFAKE"
        color = "#D32F2F"  # Red
        reason = f"High-confidence dual GenD flag (PE: {pe_score:.2f}, CLIP: {clip_score:.2f})"
    elif pe_fake or clip_fake:
        verdict = "SUSPECTED DEEPFAKE"
        color = "#F57C00"  # Orange
        reason = f"Single GenD model threshold breach (PE: {pe_score:.2f}, CLIP: {clip_score:.2f})"
    elif vit_fake and not (pe_fake or clip_fake):
        verdict = "COMPRESSION NOISE / UNCERTAIN"
        color = "#FBC02D"  # Yellow
        reason = f"ViT triggered on H.264 social compression (ViT: {vit_score:.2f}), GenD cleared."
    else:
        verdict = "VERIFIED AUTHENTIC"
        color = "#388E3C"  # Green
        reason = f"All models cleared below calibrated threshold (PE: {pe_score:.2f}, CLIP: {clip_score:.2f})"

    return verdict, color, reason


def plot_social_media_audit_dashboard(
    audit_results: List[Dict[str, Any]],
    model_names: List[str],
    save_path: str
):
    """Generate 4-panel executive social media forensic dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), dpi=150)
    video_labels = [r["video_name"][:18] for r in audit_results]
    n_vids = len(audit_results)
    x = np.arange(n_vids)

    # 1. Panel 1: Multi-Model Manipulation Scores per Video
    ax1 = axes[0, 0]
    w = 0.8 / max(1, len(model_names))
    model_colors = {
        "GenD (PE-L)": "#D32F2F",
        "GenD (CLIP-L)": "#388E3C",
        "ViT": "#1976D2",
        "EfficientNet-B4 (SBI)": "#7B1FA2"
    }

    for i, m_name in enumerate(model_names):
        scores = [r["scores"].get(m_name, 0.0) for r in audit_results]
        col = model_colors.get(m_name, "#546E7A")
        rects = ax1.bar(x + (i - len(model_names)/2 + 0.5) * w, scores, w, label=m_name, color=col, alpha=0.88)
        for r in rects:
            h = r.get_height()
            ax1.annotate(f"{h:.2f}", xy=(r.get_x() + r.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7.5)

    ax1.axhline(0.50, color="black", linestyle=":", lw=1.5, label="Default Threshold (0.50)")
    ax1.axhline(0.78, color="#D32F2F", linestyle="--", lw=1.8, label="Calibrated PE-L T* (0.78)")
    ax1.axhline(0.74, color="#388E3C", linestyle="--", lw=1.8, label="Calibrated CLIP-L T* (0.74)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(video_labels, rotation=20, ha="right", fontsize=9, fontweight="bold")
    ax1.set_ylim([0.0, 1.15])
    ax1.set_ylabel("Manipulation Score p(Fake)", fontweight="bold")
    ax1.set_title("STAGE 3: MULTI-MODEL FORENSIC MANIPULATION SCORES", fontsize=11, fontweight="bold", pad=10)
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=8.5)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # 2. Panel 2: Temporal Identity Consistency (ArcFace)
    ax2 = axes[0, 1]
    pairwise_sims = [r["arcface_pairwise_sim"] for r in audit_results]
    centroid_sims = [r["arcface_centroid_sim"] for r in audit_results]

    w2 = 0.35
    ax2.bar(x - w2/2, pairwise_sims, w2, label="Consecutive Pairwise Sim", color="#009688", alpha=0.85)
    ax2.bar(x + w2/2, centroid_sims, w2, label="Centroid Profile Sim", color="#3F51B5", alpha=0.85)
    ax2.axhline(0.55, color="red", linestyle="--", lw=1.5, label="Marketplace Identity Threshold (0.55)")

    for xi, (ps, cs) in enumerate(zip(pairwise_sims, centroid_sims)):
        ax2.text(xi - w2/2, ps + 0.02, f"{ps:.2f}", ha="center", fontsize=8, fontweight="bold")
        ax2.text(xi + w2/2, cs + 0.02, f"{cs:.2f}", ha="center", fontsize=8, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(video_labels, rotation=20, ha="right", fontsize=9, fontweight="bold")
    ax2.set_ylim([0.0, 1.15])
    ax2.set_ylabel("Cosine Similarity (0.0 - 1.0)", fontweight="bold")
    ax2.set_title("STAGE 2: ARCFACE BIOMETRIC TEMPORAL IDENTITY STABILITY", fontsize=11, fontweight="bold", pad=10)
    ax2.legend(loc="lower right", framealpha=0.9, fontsize=8.5)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    # 3. Panel 3: Representative Thumbnails & Overlays
    ax3 = axes[1, 0]
    ax3.axis("off")
    ax3.set_title("STAGE 1 & 2: IN-THE-WILD VISUAL INSPECTION SAMPLES", fontsize=11, fontweight="bold", pad=10)

    # Build a small multi-thumbnail composite inside panel 3
    n_display = min(4, n_vids)
    if n_display > 0:
        sub_grid = np.zeros((240, 240 * n_display, 3), dtype=np.uint8)
        for i in range(n_display):
            thumb = audit_results[i].get("representative_thumb")
            if thumb is not None:
                h_t, w_t = thumb.shape[:2]
                resized = cv2.resize(thumb, (240, 240))
                # Add status border
                col_bgr = (0, 0, 255) if "DEEPFAKE" in audit_results[i]["verdict"] else (0, 255, 0)
                cv2.rectangle(resized, (0, 0), (239, 239), col_bgr, 6)
                sub_grid[:, i*240:(i+1)*240] = resized

        ax3.imshow(cv2.cvtColor(sub_grid, cv2.COLOR_BGR2RGB))
        for i in range(n_display):
            v_name = audit_results[i]["video_name"][:14]
            verd = audit_results[i]["verdict"]
            ax3.text(i*240 + 120, 255, f"{v_name}\n{verd}", ha="center", va="top", fontsize=8, fontweight="bold")

    # 4. Panel 4: Platform Breakdown & Consensus Verdict Summary
    ax4 = axes[1, 1]
    verdict_counts = {}
    for r in audit_results:
        v = r["verdict"]
        verdict_counts[v] = verdict_counts.get(v, 0) + 1

    v_labels = list(verdict_counts.keys())
    v_vals = list(verdict_counts.values())
    v_colors = ["#D32F2F" if "DEEPFAKE" in vl else ("#388E3C" if "AUTHENTIC" in vl else "#F57C00") for vl in v_labels]

    wedges, texts, autotexts = ax4.pie(
        v_vals, labels=v_labels, autopct="%1.1f%%", colors=v_colors,
        startangle=140, textprops=dict(fontweight="bold", fontsize=9)
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
    ax4.set_title("CONSENSUS AUDIT VERDICT BREAKDOWN", fontsize=11, fontweight="bold", pad=10)

    plt.suptitle("IN-THE-WILD SOCIAL MEDIA FORENSIC AUDIT (FB, INSTA, YT)", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved Executive Audit Dashboard to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Audit arbitrary social media videos (FB, Insta, YT) through multi-stage forensics.")
    parser.add_argument("--input-dir", type=str, default="video_storage", help="Path to directory containing input videos or single video file.")
    parser.add_argument("--models", nargs="+", default=["pe", "clip", "vit"], help="Forensic models to benchmark (pe, clip, vit).")
    parser.add_argument("--max-frames", type=int, default=16, help="Uniform frames to sample per video.")
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/results/social_media_audit", help="Output directory for reports.")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face Token for authenticated model loading.")
    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*96)
    print("      IN-THE-WILD SOCIAL MEDIA FORENSIC AUDIT PIPELINE (FB, INSTAGRAM, YOUTUBE)")
    print("="*96)

    # 1. Discover Videos
    video_records = discover_input_videos(args.input_dir)
    if not video_records:
        return

    print(f"[+] Loaded {len(video_records)} social media videos for evaluation:")
    for v in video_records:
        print(f"    * [{v['platform']:10s}] {v['video_name']:36s} ({v['file_size_mb']:.2f} MB)")

    # 2. Initialize Engines
    print("\nInitializing Pipeline Stages...")
    detector = SCRFDDetector(conf_threshold=0.50)
    encoder = ArcFaceEncoder()
    forensic_models = init_forensic_models(args.models)

    if not forensic_models:
        logger.error("No forensic models loaded. Exiting.")
        return

    audit_results = []
    print("\n" + "-"*96)
    print("EXECUTING MULTI-STAGE INFERENCE ACROSS VIDEOS...")
    print("-" * 96)

    for v_idx, meta in enumerate(video_records):
        v_path = meta["video_path"]
        print(f"\n[{v_idx+1}/{len(video_records)}] Auditing: {meta['video_name']} ({meta['platform']})...")

        reader = VideoReader(v_path)
        frames = reader.sample_uniform(max_frames=args.max_frames)
        if not frames:
            logger.warning(f"Could not read video frames from '{v_path}'. Skipping.")
            continue

        vid_crops_by_model = {m: [] for m in forensic_models}
        frame_confidences = []
        face_embeddings = []
        representative_thumb = None

        # Stage 1: SCRFD & Stage 2: ArcFace
        t0 = time.time()
        for f in frames:
            dets = detector.detect_frame(f.image_bgr, f.frame_index, f.timestamp_sec)
            if dets:
                primary = max(dets, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
                frame_confidences.append(float(primary.confidence))

                # ArcFace embedding
                if primary.landmarks_5 is not None:
                    aligned = align_face_5point(f.image_bgr, primary.landmarks_5)
                    if aligned is not None:
                        emb = encoder.extract_embedding(aligned)
                        if emb is not None:
                            face_embeddings.append(emb)

                # Prepare thumbnail
                if representative_thumb is None:
                    thumb_copy = f.image_bgr.copy()
                    x1, y1, x2, y2 = [int(v) for v in primary.bbox]
                    cv2.rectangle(thumb_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    if primary.landmarks_5 is not None:
                        for lx, ly in primary.landmarks_5:
                            cv2.circle(thumb_copy, (int(lx), int(ly)), 3, (0, 0, 255), -1)
                    representative_thumb = thumb_copy

                # Forensic crops
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

        det_arc_time = time.time() - t0

        # Stage 2 temporal metrics
        pair_sim, cent_sim = compute_temporal_identity_consistency(face_embeddings)

        # Stage 3: Inference
        scores = {}
        for m_name, model in forensic_models.items():
            crops = vid_crops_by_model[m_name]
            if crops:
                f_scores = model.predict_crops(crops)
                scores[m_name] = round(float(model.aggregate_scores(f_scores)), 4)
            else:
                scores[m_name] = 0.0

        # Consensus verdict
        verdict, color, reason = determine_consensus_verdict(scores)

        audit_results.append({
            "video_name": meta["video_name"],
            "platform": meta["platform"],
            "frames_evaluated": len(frames),
            "faces_detected": len(frame_confidences),
            "mean_det_confidence": round(float(np.mean(frame_confidences)), 4) if frame_confidences else 0.0,
            "arcface_pairwise_sim": pair_sim,
            "arcface_centroid_sim": cent_sim,
            "scores": scores,
            "verdict": verdict,
            "verdict_reason": reason,
            "representative_thumb": representative_thumb
        })

        scores_str = " | ".join([f"{k}: {v:.4f}" for k, v in scores.items()])
        print(f"    -> Identity Sim: {pair_sim:.4f} | {scores_str} | Verdict: [{verdict}]")

    # 4. Print Executive Table
    print("\n" + "="*124)
    print("                TABLE 1: IN-THE-WILD SOCIAL MEDIA MULTI-MODEL FORENSIC AUDIT")
    print("="*124)
    header = f"{'Video Identifier':32s} | {'Platform':10s} | {'Det p':6s} | {'Id Sim':6s} | " + \
             " | ".join([f"{m[:12]:12s}" for m in forensic_models]) + " | " + f"{'Consensus Verdict':24s}"
    print(header)
    print("-" * len(header))

    for r in audit_results:
        m_str = " | ".join([f"{r['scores'].get(m, 0.0):12.4f}" for m in forensic_models])
        row = f"{r['video_name'][:32]:32s} | {r['platform'][:10]:10s} | {r['mean_det_confidence']:6.2f} | {r['arcface_pairwise_sim']:6.2f} | {m_str} | {r['verdict']:24s}"
        print(row)

    print("="*124 + "\n")

    # 5. Save JSON Export
    json_path = os.path.join(args.output_dir, "social_media_audit_results.json")
    export_data = []
    for r in audit_results:
        item = dict(r)
        item.pop("representative_thumb", None)  # Not JSON serializable
        export_data.append(item)

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump({
            "videos_audited": len(export_data),
            "models_evaluated": list(forensic_models.keys()),
            "calibrated_thresholds": CALIBRATED_THRESHOLDS,
            "results": export_data
        }, jf, indent=2)

    print(f"[+] Saved Machine-Readable Audit JSON to: {json_path}")

    # 6. Generate Executive Report Card
    dashboard_path = os.path.join(args.output_dir, "social_media_audit_dashboard.png")
    plot_social_media_audit_dashboard(
        audit_results=audit_results,
        model_names=list(forensic_models.keys()),
        save_path=dashboard_path
    )
    print(f"[+] Saved Executive Audit Dashboard to: {dashboard_path}\n")


if __name__ == "__main__":
    main()
