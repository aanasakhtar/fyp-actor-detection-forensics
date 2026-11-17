"""
Standalone Outlier Isolation and Forensic Diagnosis Utility for ArcFace on FF++ / Celeb-DF.

Locates genuine comparisons scoring below a threshold (e.g. cosine sim < 0.50),
extracts the exact frame, draws bounding boxes, aligns the face crop, and
saves side-by-side diagnostic figures for supervisor analysis.
"""

import os
import sys
import argparse
import json
from typing import List, Dict, Any, Optional

# Ensure local 'src' directory takes priority in sys.path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
except ImportError:
    matplotlib = None
    plt = None

from pipeline.data.dataset_loader import FFppDatasetLoader
from pipeline.data.video_reader import VideoReader
from pipeline.detection.scrfd_detector import SCRFDDetector
from pipeline.preprocessing.face_alignment import align_face_5point
from pipeline.recognition.arcface_encoder import ArcFaceEncoder
from pipeline.recognition.reference_gallery import ActorReferenceGallery
from pipeline.utils.logging import get_logger

logger = get_logger("isolate_outliers")


def estimate_pose_yaw(landmarks_5: np.ndarray) -> float:
    """Approximate head yaw angle (degrees) from 5-point landmarks."""
    if landmarks_5 is None or len(landmarks_5) < 5:
        return 0.0
    left_eye = landmarks_5[0]
    right_eye = landmarks_5[1]
    nose = landmarks_5[2]
    eye_vec = right_eye - left_eye
    eye_dist = np.linalg.norm(eye_vec)
    if eye_dist < 1e-4:
        return 0.0
    unit_eye = eye_vec / eye_dist
    proj_nose = np.dot(nose - left_eye, unit_eye)
    ratio = proj_nose / eye_dist
    return float((ratio - 0.5) * 100.0)


def create_outlier_card(
    frame_bgr: np.ndarray,
    detections: List[Any],
    evaluated_crop: np.ndarray,
    reference_crop: Optional[np.ndarray],
    similarity: float,
    meta: Any,
    frame_index: int,
    timestamp_sec: float,
    all_sims: List[float],
    save_path: str
) -> Dict[str, Any]:
    """Render a 3-panel supervisor diagnostic image showing the outlier frame and root cause."""
    if plt is None or cv2 is None:
        logger.warning("Matplotlib or OpenCV missing; skipping card rendering.")
        return {}

    num_faces = len(detections)
    yaw = estimate_pose_yaw(detections[0].landmarks_5) if detections and detections[0].landmarks_5 is not None else 0.0

    # Determine root cause diagnosis
    actor_alt_idx = -1
    best_alt_sim = -1.0
    for i, s in enumerate(all_sims):
        if i != 0 and s > best_alt_sim:
            best_alt_sim = s
            if s >= 0.60:
                actor_alt_idx = i

    if num_faces > 1 and actor_alt_idx != -1:
        category_tag = "MULTI_PERSON_HOST_INTERVIEW"
        diag_msg = (
            f"Multi-person scene ({num_faces} faces). Naive selection evaluated Face #0 (Cosine Sim: {similarity:.3f}). "
            f"Enrolled Target Actor was present as Face #{actor_alt_idx} (Cosine Sim: {best_alt_sim:.3f})."
        )
    elif abs(yaw) > 35.0:
        category_tag = "EXTREME_HEAD_PROFILE"
        diag_msg = (
            f"Single face detected with severe horizontal head turn (Yaw: {yaw:+.1f}°). "
            f"Profile angle degraded landmark alignment and ArcFace representation (Cosine Sim: {similarity:.3f})."
        )
    elif meta.is_fake:
        category_tag = "MANIPULATION_ARTIFACT_DEGRADATION"
        diag_msg = (
            f"Manipulated video ({meta.category}). Heavy facial reconstruction artifacts or expression "
            f"distortion significantly altered biometric features away from the pristine prototype (Sim: {similarity:.3f})."
        )
    else:
        category_tag = "OCCLUSION_OR_MOTION_BLUR"
        diag_msg = (
            f"Single face detected (Conf: {detections[0].confidence:.2f}). Motion blur, lighting, or partial "
            f"facial occlusion reduced cosine similarity to {similarity:.3f}."
        )

    # Plot
    fig = plt.figure(figsize=(15, 6.5), dpi=150)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.25], width_ratios=[1.3, 0.85, 0.85], hspace=0.28, wspace=0.20)

    # Panel 1: Full Frame with Detections
    ax_frame = fig.add_subplot(gs[0, 0])
    canvas = frame_bgr.copy()
    for idx, d in enumerate(detections):
        x1, y1, x2, y2 = d.bbox.astype(int)
        is_eval = (idx == 0)
        box_color = (0, 0, 255) if is_eval else (0, 255, 0)
        lbl = f"Face #{idx} (Evaluated: Sim={similarity:.3f})" if is_eval else f"Face #{idx}: Sim={all_sims[idx]:.3f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 3)
        cv2.putText(canvas, lbl, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2, cv2.LINE_AA)
        if d.landmarks_5 is not None:
            for pt in d.landmarks_5:
                cv2.circle(canvas, (int(pt[0]), int(pt[1])), 4, (255, 255, 0), -1)

    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    ax_frame.imshow(canvas_rgb)
    ax_frame.set_title(
        f"Stage 1 Full Frame ({num_faces} Face{'s' if num_faces > 1 else ''})\n"
        f"Video: {meta.category}/{meta.video_id} | Frame #{frame_index} (t={timestamp_sec:.2f}s)",
        fontsize=10.5, fontweight="bold", pad=8
    )
    ax_frame.axis("off")

    # Panel 2: Evaluated Face Crop
    ax_crop = fig.add_subplot(gs[0, 1])
    crop_rgb = cv2.cvtColor(evaluated_crop, cv2.COLOR_BGR2RGB) if evaluated_crop is not None else np.zeros((112, 112, 3), dtype=np.uint8)
    ax_crop.imshow(crop_rgb)
    ax_crop.set_title(
        f"Evaluated Face Crop (ArcFace Input)\nCosine Similarity: {similarity:.4f} [OUTLIER]",
        fontsize=10, fontweight="bold", color="darkred", pad=8
    )
    for s in ax_crop.spines.values():
        s.set_edgecolor("red")
        s.set_linewidth(3)
    ax_crop.set_xticks([])
    ax_crop.set_yticks([])

    # Panel 3: Enrolled Reference Actor Crop
    ax_ref = fig.add_subplot(gs[0, 2])
    ref_rgb = cv2.cvtColor(reference_crop, cv2.COLOR_BGR2RGB) if reference_crop is not None else np.zeros((112, 112, 3), dtype=np.uint8)
    ax_ref.imshow(ref_rgb)
    ax_ref.set_title(
        f"Enrolled Gallery Prototype\nTarget Actor ID: {meta.target_id}",
        fontsize=10, fontweight="bold", color="darkgreen", pad=8
    )
    for s in ax_ref.spines.values():
        s.set_edgecolor("green")
        s.set_linewidth(3)
    ax_ref.set_xticks([])
    ax_ref.set_yticks([])

    # Bottom Diagnosis Box
    ax_box = fig.add_subplot(gs[1, :])
    ax_box.axis("off")
    patch = patches.FancyBboxPatch(
        (0.01, 0.05), 0.98, 0.90,
        boxstyle="round,pad=0.03",
        facecolor="#FFF3CD",
        edgecolor="#856404",
        linewidth=2
    )
    ax_box.add_patch(patch)
    banner_txt = (
        f"FORENSIC ROOT CAUSE: [{category_tag}]\n"
        f"{diag_msg}\n"
        f"PIPELINE INSIGHT: In genuine verification, this sample represents a False Non-Match (False Rejection). "
        f"Post-Stage-1 filtering or identity gating prevents such degraded/secondary faces from polluting genuine metrics."
    )
    ax_box.text(0.02, 0.50, banner_txt, fontsize=9.5, fontweight="bold", color="#333333", verticalalignment="center")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "video_id": meta.video_id,
        "category": meta.category,
        "target_id": meta.target_id,
        "frame_index": frame_index,
        "timestamp_sec": timestamp_sec,
        "similarity": similarity,
        "num_faces": num_faces,
        "root_cause": category_tag,
        "diagnostic_card": save_path
    }


def main():
    parser = argparse.ArgumentParser(description="Isolate ArcFace verification outlier frames in FF++ or Celeb-DF.")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory containing dataset videos.")
    parser.add_argument("--output-dir", type=str, default="./results/arcface/outliers", help="Output directory for outlier cards.")
    parser.add_argument("--threshold", type=float, default=0.50, help="Similarity threshold below which genuine pairs are flagged as outliers.")
    parser.add_argument("--max-frames", type=int, default=10, help="Uniform frames sampled per video.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    loader = FFppDatasetLoader(args.data_dir)
    detector = SCRFDDetector(conf_threshold=0.50)
    encoder = ArcFaceEncoder()
    gallery = ActorReferenceGallery()

    originals = loader.filter(categories=["original", "Celeb-real", "YouTube-real"])
    if not originals:
        originals = [v for v in loader.videos if not v.is_fake] or loader.videos[:5]

    num_enroll = min(10, len(originals))
    logger.info(f"Enrolling {num_enroll} actors into reference gallery from pristine source videos...")

    reference_crops: Dict[str, np.ndarray] = {}

    for meta in originals[:num_enroll]:
        reader = VideoReader(meta.video_path)
        frames = reader.sample_uniform(max_frames=args.max_frames)
        embs = []
        for f in frames:
            dets = detector.detect_frame(f.image_bgr, f.frame_index, f.timestamp_sec)
            if dets and dets[0].landmarks_5 is not None:
                aligned = align_face_5point(f.image_bgr, dets[0].landmarks_5)
                if aligned is not None:
                    emb = encoder.extract_embedding(aligned)
                    if emb is not None:
                        embs.append(emb)
                        if meta.target_id not in reference_crops:
                            reference_crops[meta.target_id] = aligned
        if embs:
            gallery.enroll_actor(meta.target_id, f"Actor_{meta.target_id}", embs[:5])

    enrolled_ids = list(gallery.prototypes.keys())
    logger.info(f"Successfully enrolled {len(enrolled_ids)} actors: {enrolled_ids}")

    outliers = []
    logger.info(f"Scanning {len(loader)} videos for genuine verification outliers (similarity < {args.threshold})...")

    for meta in loader.videos:
        # Genuine comparisons are original videos or morphology-preserving reenactments
        if meta.is_fake and meta.category not in ["Face2Face", "NeuralTextures"]:
            continue

        target_proto = gallery.get_prototype(meta.target_id)
        if target_proto is None:
            continue

        reader = VideoReader(meta.video_path)
        frames = reader.sample_uniform(max_frames=max(4, args.max_frames // 2))

        for f in frames:
            dets = detector.detect_frame(f.image_bgr, f.frame_index, f.timestamp_sec)
            if not dets or dets[0].landmarks_5 is None:
                continue

            # Align Face #0 (naive choice)
            aligned_0 = align_face_5point(f.image_bgr, dets[0].landmarks_5)
            if aligned_0 is None:
                continue
            emb_0 = encoder.extract_embedding(aligned_0)
            if emb_0 is None:
                continue

            sim_0 = float(np.dot(emb_0, target_proto))

            if sim_0 < args.threshold:
                # Outlier identified! Check all faces in this frame
                all_sims = [sim_0]
                for d in dets[1:]:
                    if d.landmarks_5 is not None:
                        al = align_face_5point(f.image_bgr, d.landmarks_5)
                        if al is not None:
                            e = encoder.extract_embedding(al)
                            if e is not None:
                                all_sims.append(float(np.dot(e, target_proto)))
                                continue
                    all_sims.append(0.0)

                card_filename = f"outlier_{meta.category}_{meta.video_id}_frame_{f.frame_index}.png"
                card_path = os.path.join(args.output_dir, card_filename)

                diag_info = create_outlier_card(
                    frame_bgr=f.image_bgr,
                    detections=dets,
                    evaluated_crop=aligned_0,
                    reference_crop=reference_crops.get(meta.target_id),
                    similarity=sim_0,
                    meta=meta,
                    frame_index=f.frame_index,
                    timestamp_sec=f.timestamp_sec,
                    all_sims=all_sims,
                    save_path=card_path
                )
                outliers.append(diag_info)
                logger.warning(
                    f"ISOLATED OUTLIER: Video {meta.category}/{meta.video_id} | Frame #{f.frame_index} "
                    f"| Cosine Sim = {sim_0:.4f} < {args.threshold} | Card: {card_path}"
                )

    print("\n" + "="*75)
    print("      OUTLIER ISOLATION & FORENSIC DIAGNOSIS SUMMARY")
    print("="*75)
    print(f"Total Outlier Frames Isolated: {len(outliers)}")
    print(f"Outlier Similarity Threshold:  < {args.threshold:.2f}")
    print("-" * 75)

    for idx, o in enumerate(outliers, 1):
        print(f"[{idx}] Video: {o['category']}/{o['video_id']} | Frame #{o['frame_index']} (t={o['timestamp_sec']:.2f}s)")
        print(f"    - Target ID:     {o['target_id']}")
        print(f"    - Cosine Sim:    {o['similarity']:.4f} [OUTLIER]")
        print(f"    - Faces in Scene:{o['num_faces']}")
        print(f"    - Root Cause:    {o['root_cause']}")
        print(f"    - Diagnostic Card: {o['diagnostic_card']}")
        print("-" * 75)

    manifest_path = os.path.join(args.output_dir, "outliers_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(outliers, f, indent=2)
    print(f"[+] Outlier manifest saved to: {manifest_path}")
    print("="*75 + "\n")


if __name__ == "__main__":
    main()
