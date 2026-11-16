"""
Outlier Isolation & Forensic Diagnosis Utility for ArcFace Actor Verification.
Isolates and visualizes anomalous low-similarity genuine comparisons (e.g. cosine sim ~0.22)
to diagnose whether the root cause is:
1. Multi-person interview scene where a non-target individual was evaluated by mistake.
2. Extreme head pose / profile turn where landmarks degraded alignment.
3. Solo cutaway / non-actor frame.
"""

import os
import sys
import argparse
import json
from typing import List, Dict, Any, Optional

# Ensure src directory is in sys.path
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


def estimate_yaw_from_landmarks(landmarks_5: np.ndarray) -> float:
    """Estimate approximate head yaw angle in degrees from 5 facial keypoints."""
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


def generate_diagnostic_card(
    full_frame_bgr: np.ndarray,
    detections: List[Any],
    evaluated_det_idx: int,
    evaluated_crop: np.ndarray,
    target_crop: Optional[np.ndarray],
    evaluated_sim: float,
    all_sims: List[float],
    meta: Any,
    frame_index: int,
    timestamp_sec: float,
    save_path: str
) -> Dict[str, Any]:
    """Render a comprehensive 3-panel supervisor diagnostic card for an isolated outlier frame."""
    if plt is None or cv2 is None:
        logger.warning("Matplotlib or OpenCV missing; skipping visual rendering.")
        return {}

    # Determine root cause
    num_faces = len(detections)
    actor_found_idx = -1
    best_actor_sim = evaluated_sim

    for i, s in enumerate(all_sims):
        if i != evaluated_det_idx and s > best_actor_sim:
            best_actor_sim = s
            if s >= 0.60:
                actor_found_idx = i

    if num_faces > 1 and actor_found_idx != -1:
        root_cause = "MULTI_PERSON_COHOST_MISMATCH"
        explanation = (
            f"Multi-person frame ({num_faces} faces). Naive Stage 1 selection evaluated Face #{evaluated_det_idx} "
            f"(Host/Guest, Cosine Sim: {evaluated_sim:.3f}). The true Target Actor was present as "
            f"Face #{actor_found_idx} (Cosine Sim: {best_actor_sim:.3f} >= 0.60). "
            f"Post-Stage-1 Identity Gating completely resolves this anomaly."
        )
    elif num_faces > 1:
        root_cause = "MULTI_PERSON_TARGET_ABSENT"
        explanation = (
            f"Multi-person frame ({num_faces} faces), but no detected face matched the target actor "
            f"(Max Sim: {best_actor_sim:.3f} < 0.60). Frame represents a cutaway or host monologue. "
            f"Post-Stage-1 Identity Gating isolates and filters out such non-target frames."
        )
    else:
        det = detections[evaluated_det_idx]
        yaw = estimate_yaw_from_landmarks(det.landmarks_5)
        if abs(yaw) > 35.0:
            root_cause = "EXTREME_HEAD_POSE"
            explanation = (
                f"Single face detected with extreme head yaw ({yaw:+.1f} deg). Non-frontal pose "
                f"degraded 5-point alignment and ArcFace representation (Sim: {evaluated_sim:.3f})."
            )
        else:
            root_cause = "FACIAL_DEGRADATION_OR_OCCLUSION"
            explanation = (
                f"Single face detected (Conf: {det.confidence:.2f}). Heavy motion blur, lighting variation, "
                f"or partial occlusion reduced cosine similarity to {evaluated_sim:.3f}."
            )

    # Prepare figure
    fig = plt.figure(figsize=(16, 7), dpi=150)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.22], width_ratios=[1.4, 0.8, 0.8], hspace=0.28, wspace=0.20)

    # Panel 1: Full Frame Context with Stage 1 Detections
    ax_frame = fig.add_subplot(gs[0, 0])
    canvas_bgr = full_frame_bgr.copy()

    for idx, d in enumerate(detections):
        x1, y1, x2, y2 = d.bbox.astype(int)
        is_eval = (idx == evaluated_det_idx)
        is_actor = (idx == actor_found_idx)

        # Color: Red for evaluated outlier face, Green for detected true actor, Yellow for others
        if is_eval:
            color = (0, 0, 255)  # Red BGR
            box_label = f"Face #{idx} (Evaluated): Sim={evaluated_sim:.3f}"
        elif is_actor:
            color = (0, 255, 0)  # Green BGR
            box_label = f"Face #{idx} (Actor): Sim={best_actor_sim:.3f}"
        else:
            color = (0, 255, 255)  # Yellow BGR
            sim_val = all_sims[idx] if idx < len(all_sims) else 0.0
            box_label = f"Face #{idx}: Sim={sim_val:.3f}"

        cv2.rectangle(canvas_bgr, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            canvas_bgr, box_label, (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA
        )

        if d.landmarks_5 is not None:
            for pt in d.landmarks_5:
                cv2.circle(canvas_bgr, (int(pt[0]), int(pt[1])), 4, (255, 255, 0), -1)

    canvas_rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)
    ax_frame.imshow(canvas_rgb)
    ax_frame.set_title(
        f"Stage 1 Full Frame ({num_faces} Face{'s' if num_faces > 1 else ''} Detected)\n"
        f"Video: {meta.category}/{meta.video_id} | Frame #{frame_index} (t={timestamp_sec:.2f}s)",
        fontsize=11, fontweight="bold", pad=8
    )
    ax_frame.axis("off")

    # Panel 2: Evaluated Face Crop
    ax_eval = fig.add_subplot(gs[0, 1])
    eval_rgb = cv2.cvtColor(evaluated_crop, cv2.COLOR_BGR2RGB) if evaluated_crop is not None else np.zeros((112, 112, 3), dtype=np.uint8)
    ax_eval.imshow(eval_rgb)
    ax_eval.set_title(
        f"Evaluated Face Crop (Face #{evaluated_det_idx})\n"
        f"Cosine Sim: {evaluated_sim:.4f} (OUTLIER)",
        fontsize=10, fontweight="bold", color="darkred", pad=8
    )
    for spine in ax_eval.spines.values():
        spine.set_edgecolor("red")
        spine.set_linewidth(3)
    ax_eval.set_xticks([])
    ax_eval.set_yticks([])

    # Panel 3: Target Actor Reference from Gallery
    ax_target = fig.add_subplot(gs[0, 2])
    if target_crop is not None:
        target_rgb = cv2.cvtColor(target_crop, cv2.COLOR_BGR2RGB)
    else:
        target_rgb = np.zeros((112, 112, 3), dtype=np.uint8)
    ax_target.imshow(target_rgb)
    ax_target.set_title(
        f"Enrolled Reference Actor\nActor ID: {meta.target_id}",
        fontsize=10, fontweight="bold", color="darkgreen", pad=8
    )
    for spine in ax_target.spines.values():
        spine.set_edgecolor("green")
        spine.set_linewidth(3)
    ax_target.set_xticks([])
    ax_target.set_yticks([])

    # Bottom Diagnosis Box
    ax_banner = fig.add_subplot(gs[1, :])
    ax_banner.axis("off")

    banner_color = "#FFF3CD" if root_cause == "MULTI_PERSON_COHOST_MISMATCH" else "#F8D7DA"
    border_color = "#856404" if root_cause == "MULTI_PERSON_COHOST_MISMATCH" else "#721C24"

    box = patches.FancyBboxPatch(
        (0.01, 0.05), 0.98, 0.90,
        boxstyle="round,pad=0.03",
        facecolor=banner_color,
        edgecolor=border_color,
        linewidth=2
    )
    ax_banner.add_patch(box)

    banner_text = (
        f"FORENSIC ROOT CAUSE: [{root_cause}]\n"
        f"{explanation}\n"
        f"ARCHITECTURAL RECOMMENDATION: Enable Stage 1.5 'Identity Gating' (match against target gallery prototype) "
        f"to guarantee the primary actor is isolated before ArcFace/GenD evaluation."
    )
    ax_banner.text(
        0.03, 0.50, banner_text,
        fontsize=9.5, fontweight="bold",
        color="#212529",
        verticalalignment="center",
        horizontalalignment="left"
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "video_id": meta.video_id,
        "category": meta.category,
        "frame_index": frame_index,
        "timestamp_sec": timestamp_sec,
        "num_faces": num_faces,
        "evaluated_sim": evaluated_sim,
        "best_actor_sim": best_actor_sim,
        "actor_found_idx": actor_found_idx,
        "root_cause": root_cause,
        "diagnostic_card_path": save_path
    }


def main():
    parser = argparse.ArgumentParser(description="Isolate and diagnose ArcFace verification outlier frames.")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory of dataset videos.")
    parser.add_argument("--output-dir", type=str, default="./results/outliers", help="Directory to save diagnosis cards.")
    parser.add_argument("--sim-threshold", type=float, default=0.50, help="Similarity threshold below which genuine pairs are flagged as outliers.")
    parser.add_argument("--max-frames", type=int, default=10, help="Frames sampled per video.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    loader = FFppDatasetLoader(args.data_dir)
    detector = SCRFDDetector(conf_threshold=0.50)
    encoder = ArcFaceEncoder()
    gallery = ActorReferenceGallery()

    # Enroll actors from pristine source videos
    originals = loader.filter(categories=["original", "Celeb-real", "YouTube-real"])
    if not originals:
        originals = [v for v in loader.videos if not v.is_fake] or loader.videos[:5]

    num_enroll = min(10, len(originals))
    logger.info(f"Enrolling {num_enroll} actors into reference gallery with face crops...")

    for meta in originals[:num_enroll]:
        reader = VideoReader(meta.video_path)
        frames = reader.sample_uniform(max_frames=args.max_frames)
        embs = []
        crops = []
        for f in frames:
            dets = detector.detect_frame(f.image_bgr, f.frame_index, f.timestamp_sec)
            if dets and dets[0].landmarks_5 is not None:
                aligned = align_face_5point(f.image_bgr, dets[0].landmarks_5)
                if aligned is not None:
                    emb = encoder.extract_embedding(aligned)
                    if emb is not None:
                        embs.append(emb)
                        crops.append(aligned)
        if embs:
            gallery.enroll_actor(meta.target_id, f"Actor_{meta.target_id}", embs[:5], crops=crops[:5])

    enrolled_ids = list(gallery.prototypes.keys())
    logger.info(f"Enrolled {len(enrolled_ids)} actors: {enrolled_ids}")

    outliers_found = []
    logger.info(f"Scanning {len(loader)} videos for genuine verification outliers (similarity < {args.sim_threshold})...")

    # Scan genuine videos (or authentic category videos)
    for meta in loader.videos:
        if meta.is_fake and meta.category not in ["Face2Face", "NeuralTextures"]:
            continue  # Focus on genuine comparisons where low similarity indicates an outlier anomaly

        target_proto = gallery.get_prototype(meta.target_id)
        if target_proto is None:
            continue

        target_crop = gallery.get_reference_crop(meta.target_id)
        reader = VideoReader(meta.video_path)
        frames = reader.sample_uniform(max_frames=max(4, args.max_frames // 2))

        for f in frames:
            dets = detector.detect_frame(f.image_bgr, f.frame_index, f.timestamp_sec)
            if not dets:
                continue

            # Compute similarities for all detected faces in this frame
            frame_sims = []
            frame_crops = []
            for d in dets:
                if d.landmarks_5 is not None:
                    aligned = align_face_5point(f.image_bgr, d.landmarks_5)
                    frame_crops.append(aligned)
                    if aligned is not None:
                        emb = encoder.extract_embedding(aligned)
                        if emb is not None:
                            frame_sims.append(float(np.dot(emb, target_proto)))
                        else:
                            frame_sims.append(0.0)
                    else:
                        frame_sims.append(0.0)
                else:
                    frame_crops.append(None)
                    frame_sims.append(0.0)

            # Evaluate Face #0 (naive selection that was performed in run_arcface.py)
            eval_sim = frame_sims[0] if frame_sims else 0.0
            if eval_sim < args.sim_threshold:
                # We have isolated the outlier!
                logger.warning(
                    f"OUTLIER DETECTED! Video: {meta.video_id} | Frame: #{f.frame_index} | "
                    f"Cosine Sim: {eval_sim:.4f} < {args.sim_threshold} | Faces in Frame: {len(dets)}"
                )
                card_filename = f"outlier_{meta.category}_{meta.video_id}_f{f.frame_index}.png"
                card_path = os.path.join(args.output_dir, card_filename)

                diag_info = generate_diagnostic_card(
                    full_frame_bgr=f.image_bgr,
                    detections=dets,
                    evaluated_det_idx=0,
                    evaluated_crop=frame_crops[0],
                    target_crop=target_crop,
                    evaluated_sim=eval_sim,
                    all_sims=frame_sims,
                    meta=meta,
                    frame_index=f.frame_index,
                    timestamp_sec=f.timestamp_sec,
                    save_path=card_path
                )
                outliers_found.append(diag_info)

    print("\n" + "="*75)
    print("      OUTLIER ISOLATION & FORENSIC DIAGNOSIS RESULTS")
    print("="*75)
    print(f"Total Outlier Frames Isolated: {len(outliers_found)}")
    print(f"Outlier Similarity Cutoff:     < {args.sim_threshold:.2f}")
    print("-" * 75)

    for idx, out in enumerate(outliers_found, 1):
        print(f"[{idx}] Video: {out['category']}/{out['video_id']} | Frame #{out['frame_index']} (t={out['timestamp_sec']:.2f}s)")
        print(f"    - Faces Detected: {out['num_faces']}")
        print(f"    - Evaluated Sim:  {out['evaluated_sim']:.4f} (OUTLIER)")
        if out['actor_found_idx'] != -1:
            print(f"    - True Actor Sim: {out['best_actor_sim']:.4f} (Found in Face #{out['actor_found_idx']})")
        print(f"    - Root Cause:     {out['root_cause']}")
        print(f"    - Diagnostic Card:{out['diagnostic_card_path']}")
        print("-" * 75)

    # Save summary JSON
    summary_path = os.path.join(args.output_dir, "outliers_summary.json")
    with open(summary_path, "w") as f:
        json.dump(outliers_found, f, indent=2)
    print(f"[+] Diagnostic manifest saved to: {summary_path}")
    print("="*75 + "\n")


if __name__ == "__main__":
    main()
