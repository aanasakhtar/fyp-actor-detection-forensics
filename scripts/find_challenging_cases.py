"""
Challenging Edge Case Mining & Visualization Utility for SCRFD.
Discovers and visualizes:
1. Multi-person frames (multiple faces detected simultaneously, verifying tracking).
2. Side-profile / yaw-turned faces (non-frontal facial angles).
3. Tilted / rotated heads (extreme roll angle).
4. Distant / small background faces.
"""

import os
import sys
import argparse
import json
from typing import List, Dict, Any, Tuple

# Ensure src directory is on sys.path
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
except ImportError:
    matplotlib = None
    plt = None

from pipeline.data.dataset_loader import FFppDatasetLoader
from pipeline.data.video_reader import VideoReader
from pipeline.detection.scrfd_detector import SCRFDDetector
from pipeline.tracking.face_tracking import IoUFaceTracker
from pipeline.visualization.detection_visualization import draw_detections
from pipeline.utils.logging import get_logger

logger = get_logger("find_challenging_cases")


def compute_pose_metrics(landmarks_5: np.ndarray) -> Dict[str, float]:
    """
    Estimate head pose metrics from 5 facial keypoints:
    landmarks_5: shape (5, 2)
      0: left eye, 1: right eye, 2: nose, 3: left mouth, 4: right mouth
    """
    if landmarks_5 is None or len(landmarks_5) < 5:
        return {"yaw_asymmetry": 0.0, "approx_yaw_deg": 0.0, "roll_deg": 0.0}

    left_eye = landmarks_5[0]
    right_eye = landmarks_5[1]
    nose = landmarks_5[2]

    # Roll (in-plane tilt)
    delta_y = right_eye[1] - left_eye[1]
    delta_x = right_eye[0] - left_eye[0]
    roll_deg = float(np.degrees(np.arctan2(delta_y, delta_x)))

    # Yaw (out-of-plane horizontal turn)
    eye_vec = right_eye - left_eye
    eye_dist = np.linalg.norm(eye_vec)
    if eye_dist < 1e-4:
        return {"yaw_asymmetry": 0.0, "approx_yaw_deg": 0.0, "roll_deg": roll_deg}

    unit_eye = eye_vec / eye_dist
    proj_nose = np.dot(nose - left_eye, unit_eye)
    ratio = proj_nose / eye_dist  # 0.5 for frontal, <0.25 or >0.75 for side profile

    yaw_asymmetry = float(abs(ratio - 0.5) * 2.0)  # [0, 1]
    approx_yaw_deg = float((ratio - 0.5) * 100.0)   # Approx degrees: -50° to +50°

    return {
        "yaw_asymmetry": yaw_asymmetry,
        "approx_yaw_deg": approx_yaw_deg,
        "roll_deg": roll_deg
    }


def analyze_video_cases(
    meta, 
    detector: SCRFDDetector, 
    tracker: IoUFaceTracker, 
    max_frames: int = 32
) -> List[Dict[str, Any]]:
    """Scan a video for multi-person and profile frames."""
    cases = []
    try:
        reader = VideoReader(meta.video_path)
        frames = reader.sample_uniform(max_frames=max_frames)
        
        for frame in frames:
            dets = detector.detect_frame(frame.image_bgr, frame.frame_index, frame.timestamp_sec)
            dets = tracker.update(dets, frame.frame_index)
            if not dets:
                continue

            num_faces = len(dets)
            h_img, w_img = frame.image_bgr.shape[:2]
            img_area = float(h_img * w_img)

            # Check 1: Multi-person frame
            if num_faces >= 2:
                cases.append({
                    "type": "multi_person",
                    "video_id": meta.video_id,
                    "category": meta.category,
                    "frame_index": frame.frame_index,
                    "timestamp_sec": frame.timestamp_sec,
                    "num_faces": num_faces,
                    "frame_bgr": frame.image_bgr.copy(),
                    "detections": dets,
                    "description": f"{num_faces} Faces Localized & Tracked",
                    "score": float(num_faces)
                })

            # Check 2: Side-profile face & Tilted face
            for d in dets:
                if d.landmarks_5 is not None:
                    pose = compute_pose_metrics(d.landmarks_5)
                    # Side profile: yaw asymmetry >= 0.35 (roughly >= 25°-30° yaw)
                    if pose["yaw_asymmetry"] >= 0.35:
                        cases.append({
                            "type": "side_profile",
                            "video_id": meta.video_id,
                            "category": meta.category,
                            "frame_index": frame.frame_index,
                            "timestamp_sec": frame.timestamp_sec,
                            "num_faces": num_faces,
                            "frame_bgr": frame.image_bgr.copy(),
                            "detections": [d],
                            "description": f"Profile Face (Yaw ~{abs(pose['approx_yaw_deg']):.0f}°, Conf {d.confidence:.2f})",
                            "score": pose["yaw_asymmetry"]
                        })
                    # Head tilt: roll >= 20°
                    elif abs(pose["roll_deg"]) >= 20.0:
                        cases.append({
                            "type": "tilted_pose",
                            "video_id": meta.video_id,
                            "category": meta.category,
                            "frame_index": frame.frame_index,
                            "timestamp_sec": frame.timestamp_sec,
                            "num_faces": num_faces,
                            "frame_bgr": frame.image_bgr.copy(),
                            "detections": [d],
                            "description": f"Tilted Pose (Roll {pose['roll_deg']:.0f}°, Conf {d.confidence:.2f})",
                            "score": abs(pose["roll_deg"])
                        })

    except Exception as e:
        logger.error(f"Error analyzing {meta.video_id}: {e}")

    return cases


def create_edge_case_montage(
    multi_cases: List[Dict[str, Any]], 
    profile_cases: List[Dict[str, Any]], 
    output_path: str
):
    """Generate a combined publication-quality figure comparing edge cases."""
    if plt is None or cv2 is None:
        return

    n_multi = len(multi_cases)
    n_profile = len(profile_cases)
    max_cols = max(n_multi, n_profile, 1)

    fig, axes = plt.subplots(2, max_cols, figsize=(5.0 * max_cols, 9.0), squeeze=False)
    plt.suptitle("Stage 1 Stress Test: Multi-Person Frames & Non-Frontal Profile Detections", 
                 fontsize=15, fontweight="bold", y=0.98)

    # Row 1: Multi-Person Frames
    for col in range(max_cols):
        ax = axes[0, col]
        if col < n_multi:
            item = multi_cases[col]
            annotated = draw_detections(item["frame_bgr"], item["detections"])
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            ax.imshow(rgb)
            ax.set_title(f"Multi-Person [{item['category']}]\n{item['description']} (Video: {item['video_id']})", 
                         fontsize=11, fontweight="semibold", color="darkgreen")
        else:
            ax.axis("off")
        ax.axis("off")

    # Row 2: Side Profile / Tilted Faces
    for col in range(max_cols):
        ax = axes[1, col]
        if col < n_profile:
            item = profile_cases[col]
            annotated = draw_detections(item["frame_bgr"], item["detections"])
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            ax.imshow(rgb)
            ax.set_title(f"Side-Profile [{item['category']}]\n{item['description']} (Video: {item['video_id']})", 
                         fontsize=11, fontweight="semibold", color="darkblue")
        else:
            ax.axis("off")
        ax.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n[SUCCESS] Edge Case Montage saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Find and visualize multi-person frames and side profile faces.")
    parser.add_argument("--data-dir", type=str, default="/kaggle/working/data", help="Path to FF++ videos.")
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/results/visualizations/edge_cases")
    parser.add_argument("--max-frames", type=int, default=32, help="Frames sampled per video.")
    parser.add_argument("--max-samples", type=int, default=4, help="Max exemplars per edge case type.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    loader = FFppDatasetLoader(args.data_dir)
    logger.info(f"Scanning {len(loader)} videos for multi-person and profile face edge cases...")

    detector = SCRFDDetector(conf_threshold=0.45)
    tracker = IoUFaceTracker(iou_threshold=0.35)

    all_cases: List[Dict[str, Any]] = []
    for meta in loader.videos:
        cases = analyze_video_cases(meta, detector, tracker, max_frames=args.max_frames)
        all_cases.extend(cases)

    multi_cases = [c for c in all_cases if c["type"] == "multi_person"]
    profile_cases = [c for c in all_cases if c["type"] in ["side_profile", "tilted_pose"]]

    # Sort to get the most dramatic cases
    multi_cases.sort(key=lambda x: x["score"], reverse=True)
    profile_cases.sort(key=lambda x: x["score"], reverse=True)

    # Deduplicate by video_id to ensure diverse variety
    def get_diverse(cases_list, limit):
        seen_vids = set()
        diverse = []
        for c in cases_list:
            if c["video_id"] not in seen_vids:
                diverse.append(c)
                seen_vids.add(c["video_id"])
            if len(diverse) >= limit:
                break
        return diverse

    top_multi = get_diverse(multi_cases, args.max_samples)
    top_profile = get_diverse(profile_cases, args.max_samples)

    print("\n" + "="*75)
    print("           SCRFD STRESS TEST: EDGE CASE MINING RESULTS")
    print("="*75)
    print(f"Total Multi-Person Frames Found:  {len(multi_cases)}")
    print(f"Total Profile/Tilted Faces Found: {len(profile_cases)}")

    print("\nTop Multi-Person Frames:")
    for idx, c in enumerate(top_multi):
        print(f"  [{idx+1}] Video: {c['video_id']} ({c['category']}) | Frame: {c['frame_index']} | {c['description']}")
        if cv2 is not None:
            fn = f"multi_person_{c['category']}_{c['video_id']}_f{c['frame_index']}.jpg"
            annotated = draw_detections(c["frame_bgr"], c["detections"])
            cv2.imwrite(os.path.join(args.output_dir, fn), annotated)
            print(f"      -> Saved: {fn}")

    print("\nTop Side-Profile / Tilted Faces:")
    for idx, c in enumerate(top_profile):
        print(f"  [{idx+1}] Video: {c['video_id']} ({c['category']}) | Frame: {c['frame_index']} | {c['description']}")
        if cv2 is not None:
            fn = f"profile_{c['category']}_{c['video_id']}_f{c['frame_index']}.jpg"
            annotated = draw_detections(c["frame_bgr"], c["detections"])
            cv2.imwrite(os.path.join(args.output_dir, fn), annotated)
            print(f"      -> Saved: {fn}")

    print("="*75 + "\n")

    montage_path = os.path.join(args.output_dir, "scrfd_edge_cases_supervisor_montage.png")
    create_edge_case_montage(top_multi, top_profile, montage_path)


if __name__ == "__main__":
    main()
