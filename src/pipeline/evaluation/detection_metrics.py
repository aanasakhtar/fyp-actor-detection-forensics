"""
Face detection evaluation metrics.
"""

from typing import List, Dict, Any
import numpy as np

def compute_detection_coverage(frame_detections: List[List[Any]]) -> float:
    """
    Calculate proportion of frames containing at least one detected face.
    frame_detections is a list of lists of detections per frame.
    """
    if not frame_detections:
        return 0.0
    valid_frames = sum(1 for dets in frame_detections if len(dets) > 0)
    return float(valid_frames / len(frame_detections))

def compute_detection_confidence_summary(confidences: List[float]) -> Dict[str, float]:
    """Summary statistics for face detection confidence scores."""
    if not confidences:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    arr = np.array(confidences)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr))
    }

def compute_bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes [xmin, ymin, xmax, ymax]."""
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    inter_area = max(0.0, x_b - x_a) * max(0.0, y_b - y_a)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)

def compute_inter_pupillary_distance(kps_5: np.ndarray) -> float:
    """
    Compute Euclidean distance between left eye (index 0) and right eye (index 1) in pixels.
    kps_5 shape: (5, 2)
    """
    if kps_5 is None or len(kps_5) < 2:
        return 0.0
    left_eye = kps_5[0]
    right_eye = kps_5[1]
    return float(np.linalg.norm(left_eye - right_eye))

def compute_landmark_geometric_validity(kps_5: np.ndarray, bbox: np.ndarray) -> bool:
    """
    Verify physiological sanity of 5-point facial landmarks:
    1. Both eyes above nose, nose above mouth corners.
    2. Landmarks fall within or near the bounding box.
    3. Inter-pupillary distance > 5 pixels.
    """
    if kps_5 is None or len(kps_5) < 5:
        return False
    left_eye, right_eye, nose, left_mouth, right_mouth = kps_5[:5]
    eyes_y = (left_eye[1] + right_eye[1]) / 2.0
    mouth_y = (left_mouth[1] + right_mouth[1]) / 2.0
    if not (eyes_y <= nose[1] + 12.0 and nose[1] <= mouth_y + 12.0):
        return False
    ipd = np.linalg.norm(left_eye - right_eye)
    if ipd < 5.0:
        return False
    margin_w = (bbox[2] - bbox[0]) * 0.25
    margin_h = (bbox[3] - bbox[1]) * 0.25
    if np.any(kps_5[:, 0] < bbox[0] - margin_w) or np.any(kps_5[:, 0] > bbox[2] + margin_w):
        return False
    if np.any(kps_5[:, 1] < bbox[1] - margin_h) or np.any(kps_5[:, 1] > bbox[3] + margin_h):
        return False
    return True
