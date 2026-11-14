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
