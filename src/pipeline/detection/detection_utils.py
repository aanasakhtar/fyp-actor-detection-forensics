"""
Face detection data structures and bounding box utilities.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class FaceDetection:
    """Represents a single face detected in a frame."""
    frame_index: int
    timestamp_sec: float
    bbox: np.ndarray             # [x1, y1, x2, y2]
    confidence: float
    landmarks_5: Optional[np.ndarray] = None  # shape (5, 2)
    track_id: Optional[int] = None

def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    
    inter_area = max(0.0, x_b - x_a) * max(0.0, y_b - y_a)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    
    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)
