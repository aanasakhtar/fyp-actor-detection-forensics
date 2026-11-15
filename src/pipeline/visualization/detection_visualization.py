"""
Visual overlays for face detection bounding boxes, facial keypoints, and track IDs.
"""

try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np
from typing import List
from ..detection.detection_utils import FaceDetection

def draw_detections(
    image_bgr: np.ndarray, 
    detections: List[FaceDetection]
) -> np.ndarray:
    """Draw bounding boxes, keypoints, and track labels onto a video frame."""
    canvas = image_bgr.copy()
    if cv2 is None:
        return canvas
    
    for det in detections:
        x1, y1, x2, y2 = det.bbox.astype(int)
        # Draw bounding box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw confidence and track ID
        track_str = f"ID:{det.track_id} " if det.track_id is not None else ""
        label = f"{track_str}{det.confidence:.2f}"
        cv2.putText(canvas, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw 5 landmarks
        if det.landmarks_5 is not None:
            for pt in det.landmarks_5:
                px, py = int(pt[0]), int(pt[1])
                cv2.circle(canvas, (px, py), 3, (0, 0, 255), -1)
                
    return canvas
