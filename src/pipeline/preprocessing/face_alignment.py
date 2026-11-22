"""
Facial landmark alignment for ArcFace and GenD using standard 5-point similarity transformation.
"""

import numpy as np
from typing import Optional, Tuple

# Standard reference template for 112x112 ArcFace face crop
STANDARD_112_TEMPLATE = np.array([
    [38.2946, 51.6963],   # Left eye
    [73.5318, 51.5014],   # Right eye
    [56.0252, 71.7366],   # Nose tip
    [41.5493, 92.3655],   # Left mouth corner
    [70.7299, 92.2041]    # Right mouth corner
], dtype=np.float32)

def align_face_5point(image_bgr: np.ndarray, landmarks_5: np.ndarray, output_size: int = 112) -> Optional[np.ndarray]:
    """Perform similarity transformation to align face according to 5 facial landmarks."""
    if landmarks_5 is None or len(landmarks_5) != 5:
        return None
        
    try:
        import cv2
    except ImportError:
        return None
        
    src_pts = np.array(landmarks_5, dtype=np.float32)
    dst_pts = STANDARD_112_TEMPLATE.copy()
    
    if output_size != 112:
        dst_pts = dst_pts * (output_size / 112.0)
        
    transform_matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
    if transform_matrix is None:
        transform_matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        
    if transform_matrix is None:
        return None
        
    aligned = cv2.warpAffine(
        image_bgr, 
        transform_matrix, 
        (output_size, output_size), 
        borderMode=cv2.BORDER_REFLECT_101
    )
    return aligned


def align_face_gend(
    image_bgr: np.ndarray, 
    landmarks_5: np.ndarray, 
    target_size: Tuple[int, int] = (224, 224), 
    scale: float = 1.30
) -> Optional[np.ndarray]:
    """
    Perform 5-point affine alignment matching GenD (WACV 2026) canonical training geometry.
    Ensures eye, nose, and mouth spatial features align with GenD Vision Transformer patches.
    """
    if landmarks_5 is None or len(landmarks_5) != 5:
        return None
        
    try:
        import cv2
    except ImportError:
        return None

    dst = np.array([
        [0.34, 0.46],
        [0.66, 0.46],
        [0.50, 0.64],
        [0.37, 0.82],
        [0.63, 0.82],
    ], dtype=np.float32)

    dst_pts = dst.copy()
    dst_pts[:, 0] *= target_size[0]
    dst_pts[:, 1] *= target_size[1]

    margin_rate = scale - 1.0
    x_margin = target_size[0] * margin_rate / 2.0
    y_margin = target_size[1] * margin_rate / 2.0

    dst_pts[:, 0] += x_margin
    dst_pts[:, 1] += y_margin
    dst_pts[:, 0] *= target_size[0] / (target_size[0] + 2.0 * x_margin)
    dst_pts[:, 1] *= target_size[1] / (target_size[1] + 2.0 * y_margin)

    src_pts = np.array(landmarks_5, dtype=np.float32)
    tform, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
    if tform is None:
        tform, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    if tform is None:
        return None

    aligned = cv2.warpAffine(
        image_bgr,
        tform,
        target_size,
        borderMode=cv2.BORDER_REFLECT_101
    )
    return aligned

