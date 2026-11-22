"""
Contextual face cropping for GenD deepfake manipulation detection.
"""

import numpy as np
from typing import Optional, Tuple

def crop_context_face(
    image_bgr: np.ndarray, 
    bbox: np.ndarray, 
    expansion_factor: float = 1.30, 
    target_size: Tuple[int, int] = (224, 224)
) -> Optional[np.ndarray]:
    """
    Extract a contextual face crop with an enlarged margin around the detected bounding box,
    preserving hair, forehead, and chin boundary artifacts critical for deepfake detection.
    """
    if bbox is None or len(bbox) < 4:
        return None
        
    h_img, w_img = image_bgr.shape[:2]
    x1, y1, x2, y2 = bbox[:4]
    
    w_box = x2 - x1
    h_box = y2 - y1
    if w_box <= 0 or h_box <= 0:
        return None
        
    cx = x1 + w_box / 2.0
    cy = y1 + h_box / 2.0
    
    expanded_w = w_box * expansion_factor
    expanded_h = h_box * expansion_factor
    size = max(expanded_w, expanded_h)
    
    new_x1 = int(round(cx - size / 2.0))
    new_y1 = int(round(cy - size / 2.0))
    new_x2 = int(round(cx + size / 2.0))
    new_y2 = int(round(cy + size / 2.0))
    
    pad_left = max(0, -new_x1)
    pad_top = max(0, -new_y1)
    pad_right = max(0, new_x2 - w_img)
    pad_bottom = max(0, new_y2 - h_img)
    
    crop_x1 = max(0, new_x1)
    crop_y1 = max(0, new_y1)
    crop_x2 = min(w_img, new_x2)
    crop_y2 = min(h_img, new_y2)
    
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None
        
    sub_img = image_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
    
    # Try using cv2 if available, else numpy pad
    try:
        import cv2
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            sub_img = cv2.copyMakeBorder(
                sub_img, pad_top, pad_bottom, pad_left, pad_right, 
                borderType=cv2.BORDER_REFLECT_101
            )
        interp = cv2.INTER_AREA if (sub_img.shape[0] >= target_size[1] and sub_img.shape[1] >= target_size[0]) else cv2.INTER_LINEAR
        resized = cv2.resize(sub_img, target_size, interpolation=interp)
        return resized
    except Exception:
        # Fallback using numpy if cv2 is absent in development environment
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            sub_img = np.pad(sub_img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="reflect")
        # Basic nearest neighbor / zoom resize using numpy
        h_s, w_s = sub_img.shape[:2]
        r_indices = (np.linspace(0, h_s - 1, target_size[1])).astype(int)
        c_indices = (np.linspace(0, w_s - 1, target_size[0])).astype(int)
        resized = sub_img[np.ix_(r_indices, c_indices)]
        return resized
