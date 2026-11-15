"""
Evidence card generation for video forensic review cases.
"""

import os
try:
    import cv2
except ImportError:
    cv2 = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

import numpy as np
from typing import Dict, Any, Optional

def generate_evidence_card(
    case_summary: Dict[str, Any],
    representative_frame: np.ndarray,
    face_crop: Optional[np.ndarray],
    save_path: Optional[str] = None
) -> None:
    """Generate an evidence visual inspection card combining frame, crop, and scores."""
    if plt is None or cv2 is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Left: Full frame
    frame_rgb = cv2.cvtColor(representative_frame, cv2.COLOR_BGR2RGB)
    axes[0].imshow(frame_rgb)
    axes[0].set_title(f"Video: {case_summary.get('video_id', 'Unknown')}")
    axes[0].axis("off")
    
    # Right: Face crop
    if face_crop is not None:
        crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        axes[1].imshow(crop_rgb)
        status_text = (
            f"Actor: {case_summary.get('matched_actor', 'None')}\n"
            f"Sim Score: {case_summary.get('actor_similarity', 0.0):.2f}\n"
            f"GenD Score: {case_summary.get('manipulation_score', 0.0):.2f}\n"
            f"Status: {case_summary.get('permission_status', 'Unknown')}"
        )
        axes[1].set_title(status_text)
    axes[1].axis("off")
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
