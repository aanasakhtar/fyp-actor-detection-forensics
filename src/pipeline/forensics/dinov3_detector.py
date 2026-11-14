"""
GenD detector using DINOv3 ViT-L/16 visual encoder.
"""

from .gend_detector import BaseGenDDetector

class GenDDINOv3Detector(BaseGenDDetector):
    """GenD detector variant with Meta DINOv3 ViT-L/16 backbone."""
    
    def __init__(self, repo_id: str = "yermandy/GenD_DINOv3_L", **kwargs):
        super().__init__(repo_id=repo_id, **kwargs)
