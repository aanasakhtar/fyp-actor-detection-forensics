"""
GenD detector using CLIP ViT-L/14 visual encoder.
"""

from .gend_detector import BaseGenDDetector

class GenDCLIPDetector(BaseGenDDetector):
    """GenD detector variant with OpenAI CLIP ViT-L/14 backbone."""
    
    def __init__(self, repo_id: str = "yermandy/GenD_CLIP_L_14", **kwargs):
        super().__init__(repo_id=repo_id, **kwargs)
