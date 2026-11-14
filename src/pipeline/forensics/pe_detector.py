"""
GenD detector using Perception Encoder (PE) Large visual backbone.
"""

from .gend_detector import BaseGenDDetector

class GenDPEDetector(BaseGenDDetector):
    """GenD detector variant with Perception Encoder (PE) backbone."""
    
    def __init__(self, repo_id: str = "yermandy/GenD_PE_L", **kwargs):
        super().__init__(repo_id=repo_id, **kwargs)
