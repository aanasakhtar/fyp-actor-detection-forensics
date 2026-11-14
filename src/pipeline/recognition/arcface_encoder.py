"""
ArcFace deep facial feature extractor.
"""

import numpy as np
from typing import Optional, List
from ..utils.logging import get_logger
from ..utils.device import get_onnx_providers

logger = get_logger("arcface_encoder")

class ArcFaceEncoder:
    """ArcFace feature encoder generating 512-dimensional normalized embeddings."""
    
    def __init__(self, model_pack: str = "buffalo_l"):
        self.model_pack = model_pack
        self._rec_model = None
        self._init_model()
        
    def _init_model(self) -> None:
        """Initialize the recognition model via InsightFace."""
        try:
            from insightface.app import FaceAnalysis
            providers = get_onnx_providers()
            logger.info(f"Initializing ArcFace encoder from pack '{self.model_pack}'...")
            app = FaceAnalysis(name=self.model_pack, allowed_modules=['recognition'], providers=providers)
            app.prepare(ctx_id=0)
            for model in app.models.values():
                if hasattr(model, "get_feat"):
                    self._rec_model = model
                    break
            logger.info("ArcFace encoder successfully initialized.")
        except Exception as e:
            logger.warning(f"ArcFace recognition module unavailable: {e}. Fallback mode active.")
            self._rec_model = None

    def extract_embedding(self, aligned_face_112: np.ndarray) -> Optional[np.ndarray]:
        """Compute normalized 512-d embedding for a single aligned 112x112 face crop."""
        if aligned_face_112 is None or self._rec_model is None:
            return None
            
        try:
            feat = self._rec_model.get_feat([aligned_face_112])
            embedding = feat[0]
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Error computing ArcFace embedding: {e}")
            return None
