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
            import os
            import insightface
            from ..utils.device import get_onnx_providers

            providers = get_onnx_providers()
            ctx_id = 0 if "CUDAExecutionProvider" in providers else -1
            logger.info(f"Initializing ArcFace encoder from pack '{self.model_pack}' (ctx_id={ctx_id}, providers={providers})...")

            # 1. Preferred: Load individual w600k_r50.onnx if available locally
            home_dir = os.path.expanduser("~")
            model_path = os.path.join(home_dir, ".insightface", "models", self.model_pack, "w600k_r50.onnx")

            if os.path.exists(model_path):
                self._rec_model = insightface.model_zoo.get_model(model_path, providers=providers)
                if self._rec_model is not None:
                    self._rec_model.prepare(ctx_id=ctx_id)
                    logger.info(f"ArcFace recognition model loaded directly from {model_path}.")
                    return

            # 2. Fallback: Load via FaceAnalysis application
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name=self.model_pack, allowed_modules=['detection', 'recognition'], providers=providers)
            app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            if 'recognition' in app.models:
                self._rec_model = app.models['recognition']
            else:
                for model in app.models.values():
                    if hasattr(model, "get_feat"):
                        self._rec_model = model
                        break

            if self._rec_model is not None:
                logger.info("ArcFace encoder successfully initialized.")
            else:
                logger.warning("ArcFace recognition model not found in FaceAnalysis.")
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
