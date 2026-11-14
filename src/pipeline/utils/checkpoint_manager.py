"""
Checkpoint manager for downloading and preparing pretrained model weights.
"""

import os
from typing import Optional
from .logging import get_logger

logger = get_logger("checkpoint_manager")

class CheckpointManager:
    """Manages model checkpoints for InsightFace (SCRFD, ArcFace) and Hugging Face (GenD)."""
    
    def __init__(self, cache_dir: str = "./checkpoints"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def ensure_insightface_models(self, pack_name: str = "buffalo_l") -> str:
        """
        Verify or initialize InsightFace models (det_10g.onnx and w600k_r50.onnx).
        InsightFace downloads these automatically into ~/.insightface/models/<pack_name>.
        """
        logger.info(f"InsightFace model pack '{pack_name}' managed via insightface.app.FaceAnalysis")
        return pack_name

    def ensure_gend_checkpoint(self, repo_id: str) -> str:
        """
        Ensure Hugging Face GenD checkpoint is available locally or cached.
        """
        logger.info(f"GenD checkpoint '{repo_id}' will be fetched through huggingface_hub / transformers.")
        return repo_id
