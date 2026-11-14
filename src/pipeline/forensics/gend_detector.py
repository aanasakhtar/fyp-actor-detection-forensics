"""
Base GenD (Generalized Deepfake Detection) inference engine.
"""

import numpy as np
from typing import List, Optional, Dict, Any, Union
from ..utils.logging import get_logger
from ..utils.device import select_device

logger = get_logger("gend_detector")

class BaseGenDDetector:
    """Base class for GenD deepfake detectors hosted on Hugging Face."""
    
    def __init__(
        self, 
        repo_id: str, 
        device: Optional[Union[str, Any]] = None,
        decision_threshold: float = 0.50
    ):
        self.repo_id = repo_id
        self.device = select_device(str(device)) if isinstance(device, str) or device is None else device
        self.decision_threshold = decision_threshold
        self.model = None
        self.processor = None
        self._init_model()
        
    def _init_model(self) -> None:
        """Load the GenD model weights from Hugging Face or local checkpoint."""
        logger.info(f"Loading GenD model from '{self.repo_id}' onto {self.device}...")
        try:
            import torch
            from transformers import AutoModelForImageClassification, AutoImageProcessor
            self.processor = AutoImageProcessor.from_pretrained(self.repo_id)
            self.model = AutoModelForImageClassification.from_pretrained(self.repo_id)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Successfully loaded {self.repo_id}")
        except Exception as e:
            logger.warning(f"AutoModel loading for {self.repo_id} unavailable ({e}). Fallback mode active.")
            self.model = None

    def predict_crops(self, crop_images_bgr: List[np.ndarray]) -> List[float]:
        """
        Inference on a list of 224x224 face crops in BGR format.
        Returns list of manipulation probabilities p(Fake) in [0, 1].
        """
        if not crop_images_bgr:
            return []
            
        if self.model is None or self.processor is None:
            # Fallback for offline dev testing
            return [0.0] * len(crop_images_bgr)
            
        try:
            import torch
            from PIL import Image
            import cv2
            pil_images = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crop_images_bgr]
            inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                fake_probs = probs[:, 1].detach().cpu().numpy().tolist()
                return fake_probs
        except Exception as e:
            logger.error(f"Inference error in GenD: {e}")
            return [0.0] * len(crop_images_bgr)

    @staticmethod
    def aggregate_scores(
        frame_scores: List[float], 
        method: str = "trimmed_mean", 
        trim_ratio: float = 0.20
    ) -> float:
        """Aggregate a sequence of frame-level manipulation probabilities into a video-level score."""
        if not frame_scores:
            return 0.0
            
        scores = np.array(frame_scores, dtype=np.float32)
        if method == "mean":
            return float(np.mean(scores))
        elif method == "max":
            return float(np.max(scores))
        elif method == "trimmed_mean":
            if len(scores) <= 4:
                return float(np.mean(scores))
            k = int(len(scores) * trim_ratio)
            trimmed = np.sort(scores)[k:len(scores) - k] if k > 0 else scores
            return float(np.mean(trimmed))
        elif method == "ratio_above_threshold":
            return float(np.mean(scores > 0.50))
        return float(np.mean(scores))
