"""
Vision Transformer (ViT) Deepfake Detector using 'dima806/deepfake_vs_real_image_detection'.
"""

import os
import sys
import numpy as np
from typing import List, Optional, Union, Any
from ..utils.logging import get_logger
from ..utils.device import select_device

logger = get_logger("vit_detector")

try:
    import torch
except (ImportError, OSError):
    torch = None


class ViTDeepfakeDetector:
    """ViT deepfake detector variant fine-tuned on real vs fake facial forensics."""
    def __init__(
        self,
        repo_id: str = "dima806/deepfake_vs_real_image_detection",
        device: Optional[Union[str, Any]] = None
    ):
        self.repo_id = repo_id
        self.device = select_device(str(device)) if isinstance(device, str) or device is None else device
        self.model = None
        self.processor = None
        self._init_model()

    def _init_model(self) -> None:
        if torch is None:
            logger.warning("PyTorch not installed. ViTDeepfakeDetector running in mock mode.")
            return

        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            logger.info(f"Loading ViT Deepfake Detector from '{self.repo_id}' onto {self.device}...")
            self.processor = AutoImageProcessor.from_pretrained(self.repo_id)
            self.model = AutoModelForImageClassification.from_pretrained(self.repo_id)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Successfully loaded ViT Deepfake Detector.")
        except Exception as e:
            logger.error(f"Failed to load ViT Deepfake Detector: {e}")
            self.model = None
            self.processor = None

    def get_target_size(self) -> tuple:
        """Return native spatial resolution expected by ViT."""
        return (224, 224)

    def predict_crops(self, crop_images_bgr: List[np.ndarray], batch_size: int = 64) -> List[float]:
        """Inference on list of face crops in BGR format using chunked mini-batches."""
        if not crop_images_bgr or self.model is None or self.processor is None:
            return [0.0] * len(crop_images_bgr)

        try:
            import cv2
            from PIL import Image

            pil_images = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crop_images_bgr]
            use_cuda = hasattr(self.device, "type") and self.device.type == "cuda"
            all_probs: List[float] = []

            for start in range(0, len(pil_images), batch_size):
                chunk = pil_images[start:start + batch_size]
                inputs = self.processor(images=chunk, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    with torch.amp.autocast("cuda", enabled=use_cuda):
                        outputs = self.model(**inputs)
                    probs = torch.softmax(outputs.logits.float(), dim=-1)
                    # Class 0: Real, Class 1: Fake
                    fake_probs = probs[:, 1].detach().cpu().numpy().tolist()
                    all_probs.extend(float(p) for p in fake_probs)

            return all_probs
        except Exception as e:
            logger.error(f"Inference error in ViT Detector: {e}")
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
        elif method == "median":
            return float(np.median(scores))
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
