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

        # 1. Load Model Architecture & Weights via AutoModel
        try:
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(self.repo_id, trust_remote_code=True)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Successfully loaded model weights from '{self.repo_id}'")
        except Exception as e_model:
            logger.warning(f"Could not load AutoModel for {self.repo_id} ({e_model}). Fallback mode active.")
            self.model = None

        # 2. Load Image Processor
        # GenD repos do not bundle a standalone preprocessor_config.json; fallback to base vision backbone processor
        base_processor_id = "openai/clip-vit-large-patch14"
        if "dinov" in self.repo_id.lower():
            base_processor_id = "facebook/dinov2-large"

        try:
            from transformers import AutoImageProcessor
            try:
                self.processor = AutoImageProcessor.from_pretrained(self.repo_id, trust_remote_code=True)
            except Exception:
                logger.info(f"Using base vision processor '{base_processor_id}' for {self.repo_id}...")
                self.processor = AutoImageProcessor.from_pretrained(base_processor_id)
        except Exception as e_proc:
            logger.warning(f"AutoImageProcessor unavailable ({e_proc}). Built-in normalization active.")
            self.processor = None

    def predict_crops(self, crop_images_bgr: List[np.ndarray]) -> List[float]:
        """
        Inference on a list of 224x224 face crops in BGR format.
        Returns list of manipulation probabilities p(Fake) in [0, 1].
        """
        if not crop_images_bgr:
            return []

        if self.model is None:
            # Fallback for offline dev
            return [0.0] * len(crop_images_bgr)

        try:
            import torch
            from PIL import Image
            import cv2

            # Convert BGR crops to RGB PIL images
            pil_images = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crop_images_bgr]

            if self.processor is not None:
                inputs = self.processor(images=pil_images, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self.device)
            else:
                # Built-in tensor normalization fallback (OpenAI CLIP statistics)
                mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
                std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
                tensors = []
                for img in pil_images:
                    arr = np.array(img.resize((224, 224)), dtype=np.float32) / 255.0
                    normed = (arr - mean) / std
                    tensors.append(normed.transpose(2, 0, 1))
                pixel_values = torch.tensor(np.stack(tensors), dtype=torch.float32).to(self.device)

            with torch.no_grad():
                try:
                    outputs = self.model(pixel_values)
                except TypeError:
                    outputs = self.model(pixel_values=pixel_values)

                # Extract logits
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                elif isinstance(outputs, torch.Tensor):
                    logits = outputs
                elif hasattr(outputs, "logits_labels"):
                    logits = outputs.logits_labels
                elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
                    logits = outputs[0]
                else:
                    logits = outputs

                probs = torch.softmax(logits, dim=-1)

                # Binary classification (Class 0: Real, Class 1: Fake)
                if probs.shape[-1] >= 2:
                    fake_probs = probs[:, 1].detach().cpu().numpy().tolist()
                else:
                    fake_probs = probs[:, 0].detach().cpu().numpy().tolist()

                return [float(p) for p in fake_probs]

        except Exception as e:
            logger.error(f"Inference error in GenD ({self.repo_id}): {e}")
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
