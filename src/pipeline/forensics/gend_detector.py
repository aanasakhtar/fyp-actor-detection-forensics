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
        self.pipeline = None
        self._init_model()
        
    def _init_model(self) -> None:
        """Load the GenD model weights from Hugging Face or local checkpoint."""
        logger.info(f"Loading GenD model from '{self.repo_id}' onto {self.device}...")
        self.pipeline = None

        # 1. Try high-level Hugging Face pipeline
        try:
            from transformers import pipeline
            device_arg = 0 if str(self.device).startswith("cuda") else -1
            self.pipeline = pipeline(
                "image-classification", 
                model=self.repo_id, 
                device=device_arg,
                trust_remote_code=True
            )
            logger.info(f"Successfully loaded {self.repo_id} via HuggingFace pipeline.")
            return
        except Exception as e1:
            logger.debug(f"Pipeline loading failed ({e1}), attempting AutoModelForImageClassification...")

        # 2. Try AutoModelForImageClassification with trust_remote_code
        try:
            from transformers import AutoModelForImageClassification, AutoImageProcessor
            self.processor = AutoImageProcessor.from_pretrained(self.repo_id, trust_remote_code=True)
            self.model = AutoModelForImageClassification.from_pretrained(self.repo_id, trust_remote_code=True)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Successfully loaded {self.repo_id} via AutoModelForImageClassification.")
            return
        except Exception as e2:
            logger.debug(f"AutoModelForImageClassification failed ({e2}), attempting AutoModel...")

        # 3. Try AutoModel with trust_remote_code
        try:
            from transformers import AutoModel, AutoImageProcessor
            self.processor = AutoImageProcessor.from_pretrained(self.repo_id, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(self.repo_id, trust_remote_code=True)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Successfully loaded {self.repo_id} via AutoModel.")
            return
        except Exception as e3:
            logger.warning(f"All loaders for {self.repo_id} failed: pipeline ({e1}), AutoModelClass ({e2}), AutoModel ({e3}). Fallback mode active.")
            self.model = None
            self.processor = None

    def predict_crops(self, crop_images_bgr: List[np.ndarray]) -> List[float]:
        """
        Inference on a list of 224x224 face crops in BGR format.
        Returns list of manipulation probabilities p(Fake) in [0, 1].
        """
        if not crop_images_bgr:
            return []

        try:
            import torch
            from PIL import Image
            import cv2
        except ImportError:
            return [0.0] * len(crop_images_bgr)

        # 1. Pipeline inference path
        if getattr(self, "pipeline", None) is not None:
            try:
                pil_images = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crop_images_bgr]
                results = self.pipeline(pil_images, top_k=None)
                if isinstance(results, dict):
                    results = [[results]]
                elif isinstance(results, list) and results and isinstance(results[0], dict):
                    results = [results]

                fake_probs = []
                for r in results:
                    fake_score = 0.50
                    for item in r:
                        lbl = str(item.get("label", "")).lower()
                        score = float(item.get("score", 0.0))
                        if any(k in lbl for k in ["fake", "synth", "manip", "deepfake"]) or lbl in ["1", "label_1"]:
                            fake_score = score
                            break
                        elif any(k in lbl for k in ["real", "pristine", "original"]) or lbl in ["0", "label_0"]:
                            fake_score = 1.0 - score
                    fake_probs.append(fake_score)
                return fake_probs
            except Exception as e:
                logger.error(f"Pipeline inference failed ({e}), attempting PyTorch model...")

        # 2. PyTorch model inference path
        if self.model is not None and self.processor is not None:
            try:
                pil_images = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crop_images_bgr]
                inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                    probs = torch.softmax(logits, dim=-1)
                    fake_probs = probs[:, 1].detach().cpu().numpy().tolist()
                    return fake_probs
            except Exception as e:
                logger.error(f"Inference error in GenD PyTorch model: {e}")
                return [0.0] * len(crop_images_bgr)

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
