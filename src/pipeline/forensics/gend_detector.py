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

        # 1. Direct weights loading via huggingface_hub + safetensors (bypasses meta-device and tied-weights bugs)
        try:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            from .modeling_gend import GenD, GenDConfig
            import json
            import torch

            config_path = hf_hub_download(repo_id=self.repo_id, filename="config.json")
            with open(config_path, "r", encoding="utf-8") as f:
                cfg_dict = json.load(f)
            config = GenDConfig(**cfg_dict)
            model = GenD(config)

            try:
                weights_path = hf_hub_download(repo_id=self.repo_id, filename="model.safetensors")
                state_dict = load_file(weights_path)
            except Exception:
                weights_path = hf_hub_download(repo_id=self.repo_id, filename="pytorch_model.bin")
                state_dict = torch.load(weights_path, map_location="cpu")

            model.load_state_dict(state_dict, strict=False)
            model.to(self.device)
            model.eval()
            self.model = model
            self.processor = getattr(self.model, "feature_extractor", None)
            logger.info(f"Successfully loaded GenD weights directly for '{self.repo_id}'")
            return
        except Exception as e_direct:
            logger.warning(f"Direct weights loading encountered: {e_direct}. Trying native GenD from_pretrained...")

        # 2. Native author GenD class loading with Hugging Face auto registration
        try:
            from .modeling_gend import GenD, GenDConfig
            from transformers import AutoConfig, AutoModel
            try:
                AutoConfig.register("GenD", GenDConfig)
                AutoModel.register(GenDConfig, GenD)
            except Exception:
                pass

            self.model = GenD.from_pretrained(self.repo_id, low_cpu_mem_usage=False)
            self.model.to(self.device)
            self.model.eval()
            self.processor = getattr(self.model, "feature_extractor", None)
            logger.info(f"Successfully loaded author's native GenD architecture from '{self.repo_id}'")
            return
        except Exception as e_native:
            logger.warning(f"Native GenD loading encountered: {e_native}. Trying AutoModel fallback...")

        # 3. Try AutoModel directly
        try:
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(self.repo_id, trust_remote_code=True, low_cpu_mem_usage=False)
            self.model.to(self.device)
            self.model.eval()
            self.processor = getattr(self.model, "feature_extractor", None)
            logger.info(f"Successfully loaded via AutoModel for '{self.repo_id}'")
            return
        except Exception as e_auto:
            logger.warning(f"Could not load AutoModel for {self.repo_id} ({e_auto}). Offline fallback active.")
            self.model = None
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

            # 1. Use author's native bundled feature_extractor if available
            if hasattr(self.model, "feature_extractor") and hasattr(self.model.feature_extractor, "preprocess"):
                tensors = torch.stack([self.model.feature_extractor.preprocess(img) for img in pil_images]).to(self.device)
            elif self.processor is not None and hasattr(self.processor, "preprocess"):
                tensors = torch.stack([self.processor.preprocess(img) for img in pil_images]).to(self.device)
            elif self.processor is not None and hasattr(self.processor, "__call__"):
                inputs = self.processor(images=pil_images, return_tensors="pt")
                tensors = inputs["pixel_values"].to(self.device)
            else:
                # Built-in tensor normalization fallback (OpenAI CLIP statistics)
                mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
                std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
                batch = []
                for img in pil_images:
                    arr = np.array(img.resize((224, 224)), dtype=np.float32) / 255.0
                    normed = (arr - mean) / std
                    batch.append(normed.transpose(2, 0, 1))
                tensors = torch.tensor(np.stack(batch), dtype=torch.float32).to(self.device)

            with torch.no_grad():
                outputs = self.model(tensors)

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
