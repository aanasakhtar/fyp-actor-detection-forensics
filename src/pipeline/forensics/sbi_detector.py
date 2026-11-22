"""
SBI (Self-Blended Images) Contrastive Deepfake Detector using EfficientNet-B4.
Checkpoint: 'nikokons/contrastive-deepfake-detector' (CVPR 2022 SBI + Triplet Contrastive Loss).
Evaluated on Celeb-DF v2 achieving 93.96% AUC and 87.84% accuracy.
"""

import os
import sys
import numpy as np
from typing import List, Optional, Union, Any
from ..utils.logging import get_logger
from ..utils.device import select_device

logger = get_logger("sbi_detector")

try:
    import torch
    import torch.nn as nn
except (ImportError, OSError):
    torch = None
    nn = object


class FeatureExtractor(nn.Module if torch is not None else object):
    """Wrapper around efficientnet_pytorch backbone with advprop=True."""
    def __init__(self, model_name: str = "efficientnet-b4"):
        super().__init__()
        try:
            from efficientnet_pytorch import EfficientNet
        except ImportError:
            import subprocess
            logger.info("Installing 'efficientnet_pytorch' dependency...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "efficientnet_pytorch"])
            from efficientnet_pytorch import EfficientNet

        self.efficient_net = EfficientNet.from_pretrained(model_name, advprop=True)
        self.efficient_net._fc = nn.Identity()

    def forward(self, x: Any) -> Any:
        return self.efficient_net.extract_features(x)


class Detector(nn.Module if torch is not None else object):
    """Exact PyTorch architecture from nikokons/contrastive-deepfake-detector."""
    def __init__(self, model_name: str = "efficientnet-b4"):
        super().__init__()
        self.global_feature_extractor = FeatureExtractor(model_name)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1792, 2)

    def forward(self, img: Any) -> Any:
        features = self.global_feature_extractor(img)
        pooled = self.global_pool(features).view(features.size(0), -1)
        return self.classifier(pooled)


class SBIEfficientNetDetector:
    """
    SBI (Self-Blended Images) Contrastive Deepfake Detector using EfficientNet-B4.
    Trained by nikokons on FaceForensics++ SBI and tested on Celeb-DF v2 (93.96% AUC).
    Expects 380x380 face crops normalized to AdvProp range [-1, 1].
    """
    def __init__(
        self,
        repo_id: str = "nikokons/contrastive-deepfake-detector",
        filename: str = "79_0.9980_val.tar",
        device: Optional[Union[str, Any]] = None,
        target_size: int = 380
    ):
        self.repo_id = repo_id
        self.filename = filename
        self.target_size = target_size
        self.device = select_device(str(device)) if isinstance(device, str) or device is None else device
        self.model = None
        self._init_model()

    def _init_model(self) -> None:
        """Download checkpoint from Hugging Face Hub and load weights into Detector."""
        if torch is None:
            logger.warning("PyTorch not installed. SBIEfficientNetDetector running in mock mode.")
            return

        logger.info(f"Loading SBI EfficientNet-B4 checkpoint from '{self.repo_id}' ({self.filename})...")
        try:
            from huggingface_hub import hf_hub_download
            checkpoint_path = hf_hub_download(repo_id=self.repo_id, filename=self.filename)
            
            model = Detector("efficientnet-b4")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
            
            # Clean DDP module. prefix if present
            cleaned_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(cleaned_state_dict)
            model.to(self.device)
            model.eval()
            self.model = model
            logger.info("Successfully loaded SBI EfficientNet-B4 detector onto %s.", self.device)
        except Exception as e:
            logger.error(f"Failed to load SBI EfficientNet-B4 model: {e}")
            self.model = None

    def predict_crops(self, crop_images_bgr: List[np.ndarray]) -> List[float]:
        """
        Run inference on a batch of face crops in BGR format.
        Crops are resized to 380x380 and normalized with AdvProp stats (mean=0.5, std=0.5).
        Returns list of manipulation probabilities p(Fake) in [0, 1].
        """
        if not crop_images_bgr:
            return []

        if self.model is None:
            return [0.0] * len(crop_images_bgr)

        try:
            import cv2
            batch_tensors = []
            for crop in crop_images_bgr:
                # Resize to expected 380x380
                if crop.shape[0] != self.target_size or crop.shape[1] != self.target_size:
                    resized = cv2.resize(crop, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
                else:
                    resized = crop
                
                # Convert BGR to RGB
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                
                # AdvProp normalization: [-1, 1], mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                advprop_norm = (rgb - 0.5) / 0.5
                tensor = advprop_norm.transpose(2, 0, 1)  # (C, H, W)
                batch_tensors.append(tensor)

            batch = torch.tensor(np.stack(batch_tensors), dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                logits = self.model(batch)
                probs = torch.softmax(logits, dim=-1)
                # Class 0: Real, Class 1: Fake
                fake_probs = probs[:, 1].detach().cpu().numpy().tolist()
                return [float(p) for p in fake_probs]
        except Exception as e:
            logger.error(f"Inference error in SBI EfficientNet-B4: {e}")
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
