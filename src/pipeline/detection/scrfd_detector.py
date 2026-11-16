"""
SCRFD Face Detector implementation using InsightFace / ONNX Runtime.
"""

import numpy as np
from typing import List, Optional, Tuple
from .detection_utils import FaceDetection
from ..utils.logging import get_logger
from ..utils.device import get_onnx_providers

logger = get_logger("scrfd_detector")

class SCRFDDetector:
    """SCRFD 10G KPS face detector wrapper."""
    
    def __init__(
        self, 
        model_pack: str = "buffalo_l", 
        det_size: Tuple[int, int] = (640, 640),
        conf_threshold: float = 0.50
    ):
        self.model_pack = model_pack
        self.det_size = det_size
        self.conf_threshold = conf_threshold
        self._app = None
        self._init_model()
        
    def _init_model(self) -> None:
        """Initialize the InsightFace FaceAnalysis application."""
        try:
            import cv2
            from insightface.app import FaceAnalysis
            providers = get_onnx_providers()
            ctx_id = 0 if "CUDAExecutionProvider" in providers else -1
            logger.info(f"Initializing InsightFace FaceAnalysis ({self.model_pack}) with ctx_id={ctx_id}, providers: {providers}")
            self._app = FaceAnalysis(name=self.model_pack, allowed_modules=['detection'], providers=providers)
            self._app.prepare(ctx_id=ctx_id, det_size=self.det_size, det_thresh=self.conf_threshold)
            logger.info("SCRFD detector successfully initialized.")
        except Exception as e:
            logger.warning(f"SCRFD detector could not be loaded: {e}. Fallback mode active.")
            self._app = None

    def detect_frame(
        self, 
        image_bgr: np.ndarray, 
        frame_index: int = 0, 
        timestamp_sec: float = 0.0
    ) -> List[FaceDetection]:
        """Run SCRFD face detection on a single frame."""
        if self._app is None:
            return []
            
        faces = self._app.get(image_bgr)
        detections: List[FaceDetection] = []
        
        for face in faces:
            conf = float(face.det_score) if hasattr(face, "det_score") else 1.0
            if conf < self.conf_threshold:
                continue
                
            bbox = np.array(face.bbox, dtype=np.float32)
            kps = np.array(face.kps, dtype=np.float32) if hasattr(face, "kps") else None
            
            detections.append(FaceDetection(
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                bbox=bbox,
                confidence=conf,
                landmarks_5=kps
            ))
            
        return detections
