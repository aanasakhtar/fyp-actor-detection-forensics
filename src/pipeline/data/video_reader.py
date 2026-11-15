"""
Video decoding and temporal frame extraction.
"""

import os
import numpy as np
from dataclasses import dataclass
from typing import List, Generator, Optional
from ..utils.logging import get_logger

logger = get_logger("video_reader")

@dataclass
class SampledFrame:
    """Metadata container for an extracted video frame."""
    frame_index: int
    timestamp_sec: float
    image_bgr: np.ndarray

class VideoReader:
    """Decodes video files and performs uniform temporal frame sampling."""
    
    def __init__(self, video_path: str):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        self.video_path = video_path
        
        try:
            import cv2
            self._cap = cv2.VideoCapture(video_path)
            if not self._cap.isOpened():
                raise ValueError(f"Unable to open video: {video_path}")
            self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = float(self._cap.get(cv2.CAP_PROP_FPS))
            if self.fps <= 0:
                self.fps = 25.0
            self.duration_sec = self.total_frames / self.fps if self.total_frames > 0 else 0.0
            self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._cap.release()
        except ImportError:
            self.total_frames = 0
            self.fps = 25.0
            self.duration_sec = 0.0
            self.width = 0
            self.height = 0
        
    def sample_uniform(self, max_frames: int = 32) -> List[SampledFrame]:
        """Sample frames uniformly throughout the video duration."""
        try:
            import cv2
        except ImportError:
            logger.error("cv2 is required for video frame sampling.")
            return []
            
        if self.total_frames <= 0:
            return []
            
        if self.total_frames <= max_frames:
            target_indices = list(range(self.total_frames))
        else:
            target_indices = np.linspace(0, self.total_frames - 1, max_frames, dtype=int).tolist()
            
        target_set = set(target_indices)
        frames: List[SampledFrame] = []
        
        cap = cv2.VideoCapture(self.video_path)
        curr_idx = 0
        while cap.isOpened() and len(frames) < len(target_indices):
            ret, frame = cap.read()
            if not ret:
                break
            if curr_idx in target_set:
                ts = curr_idx / self.fps
                frames.append(SampledFrame(frame_index=curr_idx, timestamp_sec=ts, image_bgr=frame))
            curr_idx += 1
            
        cap.release()
        return frames
