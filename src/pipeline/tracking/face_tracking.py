"""
Lightweight IoU Face Tracker for associating face detections across video frames.
"""

import numpy as np
from typing import List, Dict
from ..detection.detection_utils import FaceDetection, compute_iou

class IoUFaceTracker:
    """Tracks face instances across sampled frames using bounding-box spatial overlap."""
    
    def __init__(self, iou_threshold: float = 0.40, max_lost_frames: int = 5):
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.next_track_id = 1
        self.active_tracks: Dict[int, Dict] = {}  # track_id -> {last_bbox, last_frame}
        
    def update(self, detections: List[FaceDetection], current_frame: int) -> List[FaceDetection]:
        """
        Associate detections to active tracks based on IoU.
        Assigns track_id in-place on detections.
        """
        if not detections:
            # Purge stale tracks
            self._purge_stale(current_frame)
            return detections
            
        unmatched_detections = list(range(len(detections)))
        matched_tracks = set()
        
        # Try matching with active tracks
        for track_id, track_data in list(self.active_tracks.items()):
            best_iou = -1.0
            best_det_idx = -1
            
            for d_idx in unmatched_detections:
                iou = compute_iou(track_data["last_bbox"], detections[d_idx].bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = d_idx
                    
            if best_iou >= self.iou_threshold and best_det_idx != -1:
                detections[best_det_idx].track_id = track_id
                self.active_tracks[track_id] = {
                    "last_bbox": detections[best_det_idx].bbox,
                    "last_frame": current_frame
                }
                matched_tracks.add(track_id)
                unmatched_detections.remove(best_det_idx)
                
        # Create new tracks for unmatched detections
        for d_idx in unmatched_detections:
            new_id = self.next_track_id
            self.next_track_id += 1
            detections[d_idx].track_id = new_id
            self.active_tracks[new_id] = {
                "last_bbox": detections[d_idx].bbox,
                "last_frame": current_frame
            }
            
        self._purge_stale(current_frame)
        return detections

    def _purge_stale(self, current_frame: int) -> None:
        """Remove tracks that have not been observed for more than max_lost_frames."""
        stale_ids = [
            t_id for t_id, data in self.active_tracks.items()
            if (current_frame - data["last_frame"]) > self.max_lost_frames
        ]
        for t_id in stale_ids:
            del self.active_tracks[t_id]
