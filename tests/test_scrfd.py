"""
Unit tests for face detection data structures, IoU calculation, and tracking.
"""

import numpy as np
from pipeline.detection.detection_utils import FaceDetection, compute_iou
from pipeline.tracking.face_tracking import IoUFaceTracker
from pipeline.preprocessing.face_cropping import crop_context_face

def test_compute_iou():
    box1 = np.array([0, 0, 10, 10])
    box2 = np.array([0, 0, 10, 10])
    assert compute_iou(box1, box2) == 1.0
    
    box3 = np.array([10, 10, 20, 20])
    assert compute_iou(box1, box3) == 0.0
    
    box4 = np.array([5, 0, 15, 10])
    # Inter: [5, 0, 10, 10] -> area 50; Union: 100 + 100 - 50 = 150 -> IoU = 50/150 = 1/3
    assert abs(compute_iou(box1, box4) - (1.0 / 3.0)) < 1e-4

def test_face_tracker_association():
    tracker = IoUFaceTracker(iou_threshold=0.3)
    
    # Frame 0
    det0 = [FaceDetection(frame_index=0, timestamp_sec=0.0, bbox=np.array([10, 10, 50, 50]), confidence=0.9)]
    tracked0 = tracker.update(det0, current_frame=0)
    assert tracked0[0].track_id == 1
    
    # Frame 1: slightly moved box
    det1 = [FaceDetection(frame_index=1, timestamp_sec=0.04, bbox=np.array([12, 12, 52, 52]), confidence=0.9)]
    tracked1 = tracker.update(det1, current_frame=1)
    assert tracked1[0].track_id == 1

def test_crop_context_face():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    bbox = np.array([50, 50, 100, 100])
    crop = crop_context_face(img, bbox, expansion_factor=1.3, target_size=(224, 224))
    assert crop is not None
    assert crop.shape == (224, 224, 3)
