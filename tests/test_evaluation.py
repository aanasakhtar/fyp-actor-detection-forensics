"""
Unit tests for evaluation metrics (forensic metrics, verification metrics, aggregation).
"""

from pipeline.evaluation.forensic_metrics import compute_forensic_metrics
from pipeline.evaluation.recognition_metrics import compute_verification_metrics
from pipeline.forensics.gend_detector import BaseGenDDetector

def test_forensic_metrics_perfect_classifier():
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.2, 0.8, 0.9]
    metrics = compute_forensic_metrics(y_true, y_scores, threshold=0.5)
    
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0

def test_score_aggregation():
    scores = [0.1, 0.1, 0.2, 0.2, 0.9]  # 4 real frames, 1 spurious high frame
    trimmed = BaseGenDDetector.aggregate_scores(scores, method="trimmed_mean", trim_ratio=0.20)
    assert trimmed < 0.50

def test_verification_metrics():
    genuine = [0.85, 0.90, 0.95]
    impostor = [0.10, 0.15, 0.20]
    metrics = compute_verification_metrics(genuine, impostor)
    
    assert metrics["roc_auc"] == 1.0
    assert metrics["eer"] == 0.0
    assert metrics["genuine_mean"] > metrics["impostor_mean"]

def test_detection_metrics_and_geometry():
    import numpy as np
    from pipeline.evaluation.detection_metrics import (
        compute_bbox_iou,
        compute_inter_pupillary_distance,
        compute_landmark_geometric_validity
    )
    # IoU
    box1 = np.array([0, 0, 100, 100], dtype=np.float32)
    box2 = np.array([50, 0, 150, 100], dtype=np.float32)
    iou = compute_bbox_iou(box1, box2)
    assert 0.30 < iou < 0.35

    # IPD
    kps = np.array([
        [30.0, 30.0],  # Left eye
        [70.0, 30.0],  # Right eye
        [50.0, 50.0],  # Nose
        [35.0, 75.0],  # Left mouth
        [65.0, 75.0]   # Right mouth
    ], dtype=np.float32)
    ipd = compute_inter_pupillary_distance(kps)
    assert abs(ipd - 40.0) < 1e-4

    # Geometric validity
    valid = compute_landmark_geometric_validity(kps, box1)
    assert valid is True


def test_sbi_detector_interface():
    from pipeline.forensics.sbi_detector import SBIEfficientNetDetector
    scores = [0.05, 0.08, 0.12, 0.10, 0.95]
    agg = SBIEfficientNetDetector.aggregate_scores(scores, method="trimmed_mean", trim_ratio=0.20)
    assert agg < 0.20

