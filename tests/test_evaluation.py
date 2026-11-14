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
