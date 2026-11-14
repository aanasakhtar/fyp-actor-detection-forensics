"""
Facial manipulation and deepfake detection evaluation metrics.
"""

import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

def compute_forensic_metrics(
    y_true: List[int], 
    y_scores: List[float], 
    threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Calculate ROC-AUC, PR-AUC, Accuracy, Precision, Recall, F1, and Confusion Matrix.
    """
    if len(y_true) == 0 or len(set(y_true)) < 2:
        return {
            "roc_auc": 0.0,
            "pr_auc": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
            "sample_count": len(y_true)
        }
        
    y_arr = np.array(y_true)
    score_arr = np.array(y_scores)
    y_pred = (score_arr >= threshold).astype(int)
    
    auc = float(roc_auc_score(y_arr, score_arr))
    pr_auc = float(average_precision_score(y_arr, score_arr))
    acc = float(accuracy_score(y_arr, y_pred))
    prec = float(precision_score(y_arr, y_pred, zero_division=0))
    rec = float(recall_score(y_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_arr, y_pred, zero_division=0))
    cm = confusion_matrix(y_arr, y_pred).tolist()
    
    return {
        "roc_auc": round(auc, 4),
        "pr_auc": round(pr_auc, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "confusion_matrix": cm,
        "sample_count": len(y_true)
    }
