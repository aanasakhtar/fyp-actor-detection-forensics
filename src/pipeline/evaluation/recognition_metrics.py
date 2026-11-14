"""
Biometric face verification and recognition evaluation metrics.
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import roc_curve, roc_auc_score

def compute_verification_metrics(
    genuine_scores: List[float], 
    impostor_scores: List[float]
) -> Dict[str, float]:
    """
    Compute TAR, FAR, FRR, EER and ROC-AUC from genuine and impostor similarity scores.
    """
    if not genuine_scores or not impostor_scores:
        return {"roc_auc": 0.0, "eer": 1.0, "optimal_threshold": 0.50}
        
    y_true = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    y_scores = np.concatenate([genuine_scores, impostor_scores])
    
    auc = float(roc_auc_score(y_true, y_scores))
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1.0 - tpr
    
    # Calculate Equal Error Rate (EER) where FPR ~= FNR
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    opt_threshold = float(thresholds[eer_idx])
    
    return {
        "roc_auc": auc,
        "eer": eer,
        "optimal_threshold": opt_threshold,
        "genuine_mean": float(np.mean(genuine_scores)),
        "genuine_std": float(np.std(genuine_scores)),
        "impostor_mean": float(np.mean(impostor_scores)),
        "impostor_std": float(np.std(impostor_scores))
    }
