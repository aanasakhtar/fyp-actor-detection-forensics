"""
Evaluation plots: ROC curves, PR curves, score distributions, and confusion matrices.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Optional
from sklearn.metrics import roc_curve, precision_recall_curve

def plot_roc_curve(y_true: List[int], y_scores: List[float], title: str, save_path: Optional[str] = None) -> None:
    """Plot Receiver Operating Characteristic (ROC) curve."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC Curve")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(cm: List[List[int]], title: str, save_path: Optional[str] = None) -> None:
    """Plot confusion matrix heatmap."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.title(title)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_similarity_distributions(
    genuine: List[float], 
    impostor: List[float], 
    threshold: float, 
    save_path: Optional[str] = None
) -> None:
    """Plot genuine vs impostor cosine similarity distribution histograms."""
    plt.figure(figsize=(8, 5))
    sns.histplot(genuine, color="green", label="Genuine Pairs", kde=True, stat="density", bins=30)
    sns.histplot(impostor, color="red", label="Impostor Pairs", kde=True, stat="density", bins=30)
    plt.axvline(threshold, color="black", linestyle="--", lw=2, label=f"Threshold ({threshold:.2f})")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.title("ArcFace Verification Similarity Distributions")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
