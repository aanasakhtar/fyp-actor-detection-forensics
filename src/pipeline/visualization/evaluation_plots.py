"""
Evaluation plots: ROC curves, PR curves, score distributions, and confusion matrices.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Optional
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

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

def plot_gend_evaluation(
    y_true: List[int],
    y_scores: List[float],
    title: str = "Stage 3: GenD Deepfake Detection Evaluation",
    threshold: float = 0.50,
    save_path: Optional[str] = None
) -> None:
    """Generate a 2-panel supervisor evaluation report: Score Distribution + ROC Curve."""
    if len(y_true) == 0 or len(y_scores) == 0:
        return

    scores = np.array(y_scores)
    labels = np.array(y_true)

    real_scores = scores[labels == 0]
    fake_scores = scores[labels == 1]

    auc = float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else 0.0
    fpr, tpr, _ = roc_curve(labels, scores) if len(set(labels)) > 1 else ([0, 1], [0, 1], None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

    # Panel 1: Prediction Probabilities Histogram
    if len(real_scores) > 0:
        ax1.hist(real_scores, bins=25, alpha=0.6, color="green", label=f"Real Videos (N={len(real_scores)})", density=True)
    if len(fake_scores) > 0:
        ax1.hist(fake_scores, bins=25, alpha=0.6, color="red", label=f"Manipulated Videos (N={len(fake_scores)})", density=True)
    ax1.axvline(threshold, color="black", linestyle="--", lw=2, label=f"Threshold ({threshold:.2f})")
    ax1.set_xlabel("Predicted Probability p(Fake)", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.set_title("Manipulation Probability Distribution", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: ROC Curve
    ax2.plot(fpr, tpr, color="darkorange", lw=2.5, label=f"ROC Curve (AUC = {auc:.4f})")
    ax2.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Chance")
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.02])
    ax2.set_xlabel("False Positive Rate (FPR)", fontsize=11)
    ax2.set_ylabel("True Positive Rate (TPR)", fontsize=11)
    ax2.set_title(f"GenD ROC Curve (AUC = {auc:.4f})", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

def plot_scrfd_cross_dataset_evaluation(
    dataset_metrics: dict,
    title: str = "Stage 1: SCRFD Cross-Dataset Face Detection Forensic Benchmark",
    save_path: Optional[str] = None
) -> None:
    """
    Generate a 4-panel diagnostic figure evaluating SCRFD face detection across FF++ and Celeb-DF:
    - Panel 1: Face Coverage Rate (%) by Category
    - Panel 2: Detection Confidence (p_det) Distribution
    - Panel 3: Inter-Pupillary Distance (IPD / Resolution)
    - Panel 4: Temporal IoU Tracking Stability
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    ax1, ax2, ax3, ax4 = axes.flatten()

    # Panel 1: Face Coverage Rate by Category
    cat_names = []
    cat_coverages = []
    cat_colors = []
    for d_name, d_res in dataset_metrics.items():
        by_cat = d_res.get("category_coverage", {})
        for c, cov in by_cat.items():
            cat_names.append(f"{c}\n({d_name})")
            cat_coverages.append(cov * 100.0)
            cat_colors.append("#2ca02c" if "real" in c.lower() or "original" in c.lower() else "#d62728")

    if cat_names:
        bars = ax1.bar(range(len(cat_names)), cat_coverages, color=cat_colors, alpha=0.85, edgecolor="black")
        ax1.set_xticks(range(len(cat_names)))
        ax1.set_xticklabels(cat_names, rotation=35, ha="right", fontsize=9)
        ax1.set_ylabel("Face Detection Coverage (%)", fontsize=10)
        ax1.set_title("Frame-Level Face Coverage by Category", fontsize=11, fontweight="bold")
        ax1.set_ylim([0, 105])
        ax1.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, cat_coverages):
            ax1.text(bar.get_x() + bar.get_width()/2, val + 1.5, f"{val:.1f}%", ha="center", fontsize=8, fontweight="bold")

    # Panel 2: Detection Confidence Distribution
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for idx, (d_name, d_res) in enumerate(dataset_metrics.items()):
        confs = d_res.get("confidences", [])
        if confs:
            sns.kdeplot(confs, ax=ax2, label=f"{d_name} (Mean: {np.mean(confs):.3f})", color=colors[idx % len(colors)], lw=2.5)
    ax2.set_xlabel("Detection Confidence Score p(det)", fontsize=10)
    ax2.set_ylabel("Density", fontsize=10)
    ax2.set_title("SCRFD Confidence Score Distribution", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    # Panel 3: Face Resolution / Inter-Pupillary Distance
    for idx, (d_name, d_res) in enumerate(dataset_metrics.items()):
        ipds = d_res.get("ipds", [])
        if ipds:
            sns.kdeplot(ipds, ax=ax3, label=f"{d_name} (Mean IPD: {np.mean(ipds):.1f}px)", color=colors[idx % len(colors)], lw=2.5)
    ax3.axvline(30.0, color="red", linestyle="--", lw=1.5, label="ArcFace Min Resolution (30px)")
    ax3.set_xlabel("Inter-Pupillary Distance (IPD in pixels)", fontsize=10)
    ax3.set_ylabel("Density", fontsize=10)
    ax3.set_title("Face Scale & Resolution (IPD)", fontsize=11, fontweight="bold")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    # Panel 4: Temporal Bounding Box Stability (IoU)
    for idx, (d_name, d_res) in enumerate(dataset_metrics.items()):
        ious = d_res.get("ious", [])
        if ious:
            sns.kdeplot(ious, ax=ax4, label=f"{d_name} (Mean IoU: {np.mean(ious):.3f})", color=colors[idx % len(colors)], lw=2.5)
    ax4.set_xlabel("Consecutive Frame Bounding Box IoU", fontsize=10)
    ax4.set_ylabel("Density", fontsize=10)
    ax4.set_title("Temporal Tracking Stability (Jitter Analysis)", fontsize=11, fontweight="bold")
    ax4.legend(loc="upper left")
    ax4.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
