"""
Fast Offline Recalibration Script for Full Celeb-DF v2 Benchmark.

Loads pre-computed video predictions directly from 'celebdf_full_checkpoint.jsonl'
and evaluates multiple thresholding strategies on the Held-Out Test split in 1-2 seconds
without re-running neural network inference:
  1. Default Naive Baseline (T = 0.50)
  2. Balanced Accuracy (Youden's J: TPR - FPR)
  3. F2 Score Optimization (Recall prioritized 4x over Precision)
  4. Constrained Target-Recall: Recall >= 95% (Minimum FPR satisfying >= 95% Recall)
  5. Constrained Target-Recall: Recall >= 98% (Minimum FPR satisfying >= 98% Recall)
  6. Constrained Target-Recall: Recall >= 99% (Near-Zero Missed Deepfakes)
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
from sklearn.metrics import roc_curve, precision_recall_curve

# Ensure local 'src' directory takes priority in sys.path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from pipeline.evaluation.forensic_metrics import compute_forensic_metrics
from pipeline.utils.logging import get_logger

logger = get_logger("recalibrate_celebdf")


def find_threshold_by_strategy(
    y_true: List[int],
    scores: List[float],
    strategy: str,
    target_recall: float = 0.98,
    beta: float = 2.0
) -> Tuple[float, str]:
    """Find decision threshold on training split based on the specified objective."""
    if len(set(y_true)) < 2:
        return 0.50, "Default"

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    y_arr = np.array(y_true)
    sc_arr = np.array(scores)

    if strategy.startswith("recall_"):
        t_target = target_recall
        # Find all thresholds where TPR >= t_target
        valid_indices = np.where(tpr >= t_target)[0]
        if len(valid_indices) > 0:
            # First valid index corresponds to highest threshold (lowest FPR) since thresholds descend
            best_idx = valid_indices[0]
            opt_t = float(thresholds[best_idx])
            desc = f"Target Recall >= {t_target*100:.0f}%"
        else:
            best_idx = int(np.argmax(tpr - fpr))
            opt_t = float(thresholds[best_idx])
            desc = "Fallback Youden"

    elif strategy in ["f2", "f_beta"]:
        b2 = beta ** 2
        best_f = -1.0
        best_t = 0.50
        cand_thresh = [t for t in thresholds if 0.05 <= t <= 0.98]
        for t in (cand_thresh or [0.50]):
            preds = (sc_arr >= t).astype(int)
            tp = np.sum((preds == 1) & (y_arr == 1))
            fp = np.sum((preds == 1) & (y_arr == 0))
            fn = np.sum((preds == 0) & (y_arr == 1))
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            denom = b2 * prec + rec
            f_val = (1 + b2) * (prec * rec) / denom if denom > 0 else 0.0
            if f_val > best_f:
                best_f = f_val
                best_t = t
        opt_t = float(best_t)
        desc = f"Max F_{beta:.0f} (Recall-Weighted)"

    elif strategy in ["raw_acc", "accuracy"]:
        best_acc = -1.0
        best_t = 0.50
        cand_thresh = [t for t in thresholds if 0.05 <= t <= 0.98]
        for t in (cand_thresh or [0.50]):
            acc = np.mean((sc_arr >= t).astype(int) == y_arr)
            if acc > best_acc:
                best_acc = acc
                best_t = t
        opt_t = float(best_t)
        desc = "Max Raw Accuracy"

    else:  # youden
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        opt_t = float(thresholds[best_idx])
        desc = "Youden's J (Balanced)"

    return round(max(0.05, min(0.96, opt_t)), 4), desc


def evaluate_at_threshold(y_true: List[int], scores: List[float], threshold: float) -> Dict[str, Any]:
    """Compute comprehensive metrics including Balanced Accuracy at a fixed threshold."""
    m_dict = compute_forensic_metrics(y_true, scores, threshold=threshold)
    cm = m_dict.get("confusion_matrix", [[0, 0], [0, 0]])
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]
    fpr = fp / max(1, (fp + tn))
    tpr = tp / max(1, (tp + fn))
    tnr = tn / max(1, (tn + fp))
    bal_acc = 0.5 * (tpr + tnr)
    m_dict["fpr_on_real"] = round(float(fpr), 4)
    m_dict["specificity"] = round(float(tnr), 4)
    m_dict["balanced_accuracy"] = round(float(bal_acc), 4)
    m_dict["threshold"] = round(float(threshold), 4)
    return m_dict


def plot_strategy_comparison(
    comparison_results: List[Dict[str, Any]],
    model_name: str,
    save_path: str
):
    """Plot Recall vs FPR trade-off across strategies for the champion model."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
    strats = [r["strategy_label"] for r in comparison_results]
    recalls = [r["recall"] * 100.0 for r in comparison_results]
    fprs = [r["fpr_on_real"] * 100.0 for r in comparison_results]
    thresholds = [r["learned_threshold"] for r in comparison_results]
    x = np.arange(len(strats))

    # Panel 1: Recall vs FPR
    w = 0.35
    b1 = ax1.bar(x - w/2, recalls, w, label="Recall (Deepfake Catch Rate %)", color="#D32F2F", alpha=0.88)
    b2 = ax1.bar(x + w/2, fprs, w, label="FPR (False Alarms on Real %)", color="#1976D2", alpha=0.88)
    for r in b1:
        h = r.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for r in b2:
        h = r.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(strats, rotation=20, ha="right", fontweight="bold", fontsize=9)
    ax1.set_ylim([0.0, 115.0])
    ax1.set_ylabel("Percentage (%)", fontweight="bold")
    ax1.set_title(f"HELD-OUT TEST: RECALL vs. FALSE ALARMS ({model_name})", fontweight="bold", pad=10)
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel 2: Learned Thresholds across strategies
    bars3 = ax2.bar(x, thresholds, 0.45, color="#388E3C", alpha=0.88)
    for r in bars3:
        h = r.get_height()
        ax2.annotate(f"{h:.3f}", xy=(r.get_x() + r.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax2.axhline(0.50, color="black", linestyle=":", lw=1.5, label="Default 0.50 Cutoff")
    ax2.set_xticks(x)
    ax2.set_xticklabels(strats, rotation=20, ha="right", fontweight="bold", fontsize=9)
    ax2.set_ylim([0.0, 1.10])
    ax2.set_ylabel("Learned Decision Threshold T*", fontweight="bold")
    ax2.set_title("LEARNED TRAIN THRESHOLD BY OPTIMIZATION OBJECTIVE", fontweight="bold", pad=10)
    ax2.legend(loc="upper left")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle(f"THRESHOLD RECALIBRATION: HIGH-RECALL FORENSIC PROFILES ({model_name})", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Instantly evaluate multiple threshold optimization strategies from checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="/kaggle/working/results/celebdf_full_benchmark/celebdf_full_checkpoint.jsonl",
                        help="Path to precomputed celebdf_full_checkpoint.jsonl.")
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/results/celebdf_full_benchmark",
                        help="Output directory to save comparison report and figures.")
    args = parser.parse_args()

    # Fallback search if path doesn't exist
    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        candidates = [
            ckpt_path,
            "./celebdf_full_checkpoint.jsonl",
            "results/celebdf_full_benchmark/celebdf_full_checkpoint.jsonl",
            "/kaggle/working/celebdf_full_checkpoint.jsonl"
        ]
        for c in candidates:
            if os.path.exists(c):
                ckpt_path = c
                break

    if not os.path.exists(ckpt_path):
        logger.error(f"Checkpoint file '{args.checkpoint}' not found. Please provide valid path.")
        return

    # Load checkpoint
    train_records = []
    test_records = []
    with open(ckpt_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("split") == "train":
                train_records.append(r)
            else:
                test_records.append(r)

    print("\n" + "="*128)
    print("      CELEB-DF v2 HIGH-RECALL THRESHOLD RECALIBRATION ANALYSIS")
    print("="*128)
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Train / Calibration Split: {len(train_records)} videos")
    print(f"Held-Out Test Split:       {len(test_records)} videos\n")

    train_y = [1 if r["is_fake"] else 0 for r in train_records]
    test_y = [1 if r["is_fake"] else 0 for r in test_records]
    models = list(train_records[0]["scores"].keys())

    train_scores = {m: [r["scores"].get(m, 0.0) for r in train_records] for m in models}
    test_scores = {m: [r["scores"].get(m, 0.0) for r in test_records] for m in models}

    # Strategies to compare
    strategies = [
        ("Default (T = 0.50)", "default", 0.50),
        ("Youden's J (Balanced)", "youden", 0.0),
        ("Max F2 Score", "f2", 0.0),
        ("Constrained Recall >= 95%", "recall_95", 0.95),
        ("Constrained Recall >= 98%", "recall_98", 0.98),
        ("Constrained Recall >= 99%", "recall_99", 0.99),
    ]

    all_comparison = {}
    for m in models:
        all_comparison[m] = []
        for label, strat_key, param in strategies:
            if strat_key == "default":
                learned_t = 0.50
                desc = "Fixed 0.50 Cutoff"
            else:
                learned_t, desc = find_threshold_by_strategy(
                    train_y, train_scores[m], strategy=strat_key, target_recall=param
                )

            # Evaluate on held-out test split
            test_m = evaluate_at_threshold(test_y, test_scores[m], learned_t)
            res = {
                "model": m,
                "strategy_label": label,
                "strategy_key": strat_key,
                "description": desc,
                "learned_threshold": learned_t,
                "accuracy": test_m["accuracy"],
                "balanced_accuracy": test_m["balanced_accuracy"],
                "recall": test_m["recall"],
                "precision": test_m["precision"],
                "fpr_on_real": test_m["fpr_on_real"],
                "f1": test_m.get("f1", 0.0)
            }
            all_comparison[m].append(res)

    # Print Comparison Tables per Model
    for m in models:
        print("="*128)
        print(f"   HELD-OUT TEST EVALUATION ACROSS THRESHOLD STRATEGIES: {m.upper()}")
        print("="*128)
        print(f"{'Strategy / Objective':28s} | {'Learned T*':10s} | {'Recall':7s} | {'FPR (Real)':10s} | {'Accuracy':8s} | {'Bal Acc':8s} | {'Precision':9s} | {'F1':7s}")
        print("-" * 128)
        for r in all_comparison[m]:
            highlight = " <-- TARGET" if "98%" in r["strategy_label"] or "95%" in r["strategy_label"] else ""
            print(f"{r['strategy_label']:28s} | {r['learned_threshold']:10.3f} | {r['recall']*100:5.2f}% | {r['fpr_on_real']*100:8.2f}% | {r['accuracy']*100:6.2f}% | {r['balanced_accuracy']*100:6.2f}% | {r['precision']*100:7.2f}% | {r['f1']:7.4f}{highlight}")
        print("="*128 + "\n")

    # Plot chart for top model (GenD CLIP-L or GenD PE-L)
    top_model = "GenD (CLIP-L)" if "GenD (CLIP-L)" in all_comparison else models[0]
    chart_path = os.path.join(args.output_dir, "recalibrated_strategies_comparison.png")
    plot_strategy_comparison(all_comparison[top_model], top_model, chart_path)

    # Save JSON summary
    json_path = os.path.join(args.output_dir, "recalibrated_strategies_summary.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(all_comparison, jf, indent=2)

    print(f"[+] Saved Recalibration Summary JSON: {json_path}")
    print(f"[+] Saved Strategy Comparison Chart:  {chart_path}\n")


if __name__ == "__main__":
    main()
