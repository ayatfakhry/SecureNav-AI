"""
evaluation.py
─────────────
Comprehensive evaluation suite for SecureNav AI classifiers.

Produces
────────
  • Per-class and macro/weighted precision, recall, F1, support
  • Confusion matrix (raw counts + normalised)
  • ROC curves and AUC (one-vs-rest for multi-class)
  • Cross-validation summary table
  • Model comparison DataFrame
  • Text report (saved to results/)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelBinarizer

from src.feature_extraction import CLASS_NAMES


# ──────────────────────────────────────────────────────────────
# Core metrics
# ──────────────────────────────────────────────────────────────
def compute_metrics(
    y_true      : np.ndarray,
    y_pred      : np.ndarray,
    class_names : List[str] = CLASS_NAMES,
) -> Dict:
    """
    Compute a full set of classification metrics.

    Returns
    -------
    dict with keys: accuracy, report_str, report_df, confusion_matrix,
                    confusion_matrix_norm, per_class
    """
    acc = float(accuracy_score(y_true, y_pred))

    # Unique labels present in predictions / truth
    labels = sorted(set(y_true) | set(y_pred))
    names  = [class_names[i] if i < len(class_names) else str(i) for i in labels]

    report_str = classification_report(
        y_true, y_pred,
        labels      = labels,
        target_names= names,
        digits      = 4,
        zero_division= 0,
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred,
        labels      = labels,
        zero_division= 0,
    )

    report_df = pd.DataFrame({
        "class"    : names,
        "precision": precision,
        "recall"   : recall,
        "f1_score" : f1,
        "support"  : support,
    })

    cm      = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    per_class = {
        names[i]: {
            "precision": float(precision[i]),
            "recall"   : float(recall[i]),
            "f1_score" : float(f1[i]),
            "support"  : int(support[i]),
        }
        for i in range(len(names))
    }

    # Macro and weighted averages
    macro_f1    = float(np.mean(f1))
    weighted_f1 = float(np.average(f1, weights=support))
    macro_prec  = float(np.mean(precision))
    macro_rec   = float(np.mean(recall))

    return {
        "accuracy"           : acc,
        "macro_f1"           : macro_f1,
        "weighted_f1"        : weighted_f1,
        "macro_precision"    : macro_prec,
        "macro_recall"       : macro_rec,
        "report_str"         : report_str,
        "report_df"          : report_df,
        "confusion_matrix"   : cm,
        "confusion_matrix_norm": cm_norm,
        "class_names_present": names,
        "per_class"          : per_class,
        "labels"             : labels,
    }


# ──────────────────────────────────────────────────────────────
# ROC / AUC
# ──────────────────────────────────────────────────────────────
def compute_roc(
    y_true      : np.ndarray,
    y_prob      : np.ndarray,
    class_names : List[str] = CLASS_NAMES,
) -> Dict:
    """
    Compute one-vs-rest ROC curves and AUC for each class.

    Parameters
    ----------
    y_true : integer label array
    y_prob : (n_samples, n_classes) probability array

    Returns
    -------
    dict mapping class name → {'fpr', 'tpr', 'auc'}
    """
    n_classes = y_prob.shape[1]
    lb        = LabelBinarizer().fit(list(range(n_classes)))
    y_bin     = lb.transform(y_true)

    # Handle binary edge case
    if y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])

    roc_data = {}
    for i in range(min(n_classes, len(class_names))):
        if i >= y_bin.shape[1]:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc         = roc_auc_score(y_bin[:, i], y_prob[:, i])
        roc_data[class_names[i]] = {"fpr": fpr, "tpr": tpr, "auc": float(auc)}

    return roc_data


# ──────────────────────────────────────────────────────────────
# Cross-validation summary
# ──────────────────────────────────────────────────────────────
def summarise_cv(cv_results: Dict, model_name: str = "") -> pd.DataFrame:
    """
    Convert cross_validate_model output to a tidy DataFrame.
    """
    rows = []
    for metric, stats in cv_results.items():
        rows.append({
            "model"  : model_name,
            "metric" : metric,
            "mean"   : round(stats["mean"], 4),
            "std"    : round(stats["std"],  4),
            "scores" : stats["scores"],
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# Model comparison table
# ──────────────────────────────────────────────────────────────
def compare_models(results_dict: Dict[str, Dict]) -> pd.DataFrame:
    """
    Build a single comparison DataFrame from per-model metrics dicts.

    Parameters
    ----------
    results_dict : { model_name: metrics_dict from compute_metrics() }
    """
    rows = []
    for name, m in results_dict.items():
        rows.append({
            "model"           : name,
            "accuracy"        : round(m["accuracy"],        4),
            "macro_f1"        : round(m["macro_f1"],        4),
            "weighted_f1"     : round(m["weighted_f1"],     4),
            "macro_precision" : round(m["macro_precision"], 4),
            "macro_recall"    : round(m["macro_recall"],    4),
        })
    df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────────
# Text report writer
# ──────────────────────────────────────────────────────────────
def save_report(
    metrics     : Dict,
    model_name  : str,
    cv_summary  : Optional[pd.DataFrame] = None,
    output_dir  : str | Path = "results",
) -> Path:
    """
    Write a human-readable evaluation report to a .txt file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"report_{model_name}.txt"

    sep  = "=" * 70
    sep2 = "-" * 70

    lines = [
        sep,
        f"  SecureNav AI — Evaluation Report",
        f"  Model: {model_name.upper()}",
        sep,
        "",
        f"  Accuracy         : {metrics['accuracy']:.4f}",
        f"  Macro F1         : {metrics['macro_f1']:.4f}",
        f"  Weighted F1      : {metrics['weighted_f1']:.4f}",
        f"  Macro Precision  : {metrics['macro_precision']:.4f}",
        f"  Macro Recall     : {metrics['macro_recall']:.4f}",
        "",
        sep2,
        "  Per-Class Metrics",
        sep2,
        metrics["report_str"],
    ]

    if cv_summary is not None:
        lines += [
            sep2,
            "  Cross-Validation Summary (k=5)",
            sep2,
            cv_summary.to_string(index=False),
            "",
        ]

    lines += [sep, ""]

    path.write_text("\n".join(lines))
    print(f"  [✓] Report saved → {path}")
    return path
