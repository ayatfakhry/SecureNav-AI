"""
visualization.py
────────────────
Publication-quality plots for SecureNav AI.

All functions accept pre-computed data structures and return
Matplotlib Figure objects (or save directly).

Plot catalogue
──────────────
  1. plot_signal_overview      – SNR / satellite count / PDOP time series
  2. plot_confusion_matrix     – Annotated heatmap (raw + normalised)
  3. plot_feature_importance   – Horizontal bar chart (RF feature importance)
  4. plot_roc_curves           – Multi-class ROC / AUC overlay
  5. plot_pca_scatter          – 2-D PCA projection coloured by class
  6. plot_position_trajectory  – Lat/lon scatter per class
  7. plot_model_comparison     – Grouped bar chart of accuracy/F1 per model
  8. plot_anomaly_scores       – Time-series anomaly score with threshold line
  9. save_dashboard            – Saves a multi-panel summary figure
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")       # non-interactive backend for server / CI use
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.decomposition  import PCA
from sklearn.preprocessing  import StandardScaler

from src.feature_extraction import FEATURE_COLUMNS, CLASS_NAMES


# ──────────────────────────────────────────────────────────────
# Style
# ──────────────────────────────────────────────────────────────
CLASS_COLORS = {
    "NORMAL"   : "#2ecc71",
    "SPOOFING" : "#e74c3c",
    "JAMMING"  : "#f39c12",
    "DRIFT"    : "#3498db",
}

PALETTE = list(CLASS_COLORS.values())

sns.set_theme(style="darkgrid", palette=PALETTE, font_scale=1.1)
plt.rcParams.update({
    "figure.facecolor" : "#0f1117",
    "axes.facecolor"   : "#1a1d27",
    "axes.edgecolor"   : "#3d4156",
    "axes.labelcolor"  : "#c8cdd8",
    "xtick.color"      : "#c8cdd8",
    "ytick.color"      : "#c8cdd8",
    "text.color"       : "#e0e4ef",
    "grid.color"       : "#2a2e3e",
    "grid.linewidth"   : 0.6,
    "legend.facecolor" : "#1a1d27",
    "legend.edgecolor" : "#3d4156",
})

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────
# 1. Signal overview
# ──────────────────────────────────────────────────────────────
def plot_signal_overview(
    df         : pd.DataFrame,
    max_epochs : int  = 500,
    save_path  : Optional[Path] = None,
) -> plt.Figure:
    """Plot SNR mean, satellite count, and PDOP over time per class."""
    sample = df.head(max_epochs).copy()
    t      = np.arange(len(sample))
    labels = sample["label"].values

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("GNSS Signal Overview", fontsize=15, fontweight="bold", y=0.98)

    metrics = [
        ("snr_mean",       "SNR Mean (dB-Hz)", (0, 60)),
        ("num_satellites", "Visible Satellites", (0, 16)),
        ("pdop",           "PDOP",              (0, 25)),
    ]

    for ax, (col, ylabel, ylim) in zip(axes, metrics):
        vals = sample[col].values
        # Background shading per class
        prev_label = labels[0]
        start      = 0
        for i, lbl in enumerate(labels):
            if lbl != prev_label or i == len(labels) - 1:
                ax.axvspan(start, i, alpha=0.12,
                           color=CLASS_COLORS.get(prev_label, "#888"))
                start      = i
                prev_label = lbl

        ax.plot(t, vals, lw=0.9, color="#c8cdd8", alpha=0.9)

        # Coloured dots per class
        for cls, col_hex in CLASS_COLORS.items():
            mask = labels == cls
            ax.scatter(t[mask], vals[mask], s=8, c=col_hex,
                       label=cls, alpha=0.7, zorder=3)

        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(ylim)

    axes[-1].set_xlabel("Epoch Index", fontsize=10)

    # Single legend
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=c, markersize=8, label=n)
               for n, c in CLASS_COLORS.items()]
    fig.legend(handles=handles, loc="upper right", ncol=4,
               framealpha=0.9, fontsize=9)
    fig.tight_layout()

    _save(fig, save_path or OUTPUT_DIR / "signal_overview.png")
    return fig


# ──────────────────────────────────────────────────────────────
# 2. Confusion matrix
# ──────────────────────────────────────────────────────────────
def plot_confusion_matrix(
    cm          : np.ndarray,
    class_names : List[str]  = CLASS_NAMES,
    normalised  : bool       = False,
    title       : str        = "Confusion Matrix",
    save_path   : Optional[Path] = None,
) -> plt.Figure:
    """Annotated heatmap confusion matrix."""
    cmap = LinearSegmentedColormap.from_list(
        "securenav", ["#1a1d27", "#1e3a5f", "#2980b9", "#27ae60"], N=256
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    matrix  = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9) if normalised else cm
    fmt     = ".2f" if normalised else "d"

    sns.heatmap(
        matrix,
        annot        = True,
        fmt          = fmt,
        cmap         = cmap,
        xticklabels  = class_names,
        yticklabels  = class_names,
        linewidths   = 0.5,
        linecolor    = "#2a2e3e",
        ax           = ax,
        cbar_kws     = {"shrink": 0.8},
    )
    ax.set_title(f"{title}{'  (normalised)' if normalised else ''}", fontsize=13, pad=12)
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)
    fig.tight_layout()

    _save(fig, save_path or OUTPUT_DIR / f"confusion_matrix{'_norm' if normalised else ''}.png")
    return fig


# ──────────────────────────────────────────────────────────────
# 3. Feature importance
# ──────────────────────────────────────────────────────────────
def plot_feature_importance(
    importances : pd.Series,
    top_n       : int = 20,
    save_path   : Optional[Path] = None,
) -> plt.Figure:
    """Horizontal bar chart of top-N feature importances."""
    top  = importances.head(top_n)
    cmap = plt.cm.get_cmap("RdYlGn", len(top))
    colors = [cmap(i / len(top)) for i in range(len(top))]

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1], edgecolor="none")

    ax.set_xlabel("Importance Score", fontsize=10)
    ax.set_title("Top Feature Importances (Random Forest)", fontsize=13, pad=10)
    ax.set_xlim(0, top.values.max() * 1.18)

    for bar, val in zip(bars, top.values[::-1]):
        ax.text(bar.get_width() + top.values.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)

    fig.tight_layout()
    _save(fig, save_path or OUTPUT_DIR / "feature_importance.png")
    return fig


# ──────────────────────────────────────────────────────────────
# 4. ROC curves
# ──────────────────────────────────────────────────────────────
def plot_roc_curves(
    roc_data    : Dict,
    save_path   : Optional[Path] = None,
) -> plt.Figure:
    """One-vs-rest ROC curves for all classes."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot([0, 1], [0, 1], "--", color="#555", lw=1, label="Random (AUC=0.50)")

    for cls, data in roc_data.items():
        color = CLASS_COLORS.get(cls, "#aaa")
        ax.plot(data["fpr"], data["tpr"], lw=2, color=color,
                label=f"{cls}  (AUC = {data['auc']:.4f})")

    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title("ROC Curves — One-vs-Rest (Multi-class)", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    fig.tight_layout()

    _save(fig, save_path or OUTPUT_DIR / "roc_curves.png")
    return fig


# ──────────────────────────────────────────────────────────────
# 5. PCA scatter
# ──────────────────────────────────────────────────────────────
def plot_pca_scatter(
    X           : np.ndarray,
    y_labels    : np.ndarray,
    class_names : List[str]  = CLASS_NAMES,
    save_path   : Optional[Path] = None,
) -> plt.Figure:
    """2-D PCA projection coloured by class label."""
    scaler   = StandardScaler()
    pca      = PCA(n_components=2, random_state=42)
    X_scaled = scaler.fit_transform(X)
    X_pca    = pca.fit_transform(X_scaled)

    fig, ax  = plt.subplots(figsize=(9, 7))
    for i, cls in enumerate(class_names):
        if i >= len(class_names):
            break
        mask = y_labels == i
        if not np.any(mask):
            continue
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=CLASS_COLORS.get(cls, "#aaa"),
                   s=14, alpha=0.55, label=cls, edgecolors="none")

    var_exp = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1  ({var_exp[0]*100:.1f}% var)", fontsize=10)
    ax.set_ylabel(f"PC2  ({var_exp[1]*100:.1f}% var)", fontsize=10)
    ax.set_title("PCA Feature Space — Class Separation", fontsize=13)
    ax.legend(fontsize=9, markerscale=2)
    fig.tight_layout()

    _save(fig, save_path or OUTPUT_DIR / "pca_scatter.png")
    return fig


# ──────────────────────────────────────────────────────────────
# 6. Position trajectory
# ──────────────────────────────────────────────────────────────
def plot_position_trajectory(
    df          : pd.DataFrame,
    save_path   : Optional[Path] = None,
) -> plt.Figure:
    """Lat/Lon scatter coloured by label."""
    fig, ax = plt.subplots(figsize=(9, 7))

    for cls, color in CLASS_COLORS.items():
        mask = df["label"] == cls
        if not mask.any():
            continue
        ax.scatter(df.loc[mask, "lon"], df.loc[mask, "lat"],
                   s=12, c=color, alpha=0.55, label=cls, edgecolors="none")

    ax.set_xlabel("Longitude (°)", fontsize=10)
    ax.set_ylabel("Latitude (°)",  fontsize=10)
    ax.set_title("Position Trajectory by Class", fontsize=13)
    ax.legend(fontsize=9, markerscale=2)
    fig.tight_layout()

    _save(fig, save_path or OUTPUT_DIR / "position_trajectory.png")
    return fig


# ──────────────────────────────────────────────────────────────
# 7. Model comparison
# ──────────────────────────────────────────────────────────────
def plot_model_comparison(
    comparison_df : pd.DataFrame,
    save_path     : Optional[Path] = None,
) -> plt.Figure:
    """Grouped bar chart comparing model accuracy / F1."""
    metrics = ["accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall"]
    metrics = [m for m in metrics if m in comparison_df.columns]

    n_models  = len(comparison_df)
    n_metrics = len(metrics)
    x         = np.arange(n_metrics)
    width     = 0.22
    offsets   = np.linspace(-(n_models - 1) / 2 * width, (n_models - 1) / 2 * width, n_models)

    bar_colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    fig, ax    = plt.subplots(figsize=(11, 6))

    for idx, (_, row) in enumerate(comparison_df.iterrows()):
        vals = [row[m] for m in metrics]
        bars = ax.bar(x + offsets[idx], vals, width,
                      label=row["model"], color=bar_colors[idx % len(bar_colors)],
                      alpha=0.85, edgecolor="none")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=9)
    ax.set_ylim(0.5, 1.04)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Model Comparison", fontsize=13)
    ax.legend(fontsize=9)
    fig.tight_layout()

    _save(fig, save_path or OUTPUT_DIR / "model_comparison.png")
    return fig


# ──────────────────────────────────────────────────────────────
# 8. Anomaly scores
# ──────────────────────────────────────────────────────────────
def plot_anomaly_scores(
    scores      : np.ndarray,
    labels      : np.ndarray,
    threshold   : float = 0.5,
    max_epochs  : int   = 600,
    save_path   : Optional[Path] = None,
) -> plt.Figure:
    """Time-series anomaly score with threshold and true-label colouring."""
    n = min(len(scores), max_epochs)
    t = np.arange(n)
    s = scores[:n]
    l = labels[:n]

    fig, ax = plt.subplots(figsize=(14, 5))

    # Background by true class
    for cls, col in CLASS_COLORS.items():
        mask = l == cls
        ax.fill_between(t, 0, s, where=mask, alpha=0.35, color=col, label=cls)

    ax.plot(t, s, lw=0.8, color="#c8cdd8", alpha=0.9, zorder=4)
    ax.axhline(threshold, color="#e74c3c", lw=1.4, ls="--", label=f"Threshold={threshold}")

    ax.set_xlabel("Epoch Index", fontsize=10)
    ax.set_ylabel("Anomaly Score", fontsize=10)
    ax.set_title("Isolation Forest Anomaly Score over Time", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()

    _save(fig, save_path or OUTPUT_DIR / "anomaly_scores.png")
    return fig


# ──────────────────────────────────────────────────────────────
# 9. Dashboard (multi-panel)
# ──────────────────────────────────────────────────────────────
def save_dashboard(
    df            : pd.DataFrame,
    metrics_rf    : Dict,
    importances   : pd.Series,
    comparison_df : pd.DataFrame,
    save_path     : Optional[Path] = None,
) -> plt.Figure:
    """
    Generate a 2×3 summary dashboard and save as a single PNG.
    """
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("SecureNav AI — Detection Dashboard", fontsize=18,
                 fontweight="bold", color="#e0e4ef", y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel 1: Signal overview (SNR) ───────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    sample = df.head(400)
    t      = np.arange(len(sample))
    labels_s = sample["label"].values
    for cls, color in CLASS_COLORS.items():
        mask = labels_s == cls
        ax1.scatter(t[mask], sample["snr_mean"].values[mask],
                    s=10, c=color, alpha=0.7, label=cls)
    ax1.set_title("SNR Mean over Time", fontsize=11)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("SNR (dB-Hz)")
    ax1.legend(fontsize=8, ncol=4)

    # ── Panel 2: Confusion matrix ─────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    cm_norm = metrics_rf.get("confusion_matrix_norm", np.eye(4))
    cmap = LinearSegmentedColormap.from_list("sn", ["#1a1d27", "#2980b9", "#27ae60"])
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap=cmap,
                xticklabels=metrics_rf.get("class_names_present", CLASS_NAMES),
                yticklabels=metrics_rf.get("class_names_present", CLASS_NAMES),
                linewidths=0.4, ax=ax2, cbar=False)
    ax2.set_title("Confusion Matrix (RF)", fontsize=11)
    ax2.set_xlabel("Predicted"); ax2.set_ylabel("True")

    # ── Panel 3: Feature importance ───────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    top = importances.head(12) if len(importances) > 0 else pd.Series()
    if len(top) > 0:
        ax3.barh(top.index[::-1], top.values[::-1], color="#3498db", alpha=0.85)
        ax3.set_title("Feature Importance (Top 12)", fontsize=11)
        ax3.set_xlabel("Score")

    # ── Panel 4: PCA ──────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    from sklearn.decomposition import PCA as _PCA
    from sklearn.preprocessing import StandardScaler as _SS
    X_feat = df[FEATURE_COLUMNS].fillna(0).values
    y_int  = df["label"].map({c: i for i, c in enumerate(CLASS_NAMES)}).fillna(0).astype(int).values
    X_pca  = _PCA(2, random_state=42).fit_transform(_SS().fit_transform(X_feat))
    for i, cls in enumerate(CLASS_NAMES):
        mask = y_int == i
        if mask.any():
            ax4.scatter(X_pca[mask, 0], X_pca[mask, 1],
                        c=CLASS_COLORS[cls], s=8, alpha=0.5, label=cls)
    ax4.set_title("PCA Feature Space", fontsize=11)
    ax4.legend(fontsize=7, markerscale=2)

    # ── Panel 5: Model comparison ─────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    if comparison_df is not None and len(comparison_df) > 0:
        colors_bar = ["#3498db", "#e74c3c", "#2ecc71"]
        x_ = np.arange(len(comparison_df))
        ax5.bar(x_, comparison_df["macro_f1"].values, color=colors_bar[:len(comparison_df)], alpha=0.85)
        ax5.set_xticks(x_)
        ax5.set_xticklabels(comparison_df["model"].values, fontsize=9)
        for xi, v in zip(x_, comparison_df["macro_f1"].values):
            ax5.text(xi, v + 0.004, f"{v:.3f}", ha="center", fontsize=9)
        ax5.set_ylim(0.7, 1.02)
        ax5.set_title("Macro F1 by Model", fontsize=11)
        ax5.set_ylabel("Macro F1")

    path = save_path or OUTPUT_DIR / "dashboard.png"
    _save(fig, path)
    return fig


# ──────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────
def _save(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [✓] Plot saved → {path}")
