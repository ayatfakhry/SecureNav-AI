"""
scripts/run_detection.py
────────────────────────
Command-line GNSS threat detection script.

Usage
─────
  python scripts/run_detection.py [OPTIONS]

Options
───────
  --samples   INT     Number of test epochs to generate (default: 2000)
  --model     STR     Model to use: rf | svm | mlp  (default: rf)
  --mode      STR     Attack mix: mixed | spoofing | jamming | drift | normal
  --plot              Generate and save output plots
  --verbose           Print per-alert output
  --seed      INT     Random seed (default: 0)
  --output    PATH    Results output directory (default: results/live_run)

Example
───────
  python scripts/run_detection.py --samples 5000 --model rf --plot --verbose
"""

import sys
import argparse
import warnings
import time
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.gnss_simulator     import GNSSSimulator
from src.spoofing_simulator import SpoofingSimulator
from src.jamming_simulator  import JammingSimulator
from src.drift_simulator    import DriftSimulator
from src.feature_extraction import extract_features, FEATURE_COLUMNS, CLASS_NAMES
from src.model_training     import (
    build_random_forest, build_svm, build_mlp,
    prepare_data, train_model,
)
from src.anomaly_detection  import (
    fit_anomaly_detector_from_df, ensemble_anomaly_score,
)
from src.evaluation         import compute_metrics, compute_roc, save_report
from src.alert_system       import (
    AlertGenerator, process_batch, print_alert_summary,
)
from src.visualization      import (
    plot_signal_overview, plot_confusion_matrix,
    plot_roc_curves, plot_anomaly_scores,
)


# ──────────────────────────────────────────────────────────────
# CLI parser
# ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="SecureNav AI — Live GNSS Threat Detection"
    )
    p.add_argument("--samples", type=int,   default=2000,
                   help="Total test epochs to generate (default: 2000)")
    p.add_argument("--model",   type=str,   default="rf",
                   choices=["rf", "svm", "mlp"],
                   help="Classifier: rf | svm | mlp  (default: rf)")
    p.add_argument("--mode",    type=str,   default="mixed",
                   choices=["mixed", "spoofing", "jamming", "drift", "normal"],
                   help="Attack scenario (default: mixed)")
    p.add_argument("--plot",    action="store_true",
                   help="Save output plots")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-alert lines")
    p.add_argument("--seed",    type=int,   default=0,
                   help="Random seed (default: 0)")
    p.add_argument("--output",  type=str,   default="results/live_run",
                   help="Output directory (default: results/live_run)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────
# Dataset builder
# ──────────────────────────────────────────────────────────────
def build_live_dataset(n_samples: int, mode: str, seed: int) -> pd.DataFrame:
    """
    Generate a balanced dataset for the requested attack mode.
    Each class receives n_samples // 4 epochs (or n_samples for 'normal').
    """
    n_each = n_samples // 4 if mode == "mixed" else n_samples // 2

    base_sim = GNSSSimulator(seed=seed)
    eps_normal = base_sim.generate_dataset(n_each)

    all_epochs = list(eps_normal)

    if mode in ("mixed", "spoofing"):
        base2  = GNSSSimulator(seed=seed + 1).generate_dataset(n_each)
        eps    = SpoofingSimulator(mode="mixed", seed=seed + 10).generate_dataset(base2)
        all_epochs += eps

    if mode in ("mixed", "jamming"):
        base3  = GNSSSimulator(seed=seed + 2).generate_dataset(n_each)
        eps    = JammingSimulator(mode="mixed", seed=seed + 20).generate_dataset(base3)
        all_epochs += eps

    if mode in ("mixed", "drift"):
        base4  = GNSSSimulator(seed=seed + 3).generate_dataset(n_each)
        eps    = DriftSimulator(mode="mixed", seed=seed + 30).generate_dataset(base4)
        all_epochs += eps

    df = extract_features(all_epochs)
    return df


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    SEP = "═" * 64
    print(f"\n{SEP}")
    print(f"  SecureNav AI — GNSS Threat Detection  (CLI)")
    print(f"{SEP}")
    print(f"  Samples : {args.samples}   Model: {args.model.upper()}"
          f"   Mode: {args.mode}   Seed: {args.seed}")
    print(f"  Output  : {outdir}\n")

    # ── 1. Generate ───────────────────────────────────────────
    t0 = time.time()
    print("  [1/5] Generating dataset …", end="", flush=True)
    df = build_live_dataset(args.samples, args.mode, args.seed)
    print(f" done  ({time.time()-t0:.1f}s)  shape={df.shape}")
    df.to_csv(outdir / "live_dataset.csv", index=False)

    # ── 2. Prepare data & train ───────────────────────────────
    print("  [2/5] Training classifier …", end="", flush=True)
    t1 = time.time()
    X, y, le = prepare_data(df, seed=args.seed)

    model_map = {"rf": build_random_forest, "svm": build_svm, "mlp": build_mlp}
    pipe = model_map[args.model](seed=args.seed)
    pipe = train_model(pipe, X, y)
    print(f" done  ({time.time()-t1:.1f}s)")

    # ── 3. Anomaly detection ──────────────────────────────────
    print("  [3/5] Fitting anomaly detector …", end="", flush=True)
    t2 = time.time()
    detector       = fit_anomaly_detector_from_df(df, seed=args.seed)
    anomaly_scores = detector.score_samples(X)
    print(f" done  ({time.time()-t2:.1f}s)")

    # ── 4. Predict + alerts ───────────────────────────────────
    print("  [4/5] Running predictions + alerts …")
    y_pred   = pipe.predict(X)
    y_prob   = pipe.predict_proba(X) if hasattr(pipe, "predict_proba") \
               else np.eye(len(le.classes_))[y_pred]

    normal_idx = list(le.classes_).index("NORMAL") if "NORMAL" in le.classes_ else 0
    ens_scores = ensemble_anomaly_score(y_prob, anomaly_scores, normal_idx)

    generator = AlertGenerator(
        cooldown_s   = 0,
        min_severity = "WARNING",
        log_path     = outdir / "alerts.json",
    )

    alerts = process_batch(
        predictions     = y_pred,
        probabilities   = y_prob,
        anomaly_scores  = anomaly_scores,
        ensemble_scores = ens_scores,
        feature_df      = df.reset_index(drop=True),
        class_names     = list(le.classes_),
        generator       = generator,
        verbose         = args.verbose,
    )
    print_alert_summary(alerts)

    # ── 5. Evaluate ───────────────────────────────────────────
    print("  [5/5] Evaluating …")
    m = compute_metrics(y_true=y, y_pred=y_pred, class_names=le.classes_.tolist())
    print(f"\n  Accuracy  : {m['accuracy']:.4f}")
    print(f"  Macro F1  : {m['macro_f1']:.4f}")
    print(f"  Weighted F1: {m['weighted_f1']:.4f}")
    print(f"\n{m['report_str']}")
    save_report(m, f"{args.model}_live", output_dir=outdir)

    # ── Optional plots ────────────────────────────────────────
    if args.plot:
        print("  Generating plots …")
        plot_signal_overview(df,     save_path=outdir / "signal_overview.png")
        plot_confusion_matrix(
            m["confusion_matrix"],
            class_names=m["class_names_present"],
            save_path=outdir / "confusion_matrix.png",
        )
        plot_confusion_matrix(
            m["confusion_matrix"],
            class_names=m["class_names_present"],
            normalised=True,
            save_path=outdir / "confusion_matrix_norm.png",
        )
        roc = compute_roc(y, y_prob, class_names=le.classes_.tolist())
        plot_roc_curves(roc, save_path=outdir / "roc_curves.png")
        y_str = np.array([le.classes_[i] for i in y])
        plot_anomaly_scores(anomaly_scores, y_str,
                            save_path=outdir / "anomaly_scores.png")
        print(f"  Plots saved → {outdir}/")

    elapsed = time.time() - t0
    print(f"\n{SEP}")
    print(f"  Done in {elapsed:.1f}s   |   Results → {outdir}/")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
