"""
Evaluation script for the Sox2 fine-tuning results.

Reads the training history CSV and the per-split prediction CSVs produced
by fine_tune_sox2.py and computes a full suite of performance metrics for
the train, val, and test splits.

Works with both model types:
    MLP   — training_history.csv has real epoch-level values
    Ridge — training_history.csv has a single NaN placeholder row;
             train/val metrics are read from fine_tune_summary.json

Usage
-----
    python evaluate.py --results-dir results/fine_tuned/

    # or point at specific files
    python evaluate.py \
        --results-dir results/fine_tuned/ \
        --history-csv results/fine_tuned/training_history.csv \
        --test-csv    results/fine_tuned/test_predictions.csv \
        --summary-json results/fine_tuned/fine_tune_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _safe_pearson(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Pearson r and two-tailed p-value; returns (nan, nan) if degenerate."""
    if len(a) < 3 or a.std() < 1e-8 or b.std() < 1e-8:
        return float("nan"), float("nan")
    r, p = stats.pearsonr(a, b)
    return float(r), float(p)


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Spearman ρ and two-tailed p-value; returns (nan, nan) if degenerate."""
    if len(a) < 3:
        return float("nan"), float("nan")
    r, p = stats.spearmanr(a, b)
    return float(r), float(p)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    mask = np.abs(y_true) >= eps
    if mask.sum() == 0:
        return float("nan")
    return float(
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    )


def compute_metrics(
    y_true_orig: np.ndarray,
    y_pred_orig: np.ndarray,
) -> dict[str, float]:
    """Compute the full metric suite on the original activity scale.

    Error metrics (MSE / RMSE / MAE) are additionally computed in log1p
    space so they are directly comparable with the training objective.
    """
    y_true_log    = np.log1p(np.clip(y_true_orig, 0.0, None))
    y_pred_log    = np.log1p(np.clip(y_pred_orig, 0.0, None))
    residuals_log = y_true_log - y_pred_log

    pearson_r,  pearson_p  = _safe_pearson(y_true_orig, y_pred_orig)
    spearman_r, spearman_p = _safe_spearman(y_true_orig, y_pred_orig)

    return {
        "pearson_r":    pearson_r,
        "pearson_p":    pearson_p,
        "pearson_r2":   pearson_r ** 2 if not np.isnan(pearson_r) else float("nan"),
        "spearman_rho": spearman_r,
        "spearman_p":   spearman_p,
        "r2":           _r2(y_true_orig, y_pred_orig),
        "mse_log1p":    float(np.mean(residuals_log ** 2)),
        "rmse_log1p":   float(np.sqrt(np.mean(residuals_log ** 2))),
        "mae_log1p":    float(np.mean(np.abs(residuals_log))),
        "mape_pct":     _mape(y_true_orig, y_pred_orig),
        "n_samples":    int(len(y_true_orig)),
        "mean_true":    float(y_true_orig.mean()),
        "mean_pred":    float(y_pred_orig.mean()),
        "std_true":     float(y_true_orig.std()),
        "std_pred":     float(y_pred_orig.std()),
    }


# ---------------------------------------------------------------------------
# Model type detection
# ---------------------------------------------------------------------------

def _is_ridge_run(history_df: pd.DataFrame) -> bool:
    """Return True if training_history.csv is the Ridge NaN placeholder.

    Handles:
    - single row with all NaN
    - single row with NaN in val_loss only
    - empty dataframe
    - multiple rows all NaN (should never happen but guarded)
    """
    if len(history_df) == 0:
        return True
    return bool(history_df["val_loss"].isna().all())


# ---------------------------------------------------------------------------
# NaN fallback for missing metrics
# ---------------------------------------------------------------------------

def _nan_split_metrics() -> dict[str, dict]:
    """Return all-NaN train/val dicts for Ridge runs with no epoch history."""
    nan_dict = {
        "best_epoch":   float("nan"),
        "loss_log1p":   float("nan"),
        "rmse_log1p":   float("nan"),
        "pearson_r":    float("nan"),
        "pearson_r2":   float("nan"),
        "spearman_rho": float("nan"),
        "spearman_p":   float("nan"),
        "r2":           float("nan"),
        "mse_log1p":    float("nan"),
        "mae_log1p":    float("nan"),
        "mape_pct":     float("nan"),
        "n_samples":    float("nan"),
        "mean_true":    float("nan"),
        "mean_pred":    float("nan"),
        "std_true":     float("nan"),
        "std_pred":     float("nan"),
    }
    return {"train": nan_dict.copy(), "val": nan_dict.copy()}


# ---------------------------------------------------------------------------
# Per-split metrics from history (MLP only)
# ---------------------------------------------------------------------------

def _best_epoch_metrics(history_df: pd.DataFrame) -> dict[str, dict]:
    """Extract train/val metrics at the best (lowest val loss) epoch.

    Only called for MLP runs. Drops NaN rows before finding the minimum
    so a stale Ridge placeholder can never crash this function.
    """
    valid = history_df.dropna(subset=["val_loss"])

    if len(valid) == 0:
        log.warning(
            "training_history.csv has no valid val_loss rows — "
            "returning NaN metrics. Check that this is not a Ridge run."
        )
        return _nan_split_metrics()

    best_idx = valid["val_loss"].idxmin()
    best_row = valid.loc[best_idx]

    return {
        "train": {
            "best_epoch": int(best_row["epoch"]),
            "loss_log1p": float(best_row["train_loss"]),
            "rmse_log1p": float(np.sqrt(best_row["train_loss"])),
            "pearson_r":  float(best_row["train_r"]),
            "pearson_r2": float(best_row["train_r"] ** 2),
        },
        "val": {
            "best_epoch": int(best_row["epoch"]),
            "loss_log1p": float(best_row["val_loss"]),
            "rmse_log1p": float(np.sqrt(best_row["val_loss"])),
            "pearson_r":  float(best_row["val_r"]),
            "pearson_r2": float(best_row["val_r"] ** 2),
        },
    }


# ---------------------------------------------------------------------------
# Per-split metrics from summary JSON (Ridge)
# ---------------------------------------------------------------------------

def _metrics_from_summary(summary: dict, key: str) -> dict:
    """Load a metrics dict from the summary JSON by key name.

    Uses explicit None check to avoid false negatives when metric
    values happen to be 0.0 or negative.

    Parameters
    ----------
    summary : dict
        Parsed fine_tune_summary.json
    key : str
        One of 'train_metrics', 'val_metrics', 'test_metrics'
    """
    m = summary.get(key)   # None if key missing — never {} by default

    if m is None:
        log.warning("'%s' not found in summary JSON — returning NaN", key)
        split_name = key.replace("_metrics", "")
        return _nan_split_metrics().get(split_name, _nan_split_metrics()["train"])

    # Return a unified dict that contains both the history-format keys
    # (best_epoch, loss_log1p …) and the full compute_metrics keys so
    # every row in the console table is populated.
    return {
        # history-format keys (Ridge has no epochs)
        "best_epoch":   float("nan"),
        "loss_log1p":   m.get("mse_log1p",   float("nan")),
        # full metric set
        "pearson_r":    m.get("pearson_r",    float("nan")),
        "pearson_p":    m.get("pearson_p",    float("nan")),
        "pearson_r2":   m.get("pearson_r2",   float("nan")),
        "spearman_rho": m.get("spearman_rho", float("nan")),
        "spearman_p":   m.get("spearman_p",   float("nan")),
        "r2":           m.get("r2",           float("nan")),
        "mse_log1p":    m.get("mse_log1p",    float("nan")),
        "rmse_log1p":   m.get("rmse_log1p",   float("nan")),
        "mae_log1p":    m.get("mae_log1p",    float("nan")),
        "mape_pct":     m.get("mape_pct",     float("nan")),
        "n_samples":    m.get("n_samples",    float("nan")),
        "mean_true":    m.get("mean_true",    float("nan")),
        "mean_pred":    m.get("mean_pred",    float("nan")),
        "std_true":     m.get("std_true",     float("nan")),
        "std_pred":     m.get("std_pred",     float("nan")),
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_METRIC_LABELS: dict[str, str] = {
    "pearson_r":    "Pearson r        ",
    "pearson_r2":   "Pearson r²       ",
    "spearman_rho": "Spearman ρ       ",
    "r2":           "R²               ",
    "mse_log1p":    "MSE  (log1p)     ",
    "rmse_log1p":   "RMSE (log1p)     ",
    "mae_log1p":    "MAE  (log1p)     ",
    "mape_pct":     "MAPE (%)         ",
    "n_samples":    "N samples        ",
    "mean_true":    "Mean true        ",
    "mean_pred":    "Mean pred        ",
    "std_true":     "Std  true        ",
    "std_pred":     "Std  pred        ",
}

_TABLE_METRICS = [
    "pearson_r", "pearson_r2", "spearman_rho",
    "r2", "rmse_log1p", "mae_log1p", "mape_pct",
    "n_samples",
]


def _fmt(value: float, key: str) -> str:
    if np.isnan(float(value)):
        return "     nan"
    if key == "n_samples":
        return f"{int(value):>8d}"
    if key == "mape_pct":
        return f"{value:>8.2f}"
    return f"{value:>8.4f}"


def _print_table(metrics_by_split: dict[str, dict]) -> None:
    splits = list(metrics_by_split.keys())
    col_w  = 12

    header = f"{'Metric':<25}" + "".join(f"{s:>{col_w}}" for s in splits)
    sep    = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    for key in _TABLE_METRICS:
        label = _METRIC_LABELS.get(key, key)
        row   = f"{label:<25}"
        for split in splits:
            val = metrics_by_split[split].get(key, float("nan"))
            row += f"{_fmt(val, key):>{col_w}}"
        print(row)

    print(sep)


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def run_evaluation(
    results_dir:  str,
    history_csv:  Optional[str] = None,
    test_csv:     Optional[str] = None,
    summary_json: Optional[str] = None,
    output_dir:   Optional[str] = None,
) -> dict[str, dict]:
    """Compute and report performance metrics for train, val, and test splits.

    Parameters
    ----------
    results_dir : str
        Directory produced by fine_tune_sox2.py.
    history_csv : str, optional
        Explicit path to training_history.csv.
    test_csv : str, optional
        Explicit path to test_predictions.csv.
    summary_json : str, optional
        Explicit path to fine_tune_summary.json.
    output_dir : str, optional
        Where to write eval_metrics.json and eval_metrics.csv.
        Defaults to results_dir.

    Returns
    -------
    dict  split_name → metrics_dict
    """
    base = Path(results_dir)

    history_path = Path(history_csv)  if history_csv  else base / "training_history.csv"
    test_path    = Path(test_csv)     if test_csv     else base / "test_predictions.csv"
    summary_path = Path(summary_json) if summary_json else base / "fine_tune_summary.json"
    out_dir      = Path(output_dir)   if output_dir   else base

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- load
    for p in (history_path, test_path):
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    history_df = pd.read_csv(history_path)
    test_df    = pd.read_csv(test_path)

    summary: dict = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        log.info(
            "Summary loaded — category='%s'  seed=%s  split=%s",
            summary.get("category"),
            summary.get("split_seed"),
            summary.get("split_sizes"),
        )
    else:
        log.warning("fine_tune_summary.json not found — skipping metadata display")

    # ---------------------------------------------------------------- model type
    # Use summary JSON as the primary signal — more reliable than CSV inspection.
    # Fall back to CSV inspection only if summary is unavailable.
    if summary:
        model_type = summary.get("model_type", "mlp")
    else:
        model_type = "ridge" if _is_ridge_run(history_df) else "mlp"

    ridge_run = (model_type == "ridge")
    log.info("Model type: %s", model_type.upper())

    # ----------------------------------------- train / val metrics
    if ridge_run:
        log.info("Ridge run — loading train/val metrics from summary JSON")
        history_metrics = {
            "train": _metrics_from_summary(summary, "train_metrics"),
            "val":   _metrics_from_summary(summary, "val_metrics"),
        }
    else:
        log.info("MLP run — deriving train/val metrics from training history")
        history_metrics = _best_epoch_metrics(history_df)

    # ------------------------------------------------ full test metrics
    log.info("Computing full test metrics from test_predictions.csv…")

    if "activity" not in test_df.columns or "pred_activity" not in test_df.columns:
        raise ValueError(
            "test_predictions.csv must contain 'activity' and 'pred_activity'. "
            f"Found: {test_df.columns.tolist()}"
        )

    y_true = test_df["activity"].to_numpy(dtype=np.float64)
    y_pred = test_df["pred_activity"].to_numpy(dtype=np.float64)
    test_metrics = compute_metrics(y_true, y_pred)

    # ------------------------------------------------ assemble all splits
    all_metrics: dict[str, dict] = {
        "train": history_metrics["train"],
        "val":   history_metrics["val"],
        "test":  test_metrics,
    }

    # Pad train/val dicts with nan for any key only present in test
    all_keys = sorted(
        {k for m in all_metrics.values() for k in m},
        key=lambda k: list(_METRIC_LABELS).index(k) if k in _METRIC_LABELS else 999,
    )
    for split in ("train", "val"):
        for k in all_keys:
            all_metrics[split].setdefault(k, float("nan"))

    # -------------------------------------------------------------- report
    print("\n")
    print("=" * 60)
    print("  SOX2 FINE-TUNE EVALUATION")
    print(f"  Model type: {model_type.upper()}")
    if summary:
        print(f"  Category  : {summary.get('category', 'n/a')}")
        print(f"  Split seed: {summary.get('split_seed', 'n/a')}")
        sizes = summary.get("split_sizes", {})
        print(
            f"  Sizes     : train={sizes.get('train','?')}  "
            f"val={sizes.get('val','?')}  test={sizes.get('test','?')}"
        )
        if ridge_run:
            print(f"  Alpha     : {summary.get('selected_alpha', 'n/a')}")
        else:
            best_ep = summary.get("training", {}).get("best_epoch")
            if best_ep:
                print(f"  Best epoch: {best_ep}")
    print("=" * 60)
    _print_table(all_metrics)
    print()

    # Significance on test
    pr, pp = _safe_pearson(y_true, y_pred)
    sr, sp = _safe_spearman(y_true, y_pred)
    print("  Test significance")
    print(f"    Pearson  r = {pr:.4f}  (p = {pp:.3g})")
    print(f"    Spearman ρ = {sr:.4f}  (p = {sp:.3g})")
    print()

    # Training curve (MLP only)
    if not ridge_run:
        valid_history = history_df.dropna(subset=["val_loss"])
        if len(valid_history) > 0:
            final_train_r = history_df["train_r"].iloc[-1]
            final_val_r   = history_df["val_r"].iloc[-1]
            best_val_loss = valid_history["val_loss"].min()
            best_epoch    = int(valid_history["val_loss"].idxmin()) + 1
            total_epochs  = len(history_df)

            print("  Training curve")
            print(f"    Epochs run    : {total_epochs}")
            print(f"    Best epoch    : {best_epoch}")
            print(f"    Best val loss : {best_val_loss:.4f}")
            print(f"    Final train r : {final_train_r:.4f}")
            print(f"    Final val r   : {final_val_r:.4f}")
            gap = abs(final_train_r - final_val_r)
            if gap > 0.15:
                print(
                    f"    [warn] train/val r gap = {gap:.3f} "
                    f"— possible overfitting"
                )
            print()
    else:
        print(f"  Ridge — selected alpha : {summary.get('selected_alpha', 'n/a')}")
        print("  No training curve (analytical fit)")
        print()

    # ----------------------------------------------------------- save
    metrics_path = out_dir / "eval_metrics.json"
    metrics_path.write_text(
        json.dumps(all_metrics, indent=2, default=str) + "\n"
    )
    log.info("Metrics JSON → %s", metrics_path)

    rows = [{"split": split, **m} for split, m in all_metrics.items()]
    pd.DataFrame(rows).to_csv(out_dir / "eval_metrics.csv", index=False)
    log.info("Metrics CSV  → %s", out_dir / "eval_metrics.csv")

    return all_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Sox2 fine-tuning results (MLP or Ridge)"
    )
    parser.add_argument(
        "--results-dir", type=str, required=True,
        help="Directory produced by fine_tune_sox2.py",
    )
    parser.add_argument(
        "--history-csv", type=str, default=None,
        help="Override path to training_history.csv",
    )
    parser.add_argument(
        "--test-csv", type=str, default=None,
        help="Override path to test_predictions.csv",
    )
    parser.add_argument(
        "--summary-json", type=str, default=None,
        help="Override path to fine_tune_summary.json",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Where to write eval_metrics.json/.csv (defaults to --results-dir)",
    )

    args = parser.parse_args()

    run_evaluation(
        results_dir  = args.results_dir,
        history_csv  = args.history_csv,
        test_csv     = args.test_csv,
        summary_json = args.summary_json,
        output_dir   = args.output_dir,
    )
