"""
Evaluate zero-shot predictions against ground-truth Activity for each
category's test split, using the same deterministic 60/20/20 split
that was used during fine-tuning so the test sets are identical.

Usage
-----
    python eval_zero_shot.py \
        --zero-shot-csv results/zero_shot/zero_shot_scores.csv \
        --categories "DHS Level" "Sub-DHS Level" \
        --split-seed 42 \
        --output-dir results/eval_zero_shot/
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

# Must match fine_tune_sox2.py exactly
DEFAULT_CATEGORY_COL = "Category"
TRAIN_FRAC = 0.60
VAL_FRAC   = 0.20
SPLIT_SEED = 42


# ---------------------------------------------------------------------------
# Metric helpers  (identical to eval_fine_tune.py)
# ---------------------------------------------------------------------------

def _safe_pearson(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if len(a) < 3 or a.std() < 1e-8 or b.std() < 1e-8:
        return float("nan"), float("nan")
    r, p = stats.pearsonr(a, b)
    return float(r), float(p)


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
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
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Full metric suite — mirrors eval_fine_tune.compute_metrics.

    NOTE: for zero-shot we have no log1p training objective so MSE/RMSE/MAE
    are computed on the original scale.  log1p versions are also included
    so the table is directly comparable with the fine-tune eval.
    """
    y_true_log = np.log1p(np.clip(y_true, 0.0, None))
    y_pred_log = np.log1p(np.clip(y_pred, 0.0, None))
    res_log    = y_true_log - y_pred_log

    pearson_r,  pearson_p  = _safe_pearson(y_true, y_pred)
    spearman_r, spearman_p = _safe_spearman(y_true, y_pred)

    return {
        # --- correlation (original scale) ---
        "pearson_r":    pearson_r,
        "pearson_p":    pearson_p,
        "pearson_r2":   pearson_r ** 2 if not np.isnan(pearson_r) else float("nan"),
        "spearman_rho": spearman_r,
        "spearman_p":   spearman_p,
        # --- coefficient of determination (original scale) ---
        "r2":           _r2(y_true, y_pred),
        # --- error in log1p space (matches fine-tune training objective) ---
        "mse_log1p":    float(np.mean(res_log ** 2)),
        "rmse_log1p":   float(np.sqrt(np.mean(res_log ** 2))),
        "mae_log1p":    float(np.mean(np.abs(res_log))),
        # --- percentage error (original scale) ---
        "mape_pct":     _mape(y_true, y_pred),
        # --- basic stats ---
        "n_samples":    int(len(y_true)),
        "mean_true":    float(y_true.mean()),
        "mean_pred":    float(y_pred.mean()),
        "std_true":     float(y_true.std()),
        "std_pred":     float(y_pred.std()),
    }


# ---------------------------------------------------------------------------
# Deterministic split  (must mirror fine_tune_sox2.filter_and_split exactly)
# ---------------------------------------------------------------------------

def _get_test_indices(n: int, seed: int) -> np.ndarray:
    """Return the test-split row positions for a category of size *n*."""
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n)

    n_train = max(1, int(np.floor(TRAIN_FRAC * n)))
    n_val   = max(1, int(np.floor(VAL_FRAC   * n)))
    # test = remainder, identical logic to fine_tune_sox2.py
    return shuffled[n_train + n_val:]


# ---------------------------------------------------------------------------
# Console table helper
# ---------------------------------------------------------------------------

_TABLE_METRICS = [
    "pearson_r", "pearson_r2", "spearman_rho",
    "r2", "rmse_log1p", "mae_log1p", "mape_pct",
    "n_samples",
]

_METRIC_LABELS: dict[str, str] = {
    "pearson_r":    "Pearson r        ",
    "pearson_r2":   "Pearson r²       ",
    "spearman_rho": "Spearman ρ       ",
    "r2":           "R²               ",
    "rmse_log1p":   "RMSE (log1p)     ",
    "mae_log1p":    "MAE  (log1p)     ",
    "mape_pct":     "MAPE (%)         ",
    "n_samples":    "N samples        ",
}


def _fmt(value: float, key: str) -> str:
    if np.isnan(value):
        return "     nan"
    if key == "n_samples":
        return f"{int(value):>8d}"
    if key == "mape_pct":
        return f"{value:>8.2f}"
    return f"{value:>8.4f}"


def _print_table(metrics_by_category: dict[str, dict]) -> None:
    categories = list(metrics_by_category.keys())
    col_w = 16

    header = f"{'Metric':<25}" + "".join(f"{c[:col_w]:>{col_w}}" for c in categories)
    sep    = "-" * len(header)

    print(sep)
    print(header)
    print(sep)
    for key in _TABLE_METRICS:
        label = _METRIC_LABELS.get(key, key)
        row   = f"{label:<25}"
        for cat in categories:
            val = metrics_by_category[cat].get(key, float("nan"))
            row += f"{_fmt(val, key):>{col_w}}"
        print(row)
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_zero_shot_eval(
    zero_shot_csv: str,
    categories: list[str],
    category_col: str   = DEFAULT_CATEGORY_COL,
    split_seed: int     = SPLIT_SEED,
    score_col: str      = "normalized_score",
    activity_col: str   = "Activity",
    output_dir: str     = "results/eval_zero_shot/",
) -> dict[str, dict]:
    """Evaluate zero-shot predictions on the test split for every category.

    Parameters
    ----------
    zero_shot_csv : str
        Path to zero_shot_scores.csv produced by zero_shot_inference.py.
    categories : list[str]
        Categories to evaluate — should match those used in fine-tuning.
        Case- and whitespace-insensitive.
    category_col : str
        Column in the CSV that holds category labels.
    split_seed : int
        Must match the seed used in fine_tune_sox2.py.
    score_col : str
        Column to use as the zero-shot prediction
        (``normalized_score`` or ``raw_score``).
    activity_col : str
        Ground-truth activity column.
    output_dir : str
        Where to write JSON / CSV summary files.

    Returns
    -------
    dict  category → metrics_dict
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- load
    log.info("Loading zero-shot scores from %s …", zero_shot_csv)
    df = pd.read_csv(zero_shot_csv)

    # Normalise column names
    if "MenDel.Name" in df.columns:
        df = df.rename(columns={"MenDel.Name": "mendel_name"})
    if "Sequence" in df.columns:
        df = df.rename(columns={"Sequence": "sequence"})

    # Ensure activity column is numeric
    df[activity_col] = pd.to_numeric(df[activity_col], errors="coerce")

    if category_col not in df.columns:
        raise ValueError(
            f"Category column '{category_col}' not found. "
            f"Available: {df.columns.tolist()}"
        )
    if score_col not in df.columns:
        raise ValueError(
            f"Score column '{score_col}' not found. "
            f"Available: {df.columns.tolist()}"
        )

    all_metrics: dict[str, dict] = {}
    prediction_frames: list[pd.DataFrame] = []

    for category in categories:
        log.info("─" * 50)
        log.info("Category: '%s'", category)

        # ── filter to category (case-insensitive, mirrors fine_tune_sox2) ──
        mask = (
            df[category_col].astype(str).str.strip().str.lower()
            == category.strip().lower()
        )
        subset = df[mask].dropna(subset=[activity_col, score_col]).reset_index(drop=True)

        if len(subset) == 0:
            log.warning("No rows found for category '%s' — skipping.", category)
            continue

        log.info("  Total rows in category: %d", len(subset))

        # ── reproduce identical test split ──
        test_idx = _get_test_indices(len(subset), seed=split_seed)
        test_df  = subset.iloc[test_idx].reset_index(drop=True)

        n_train = max(1, int(np.floor(TRAIN_FRAC * len(subset))))
        n_val   = max(1, int(np.floor(VAL_FRAC   * len(subset))))
        n_test  = len(subset) - n_train - n_val

        log.info(
            "  Split (seed=%d): train=%d | val=%d | test=%d",
            split_seed, n_train, n_val, n_test,
        )

        if len(test_df) == 0:
            log.warning("  Test split is empty for '%s' — skipping.", category)
            continue

        y_true = test_df[activity_col].to_numpy(dtype=np.float64)
        y_pred = test_df[score_col].to_numpy(dtype=np.float64)

        metrics = compute_metrics(y_true, y_pred)
        all_metrics[category] = metrics

        pr, pp = _safe_pearson(y_true, y_pred)
        sr, sp = _safe_spearman(y_true, y_pred)
        log.info("  Pearson  r = %.4f  (p = %.3g)", pr, pp)
        log.info("  Spearman ρ = %.4f  (p = %.3g)", sr, sp)
        log.info("  RMSE log1p = %.4f", metrics["rmse_log1p"])
        log.info("  MAPE %%     = %.2f", metrics["mape_pct"])

        # Collect predictions for the output CSV
        pred_df = test_df[["PL", "mendel_name", activity_col, score_col]].copy()
        pred_df = pred_df.rename(columns={score_col: "zero_shot_score"})
        pred_df.insert(0, "category", category)
        prediction_frames.append(pred_df)

    # ---------------------------------------------------------------- report
    print("\n")
    print("=" * 70)
    print("  ZERO-SHOT EVALUATION  —  TEST SPLITS")
    print(f"  Score column : {score_col}")
    print(f"  Split seed   : {split_seed}")
    print(f"  Split fracs  : train={TRAIN_FRAC}  val={VAL_FRAC}  "
          f"test={round(1 - TRAIN_FRAC - VAL_FRAC, 2)}")
    print("=" * 70)
    _print_table(all_metrics)
    print()

    # Per-category significance
    for cat, m in all_metrics.items():
        print(f"  [{cat}]")
        print(f"    Pearson  r = {m['pearson_r']:.4f}  (p = {m['pearson_p']:.3g})")
        print(f"    Spearman ρ = {m['spearman_rho']:.4f}  (p = {m['spearman_p']:.3g})")
        print(f"    N test     = {m['n_samples']}")
        print()

    # ---------------------------------------------------------------- save
    metrics_path = out_dir / "zero_shot_eval_metrics.json"
    metrics_path.write_text(
        json.dumps(all_metrics, indent=2, default=str) + "\n"
    )
    log.info("Metrics JSON → %s", metrics_path)

    # Flat CSV — one row per category
    rows = [{"category": cat, **m} for cat, m in all_metrics.items()]
    pd.DataFrame(rows).to_csv(out_dir / "zero_shot_eval_metrics.csv", index=False)

    # Per-sample predictions
    if prediction_frames:
        all_preds = pd.concat(prediction_frames, ignore_index=True)
        all_preds.to_csv(out_dir / "zero_shot_test_predictions.csv", index=False)
        log.info("Per-sample predictions → %s",
                 out_dir / "zero_shot_test_predictions.csv")

    return all_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate zero-shot predictions on fine-tuning test splits"
    )
    parser.add_argument(
        "--zero-shot-csv", type=str, required=True,
        help="Path to zero_shot_scores.csv",
    )
    parser.add_argument(
        "--categories", type=str, nargs="+", required=True,
        help="Category values to evaluate (e.g. 'DHS Level' 'Sub-DHS Level')",
    )
    parser.add_argument(
        "--category-col", type=str, default=DEFAULT_CATEGORY_COL,
        help=f"Column name for categories (default: '{DEFAULT_CATEGORY_COL}')",
    )
    parser.add_argument(
        "--split-seed", type=int, default=SPLIT_SEED,
        help=f"Random seed — must match fine-tuning (default: {SPLIT_SEED})",
    )
    parser.add_argument(
        "--score-col", type=str, default="normalized_score",
        choices=["normalized_score", "raw_score"],
        help="Which zero-shot score column to use as predictions",
    )
    parser.add_argument(
        "--activity-col", type=str, default="Activity",
        help="Ground-truth activity column (default: Activity)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/eval_zero_shot/",
    )

    args = parser.parse_args()

    run_zero_shot_eval(
        zero_shot_csv=args.zero_shot_csv,
        categories=args.categories,
        category_col=args.category_col,
        split_seed=args.split_seed,
        score_col=args.score_col,
        activity_col=args.activity_col,
        output_dir=args.output_dir,
    )
