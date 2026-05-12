"""
Fine-tuning AlphaGenome with a Sox2 expression regression head.

For categories with n_train >= N_RIDGE_THRESHOLD (default 40):
    MLP head  — single hidden layer, dropout regularisation

For categories with n_train < N_RIDGE_THRESHOLD:
    Ridge regression with LOO-CV alpha selection — no MLP, no early stopping

Both paths write identical output files so eval_fine_tune.py works unchanged:
    test_predictions.csv
    training_history.csv   (empty for Ridge — one row of NaN)
    fine_tune_summary.json
    feature_mean.npy / feature_std.npy
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from scipy import stats
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.isotonic import IsotonicRegression
from torch.utils.data import DataLoader, Dataset

from src.genome_model import GenomeModel
from src.model_utils import ZeroShotScorer, ZeroShotScoreWeights

import random

def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_ORGANISM    = "mouse"
DEFAULT_TRACKS      = ["dnase", "chip_histone", "chip_tf"]
DEFAULT_ACTIVITY_COL = "Activity"
DEFAULT_CATEGORY_COL = "Category"
FEATURE_NAMES       = ["dnase", "h3k27ac", "h3k4me1", "ep300"]

TRAIN_FRAC      = 0.60
VAL_FRAC        = 0.20
N_RIDGE_THRESHOLD = 40   # use Ridge when n_train < this value

RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["PL"] = df["PL"].astype(str)

    if "MenDel.Name" in df.columns:
        df = df.rename(columns={"MenDel.Name": "mendel_name"})
    if "mendel_name" not in df.columns:
        df["mendel_name"] = ""
    df["mendel_name"] = df["mendel_name"].astype(str).str.strip()

    if "Sequence" in df.columns:
        df = df.rename(columns={"Sequence": "sequence"})
    df["sequence"] = df["sequence"].astype(str).str.strip().str.upper()

    if DEFAULT_ACTIVITY_COL not in df.columns:
        raise ValueError(
            f"Activity column '{DEFAULT_ACTIVITY_COL}' not found. "
            f"Available: {df.columns.tolist()}"
        )
    df = df.rename(columns={DEFAULT_ACTIVITY_COL: "activity"})
    df["activity"] = pd.to_numeric(df["activity"], errors="coerce")

    n_before = len(df)
    df = df.dropna(subset=["sequence", "activity"]).reset_index(drop=True)
    log.info("Kept %d / %d rows after dropping NA", len(df), n_before)
    return df


# ---------------------------------------------------------------------------
# Category filtering + deterministic split
# ---------------------------------------------------------------------------

def filter_and_split(
    df: pd.DataFrame,
    category: str,
    category_col: str = DEFAULT_CATEGORY_COL,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if category_col not in df.columns:
        raise ValueError(
            f"Category column '{category_col}' not found. "
            f"Available: {df.columns.tolist()}"
        )

    mask   = df[category_col].astype(str).str.strip().str.lower() == category.strip().lower()
    subset = df[mask].reset_index(drop=True)

    if len(subset) == 0:
        available = df[category_col].astype(str).str.strip().unique().tolist()
        raise ValueError(
            f"Category '{category}' not found. Available: {available}"
        )

    log.info("Category '%s': %d rows", category, len(subset))

    rng          = np.random.default_rng(seed)
    shuffled_idx = rng.permutation(len(subset))

    n_train = max(1, int(np.floor(TRAIN_FRAC * len(subset))))
    n_val   = max(1, int(np.floor(VAL_FRAC   * len(subset))))
    n_test  = len(subset) - n_train - n_val

    if n_test < 1:
        raise ValueError(
            f"Not enough samples ({len(subset)}) for three non-empty splits."
        )

    train_df = subset.iloc[shuffled_idx[:n_train]].reset_index(drop=True)
    val_df   = subset.iloc[shuffled_idx[n_train:n_train + n_val]].reset_index(drop=True)
    test_df  = subset.iloc[shuffled_idx[n_train + n_val:]].reset_index(drop=True)

    log.info(
        "Split (seed=%d): train=%d | val=%d | test=%d",
        seed, len(train_df), len(val_df), len(test_df),
    )
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(
    genome_model: GenomeModel,
    scorer: ZeroShotScorer,
    sequence: str,
    tracks: list[str],
    resolution: int = 128,
) -> np.ndarray:
    raw_outputs = genome_model.predict_on_sequence_raw(
        sequence, tracks=tracks, resolution=resolution,
    )
    score = scorer.score(raw_outputs)
    return np.array(
        [score.component_values[k] for k in FEATURE_NAMES],
        dtype=np.float32,
    )


def build_feature_matrix(
    genome_model: GenomeModel,
    scorer: ZeroShotScorer,
    df: pd.DataFrame,
    tracks: list[str],
    resolution: int = 128,
    cache_path: Optional[Path] = None,
) -> np.ndarray:
    if cache_path is not None and cache_path.exists():
        log.info("Loading cached features from %s", cache_path)
        return np.load(cache_path)

    log.info("Extracting features for %d sequences…", len(df))
    features = np.zeros((len(df), len(FEATURE_NAMES)), dtype=np.float32)

    for i, (_, row) in enumerate(df.iterrows()):
        if i % max(1, len(df) // 10) == 0:
            log.info("  [%d / %d]  %s  (%d bp)", i, len(df),
                     row["PL"], len(row["sequence"]))
        features[i] = extract_features(
            genome_model, scorer,
            sequence=str(row["sequence"]),
            tracks=tracks,
            resolution=resolution,
        )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, features)
        log.info("Cached features → %s", cache_path)

    return features


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Sox2Dataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        activities: np.ndarray,
        log_transform: bool = True,
    ) -> None:
        super().__init__()
        self.X = torch.tensor(features, dtype=torch.float32)
        y = activities.copy().astype(np.float32)
        if log_transform:
            y = np.log1p(np.clip(y, 0.0, None))
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# MLP head
# ---------------------------------------------------------------------------

class Sox2RegressionHead(nn.Module):
    """Single hidden layer MLP for n_train >= N_RIDGE_THRESHOLD."""

    def __init__(
        self,
        n_features: int = 4,
        hidden_dim: int = 32,
        dropout: float  = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Ridge path
# ---------------------------------------------------------------------------

def _ridge_metrics(
    y_true_orig: np.ndarray,
    y_pred_orig: np.ndarray,
) -> dict[str, float]:
    """Compute the same metric set as eval_fine_tune.compute_metrics."""
    y_true_log = np.log1p(np.clip(y_true_orig, 0.0, None))
    y_pred_log = np.log1p(np.clip(y_pred_orig, 0.0, None))
    res        = y_true_log - y_pred_log

    if len(y_true_orig) >= 3 and y_true_orig.std() > 1e-8 and y_pred_orig.std() > 1e-8:
        r, p   = stats.pearsonr(y_true_orig, y_pred_orig)
        sr, sp = stats.spearmanr(y_true_orig, y_pred_orig)
    else:
        r = p = sr = sp = float("nan")

    ss_res = float(np.sum((y_true_orig - y_pred_orig) ** 2))
    ss_tot = float(np.sum((y_true_orig - y_true_orig.mean()) ** 2))

    mask = np.abs(y_true_orig) >= 1e-8
    mape = float(np.mean(np.abs(
        (y_true_orig[mask] - y_pred_orig[mask]) / y_true_orig[mask]
    )) * 100) if mask.sum() > 0 else float("nan")

    return {
        "pearson_r":    float(r),
        "pearson_p":    float(p),
        "pearson_r2":   float(r ** 2) if not np.isnan(r) else float("nan"),
        "spearman_rho": float(sr),
        "spearman_p":   float(sp),
        "r2":           float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan"),
        "mse_log1p":    float(np.mean(res ** 2)),
        "rmse_log1p":   float(np.sqrt(np.mean(res ** 2))),
        "mae_log1p":    float(np.mean(np.abs(res))),
        "mape_pct":     mape,
        "n_samples":    int(len(y_true_orig)),
        "mean_true":    float(y_true_orig.mean()),
        "mean_pred":    float(y_pred_orig.mean()),
        "std_true":     float(y_true_orig.std()),
        "std_pred":     float(y_pred_orig.std()),
    }


def run_ridge(
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
    log_transform: bool = True,
    output_dir: Path = Path("."),
) -> dict:
    """RidgeCV + val-set linear recalibration."""

    # ── 1. prepare targets ──────────────────────────────────────────────
    if log_transform:
        yt_train = np.log1p(np.clip(y_train, 0.0, None))
        yt_val   = np.log1p(np.clip(y_val,   0.0, None))
    else:
        yt_train = y_train.copy()
        yt_val   = y_val.copy()

    # ── 2. fit Ridge with LOO-CV ─────────────────────────────────────────
    ridge = RidgeCV(alphas=RIDGE_ALPHAS, scoring="r2", cv=None)
    ridge.fit(X_train, yt_train)
    log.info("RidgeCV selected alpha = %.4f", ridge.alpha_)

    # raw predictions in log1p space
    raw_train_log = ridge.predict(X_train)
    raw_val_log   = ridge.predict(X_val)
    raw_test_log  = ridge.predict(X_test)

    # ── 3. linear recalibration on val set ──────────────────────────────
    # fits  pred_calibrated = a * pred_raw + b  using val labels
    # this corrects systematic scale / offset without seeing test
    calibrator = LinearRegression()
    calibrator.fit(raw_val_log.reshape(-1, 1), yt_val)

    log.info(
        "Calibration (val): slope=%.4f  intercept=%.4f",
        calibrator.coef_[0], calibrator.intercept_,
    )

    def _predict_calibrated(raw_log: np.ndarray) -> np.ndarray:
        cal_log = calibrator.predict(raw_log.reshape(-1, 1))
        return np.expm1(cal_log) if log_transform else cal_log

    pred_train = _predict_calibrated(raw_train_log)
    pred_val   = _predict_calibrated(raw_val_log)
    pred_test  = _predict_calibrated(raw_test_log)

    # ── 4. metrics ───────────────────────────────────────────────────────
    train_metrics = _ridge_metrics(y_train, pred_train)
    val_metrics   = _ridge_metrics(y_val,   pred_val)
    test_metrics  = _ridge_metrics(y_test,  pred_test)

    log.info("=" * 60)
    log.info("RIDGE TEST RESULTS (with val recalibration)")
    log.info("  Pearson r : %.4f", test_metrics["pearson_r"])
    log.info("  RMSE log1p: %.4f", test_metrics["rmse_log1p"])
    log.info("  R²        : %.4f", test_metrics["r2"])
    log.info("=" * 60)

    # ── 5. save ──────────────────────────────────────────────────────────
    joblib.dump(
        {"ridge": ridge, "calibrator": calibrator},
        output_dir / "ridge_model.pkl",
    )

    return {
        "model":          ridge,
        "calibrator":     calibrator,
        "model_type":     "ridge",
        "selected_alpha": float(ridge.alpha_),
        "train_metrics":  train_metrics,
        "val_metrics":    val_metrics,
        "test_metrics":   test_metrics,
        "pred_train":     pred_train,
        "pred_val":       pred_val,
        "pred_test":      pred_test,
    }

# ---------------------------------------------------------------------------
# MLP training helpers
# ---------------------------------------------------------------------------

def _pearson_r(pred: torch.Tensor, target: torch.Tensor) -> float:
    p = pred.detach().cpu().numpy()
    t = target.detach().cpu().numpy()
    if p.std() < 1e-8 or t.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(p, t)[0, 1])


def train_one_epoch(
    model: Sox2RegressionHead,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float = 1.0,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    all_pred:   list[torch.Tensor] = []
    all_target: list[torch.Tensor] = []

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        all_pred.append(pred.detach().cpu())
        all_target.append(y_batch.detach().cpu())

    mean_loss = total_loss / len(loader.dataset)
    r = _pearson_r(torch.cat(all_pred), torch.cat(all_target))
    return mean_loss, r


@torch.no_grad()
def evaluate(
    model: Sox2RegressionHead,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    all_pred:   list[torch.Tensor] = []
    all_target: list[torch.Tensor] = []

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        pred = model(X_batch)
        loss = criterion(pred, y_batch)

        total_loss += loss.item() * len(y_batch)
        all_pred.append(pred.cpu())
        all_target.append(y_batch.cpu())

    mean_loss = total_loss / len(loader.dataset)
    r = _pearson_r(torch.cat(all_pred), torch.cat(all_target))
    return mean_loss, r


# def train_head(
#     model: Sox2RegressionHead,
#     train_loader: DataLoader,
#     val_loader: DataLoader,
#     device: torch.device,
#     n_epochs: int    = 200,
#     lr: float        = 1e-3,
#     weight_decay: float = 0.05,
#     patience: int    = 30,
#     grad_clip: float = 1.0,
# ) -> dict:
#     model.to(device)
#     criterion = nn.MSELoss()
#     optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode="min", factor=0.5, patience=patience // 2, verbose=True,
#     )

#     best_val_loss   = float("inf")
#     best_state: dict = {}
#     epochs_no_improve = 0

#     history: dict[str, list] = {
#         "train_losses": [], "val_losses": [],
#         "train_rs":     [], "val_rs":     [],
#     }

#     log.info("Training — max_epochs=%d | lr=%.2e | patience=%d",
#              n_epochs, lr, patience)

#     for epoch in range(1, n_epochs + 1):
#         train_loss, train_r = train_one_epoch(
#             model, train_loader, optimizer, criterion, device, grad_clip,
#         )
#         val_loss, val_r = evaluate(model, val_loader, criterion, device)
#         scheduler.step(val_loss)

#         history["train_losses"].append(train_loss)
#         history["val_losses"].append(val_loss)
#         history["train_rs"].append(train_r)
#         history["val_rs"].append(val_r)

#         if epoch % 10 == 0 or epoch == 1:
#             log.info(
#                 "Epoch %3d/%d | train loss=%.4f r=%.3f | val loss=%.4f r=%.3f",
#                 epoch, n_epochs, train_loss, train_r, val_loss, val_r,
#             )

#         if val_loss < best_val_loss - 1e-6:
#             best_val_loss     = val_loss
#             best_state        = {k: v.clone() for k, v in model.state_dict().items()}
#             epochs_no_improve = 0
#         else:
#             epochs_no_improve += 1
#             if epochs_no_improve >= patience:
#                 log.info("Early stopping at epoch %d", epoch)
#                 break

#     if best_state:
#         model.load_state_dict(best_state)

#     best_epoch = int(np.argmin(history["val_losses"])) + 1
#     history["best_epoch"]     = best_epoch
#     history["best_val_loss"]  = best_val_loss
#     log.info("Best epoch: %d | best val loss: %.4f", best_epoch, best_val_loss)
#     return history

def train_head(
    model: Sox2RegressionHead,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_epochs: int       = 200,
    lr: float           = 1e-3,
    weight_decay: float = 0.05,
    patience: int       = 30,
    grad_clip: float    = 1.0,
    rank_margin: float  = 0.0,
) -> dict:
    model.to(device)
    mse_criterion  = nn.MSELoss()
    rank_criterion = nn.MarginRankingLoss(margin=rank_margin)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience // 2, verbose=True,
    )
    best_val_loss     = float("inf")
    best_state: dict  = {}
    epochs_no_improve = 0
    history: dict[str, list] = {
        "train_losses": [], "val_losses": [],
        "train_rs":     [], "val_rs":     [],
    }
    log.info(
        "Training — max_epochs=%d | lr=%.2e | patience=%d | rank_margin=%.2f",
        n_epochs, lr, patience, rank_margin,
    )

    # bundle criteria so train_one_epoch / evaluate signatures are unchanged
    def criterion(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_i   = pred.unsqueeze(1).expand(-1, pred.size(0))
        pred_j   = pred.unsqueeze(0).expand(pred.size(0), -1)
        mask     = (target.unsqueeze(1) - target.unsqueeze(0)) > 0
        labels   = torch.ones(mask.sum(), device=pred.device)
        return mse_criterion(pred, target) + rank_criterion(pred_i[mask], pred_j[mask], labels)

    for epoch in range(1, n_epochs + 1):
        train_loss, train_r = train_one_epoch(
            model, train_loader, optimizer, criterion, device, grad_clip,
        )
        val_loss, val_r = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)
        history["train_rs"].append(train_r)
        history["val_rs"].append(val_r)
        if epoch % 10 == 0 or epoch == 1:
            log.info(
                "Epoch %3d/%d | train loss=%.4f r=%.3f | val loss=%.4f r=%.3f",
                epoch, n_epochs, train_loss, train_r, val_loss, val_r,
            )
        if val_loss < best_val_loss - 1e-6:
            best_val_loss     = val_loss
            best_state        = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                log.info("Early stopping at epoch %d", epoch)
                break
    if best_state:
        model.load_state_dict(best_state)
    best_epoch = int(np.argmin(history["val_losses"])) + 1
    history["best_epoch"]    = best_epoch
    history["best_val_loss"] = best_val_loss
    log.info("Best epoch: %d | best val loss: %.4f", best_epoch, best_val_loss)
    return history


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config/default_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _build_zero_shot_weights(config: dict) -> ZeroShotScoreWeights:
    zs = config.get("zero_shot", {})
    w  = zs.get("weights", {})
    return ZeroShotScoreWeights(
        dnase=float(w.get("dnase",    ZeroShotScoreWeights.dnase)),
        h3k27ac=float(w.get("h3k27ac", ZeroShotScoreWeights.h3k27ac)),
        h3k4me1=float(w.get("h3k4me1", ZeroShotScoreWeights.h3k4me1)),
        ep300=float(w.get("ep300",    ZeroShotScoreWeights.ep300)),
    )


def _resolve_device(config: dict) -> torch.device:
    requested = config.get("model", {}).get("device", None)
    if requested is not None:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Shared output writers
# ---------------------------------------------------------------------------

def _write_outputs(
    output_dir: Path,
    test_df: pd.DataFrame,
    pred_test: np.ndarray,
    history: dict,
    summary_extra: dict,
    log_transform: bool,
) -> None:
    """Write test_predictions.csv, training_history.csv, fine_tune_summary.json.

    Called by both the MLP and Ridge paths so eval_fine_tune.py always
    finds the same files in the same format.
    """
    # --- test predictions ---
    pred_df = test_df[["PL", "mendel_name", "activity"]].copy()
    if log_transform:
        pred_df["pred_log1p_activity"] = pred_test
        pred_df["pred_activity"]       = np.expm1(np.clip(pred_test, 0.0, None))
    else:
        pred_df["pred_activity"] = pred_test
    pred_df.to_csv(output_dir / "test_predictions.csv", index=False)

    # --- training history (Ridge → single NaN row so file always exists) ---
    if history:
        history_df = pd.DataFrame({
            "epoch":      range(1, len(history["train_losses"]) + 1),
            "train_loss": history["train_losses"],
            "val_loss":   history["val_losses"],
            "train_r":    history["train_rs"],
            "val_r":      history["val_rs"],
        })
    else:
        # Ridge has no epoch-level history — write a single placeholder row
        # so eval_fine_tune.py can always pd.read_csv("training_history.csv")
        history_df = pd.DataFrame([{
            "epoch": 1, "train_loss": float("nan"),
            "val_loss": float("nan"),
            "train_r": float("nan"), "val_r": float("nan"),
        }])
    history_df.to_csv(output_dir / "training_history.csv", index=False)

    # --- summary JSON ---
    (output_dir / "fine_tune_summary.json").write_text(
        json.dumps(summary_extra, indent=2, default=str) + "\n"
    )

    log.info("Saved artefacts to %s", output_dir)
    log.info("  test_predictions.csv")
    log.info("  training_history.csv")
    log.info("  fine_tune_summary.json")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_fine_tuning(
    data_path: str,
    category: str,
    category_col: str   = DEFAULT_CATEGORY_COL,
    split_seed: int     = 42,
    config_path: str    = "config/default_config.yaml",
    output_dir: str     = "results/fine_tuned/",
    log_transform: bool = True,
    n_epochs: int       = 200,
    batch_size: int     = 16,
    lr: float           = 1e-3,
    weight_decay: float = 0.05,
    patience: int       = 30,
    hidden_dim: int     = 32,
    dropout: float      = 0.3,
    cache_features: bool = True,
) -> dict:
    # ---------------------------------------------------------------- setup
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    config  = load_config(config_path)
    device  = _resolve_device(config)
    log.info("Device: %s", device)
    zs_config  = config.get("zero_shot", {})
    organism   = zs_config.get("organism",
                               config.get("model", {}).get("organism", DEFAULT_ORGANISM))
    tracks     = zs_config.get("requested_tracks", DEFAULT_TRACKS)
    resolution = int(config.get("inference", {}).get("prediction_resolution", 128))
    if any(t in tracks for t in ["chip_histone", "chip_tf"]) and resolution != 128:
        log.info("Forcing resolution → 128 bp")
        resolution = 128
    signal_threshold = float(zs_config.get("signal_threshold", 0.0))

    # ---------------------------------------------------------------- data
    log.info("Loading data from %s …", data_path)
    df = _load_data(data_path)
    train_df, val_df, test_df = filter_and_split(
        df, category=category, category_col=category_col, seed=split_seed,
    )

    # --------------------------------------------------------- AlphaGenome
    log.info("Initialising AlphaGenome (%s)…", organism)
    genome_model = GenomeModel(
        organism=organism,
        device=config.get("model", {}).get("device"),
    )
    genome_model.load_model()
    scorer = ZeroShotScorer(
        weights=_build_zero_shot_weights(config),
        signal_threshold=signal_threshold,
    )

    # --------------------------------------------------------- features
    safe_cat  = category.strip().lower().replace(" ", "_")
    cache_dir = output_dir_path / "feature_cache"

    def _get_features(split_df: pd.DataFrame, split_name: str) -> np.ndarray:
        cache_file = (
            cache_dir / f"features_{safe_cat}_seed{split_seed}_{split_name}.npy"
            if cache_features else None
        )
        return build_feature_matrix(
            genome_model, scorer, split_df, tracks, resolution,
            cache_path=cache_file,
        )

    X_train = _get_features(train_df, "train")
    X_val   = _get_features(val_df,   "val")
    X_test  = _get_features(test_df,  "test")

    y_train = train_df["activity"].to_numpy(dtype=np.float32)
    y_val   = val_df["activity"].to_numpy(dtype=np.float32)
    y_test  = test_df["activity"].to_numpy(dtype=np.float32)

    # --------------------------------------------------- feature normalisation
    feat_mean = X_train.mean(axis=0)
    feat_std  = X_train.std(axis=0) + 1e-8
    X_train = (X_train - feat_mean) / feat_std
    X_val   = (X_val   - feat_mean) / feat_std
    X_test  = (X_test  - feat_mean) / feat_std
    np.save(output_dir_path / "feature_mean.npy", feat_mean)
    np.save(output_dir_path / "feature_std.npy",  feat_std)

    # ------------------------------------------------ base summary fields
    base_summary = {
        "category":        category,
        "category_col":    category_col,
        "split_seed":      split_seed,
        "split_fractions": {"train": TRAIN_FRAC, "val": VAL_FRAC,
                            "test": round(1 - TRAIN_FRAC - VAL_FRAC, 2)},
        "split_sizes":     {"train": len(train_df), "val": len(val_df),
                            "test": len(test_df)},
        "organism":        organism,
        "resolution":      resolution,
        "tracks":          tracks,
        "signal_threshold": signal_threshold,
        "feature_names":   FEATURE_NAMES,
        "log_transform":   log_transform,
    }

    # ==================================================================
    # RIDGE path  (small n)
    # ==================================================================
    if len(train_df) < N_RIDGE_THRESHOLD:
        ridge_results = run_ridge(
            X_train, y_train,
            X_val,   y_val,
            X_test,  y_test,
            log_transform=log_transform,
            output_dir=output_dir_path,
        )
        summary = {
            **base_summary,
            "model_type":     "ridge",
            "selected_alpha": ridge_results["selected_alpha"],
            "train_metrics":  ridge_results["train_metrics"],
            "val_metrics":    ridge_results["val_metrics"],
            "test_metrics":   ridge_results["test_metrics"],
        }
        # Ridge already returns predictions on the original activity scale
        _write_outputs(
            output_dir    = output_dir_path,
            test_df       = test_df,
            pred_test     = ridge_results["pred_test"],  # original scale
            history       = {},                          # → placeholder CSV row
            summary_extra = summary,
            log_transform = False,                       # do NOT apply expm1 again
        )
        return {
            "model":               ridge_results["model"],
            "model_type":          "ridge",
            "history":             {},
            "test_metrics":        ridge_results["test_metrics"],
            "test_predictions_df": pd.read_csv(output_dir_path / "test_predictions.csv"),
            "output_dir":          str(output_dir_path),
        }

    # ==================================================================
    # MLP path  (sufficient n)
    # ==================================================================
    train_dataset = Sox2Dataset(X_train, y_train, log_transform=log_transform)
    val_dataset   = Sox2Dataset(X_val,   y_val,   log_transform=log_transform)
    test_dataset  = Sox2Dataset(X_test,  y_test,  log_transform=log_transform)

    generator = torch.Generator()
    generator.manual_seed(split_seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, generator=generator, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    model = Sox2RegressionHead(
        n_features=len(FEATURE_NAMES),
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Sox2RegressionHead — %d trainable parameters", n_params)

    history = train_head(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        n_epochs=n_epochs,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
    )

    # ── collect predictions for all splits on the original activity scale ──
    model.eval()
    model.to(device)

    def _mlp_predict_orig(X: np.ndarray) -> np.ndarray:
        """Run MLP inference and invert log1p transform exactly once."""
        with torch.no_grad():
            log_preds = (
                model(torch.tensor(X, dtype=torch.float32).to(device))
                .cpu().numpy()
            )
        return np.expm1(np.clip(log_preds, 0.0, None)) if log_transform else log_preds

    pred_train_orig = _mlp_predict_orig(X_train)
    pred_val_orig   = _mlp_predict_orig(X_val)
    pred_test_orig  = _mlp_predict_orig(X_test)

    # ── metrics on original scale (consistent with eval_fine_tune.py) ──
    train_metrics = _ridge_metrics(
        y_train.astype(np.float64), pred_train_orig.astype(np.float64)
    )
    val_metrics   = _ridge_metrics(
        y_val.astype(np.float64),   pred_val_orig.astype(np.float64)
    )
    test_metrics  = _ridge_metrics(
        y_test.astype(np.float64),  pred_test_orig.astype(np.float64)
    )

    # ── internal log1p-space diagnostic (not used downstream) ──────────
    criterion         = nn.MSELoss()
    test_loss, test_r = evaluate(model, test_loader, criterion, device)

    log.info("=" * 60)
    log.info("TEST RESULTS  (category='%s', seed=%d)", category, split_seed)
    log.info("  MSE (log1p space)      : %.4f", test_loss)
    log.info("  Pearson r (log1p space): %.4f", test_r)
    log.info("  Pearson r (orig scale) : %.4f", test_metrics["pearson_r"])
    log.info("  RMSE log1p (orig scale): %.4f", test_metrics["rmse_log1p"])
    log.info("  R² (orig scale)        : %.4f", test_metrics["r2"])
    log.info("=" * 60)

    # ── checkpoint ─────────────────────────────────────────────────────
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_mean":     feat_mean,
            "feature_std":      feat_std,
            "feature_names":    FEATURE_NAMES,
            "hyperparams": {
                "n_features":    len(FEATURE_NAMES),
                "hidden_dim":    hidden_dim,
                "dropout":       dropout,
                "log_transform": log_transform,
            },
            "category":   category,
            "split_seed": split_seed,
        },
        output_dir_path / "sox2_head_best.pt",
    )

    summary = {
        **base_summary,
        "model_type": "mlp",
        "hyperparams": {
            "hidden_dim":   hidden_dim,
            "dropout":      dropout,
            "n_epochs_max": n_epochs,
            "batch_size":   batch_size,
            "lr":           lr,
            "weight_decay": weight_decay,
            "patience":     patience,
        },
        "training": {
            "best_epoch":    history["best_epoch"],
            "best_val_loss": history["best_val_loss"],
            "final_train_r": history["train_rs"][-1],
            "final_val_r":   history["val_rs"][-1],
        },
        "train_metrics": train_metrics,
        "val_metrics":   val_metrics,
        "test_metrics":  test_metrics,
    }

    # pred_test_orig is already on the original scale — pass log_transform=False
    # so _write_outputs does NOT apply expm1 a second time
    _write_outputs(
        output_dir    = output_dir_path,
        test_df       = test_df,
        pred_test     = pred_test_orig,  # original scale — no further transform needed
        history       = history,
        summary_extra = summary,
        log_transform = False,           # expm1 already applied above
    )

    return {
        "model":               model,
        "model_type":          "mlp",
        "history":             history,
        "test_metrics":        test_metrics,
        "test_predictions_df": pd.read_csv(output_dir_path / "test_predictions.csv"),
        "output_dir":          str(output_dir_path),
    }

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune a Sox2 regression head (MLP or Ridge) on AlphaGenome features"
    )
    parser.add_argument("--data-file",      type=str,   required=True)
    parser.add_argument("--category",       type=str,   required=True)
    parser.add_argument("--category-col",   type=str,   default=DEFAULT_CATEGORY_COL)
    parser.add_argument("--split-seed",     type=int,   default=42)
    parser.add_argument("--output-dir",     type=str,   default="results/fine_tuned/")
    parser.add_argument("--config",         type=str,   default="config/default_config.yaml")
    parser.add_argument("--n-epochs",       type=int,   default=200)
    parser.add_argument("--batch-size",     type=int,   default=16)
    parser.add_argument("--lr",             type=float, default=5e-4)
    parser.add_argument("--weight-decay",   type=float, default=0.01)
    parser.add_argument("--patience",       type=int,   default=9999)
    parser.add_argument("--hidden-dim",     type=int,   default=32)
    parser.add_argument("--dropout",        type=float, default=0.3)
    parser.add_argument("--no-log-transform", action="store_true")
    parser.add_argument("--no-cache",         action="store_true")

    args = parser.parse_args()

    split_seed = args.split_seed

    _set_seeds(split_seed)

    run_fine_tuning(
        data_path     = args.data_file,
        category      = args.category,
        category_col  = args.category_col,
        split_seed    = split_seed,
        config_path   = args.config,
        output_dir    = args.output_dir,
        log_transform = not args.no_log_transform,
        n_epochs      = args.n_epochs,
        batch_size    = args.batch_size,
        lr            = args.lr,
        weight_decay  = args.weight_decay,
        patience      = args.patience,
        hidden_dim    = args.hidden_dim,
        dropout       = args.dropout,
        cache_features= not args.no_cache,
    )
