#!/usr/bin/env python
"""
Evaluation script comparing fine-tuned model predictions vs zero-shot predictions
against actual activity values.

Key: Zero-shot predictions are ONLY evaluated on samples that match each test set
(by PL + mendel_name), providing a fair comparison.

Compares:
1. DHS fine-tuned test predictions vs Zero-shot on DHS test samples
2. Sub-DHS fine-tuned test predictions vs Zero-shot on Sub-DHS test samples

Output: Summary statistics (txt) and comparison plots
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# File paths
# ============================================================
DHS_PRED_FILE = "/gpfs/scratch/ca3261/ai_in_genomics/final_project/AIG_Project/data/dhs_finetune/test_predictions.csv"
SUB_DHS_PRED_FILE = "/gpfs/scratch/ca3261/ai_in_genomics/final_project/AIG_Project/data/sub_dhs_finetune/test_predictions.csv"
ZERO_SHOT_FILE = "/gpfs/scratch/ca3261/ai_in_genomics/final_project/AIG_Project/data/zero_shot/zero_shot_scores.csv"

OUTPUT_DIR = "/gpfs/scratch/ca3261/ai_in_genomics/final_project/AIG_Project/data/final_evaluation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Load data
# ============================================================
print("Loading data...")

# Load fine-tuned predictions
dhs_pred = pd.read_csv(DHS_PRED_FILE)
sub_dhs_pred = pd.read_csv(SUB_DHS_PRED_FILE)

# Load zero-shot scores
zero_shot = pd.read_csv(ZERO_SHOT_FILE)

print(f"DHS fine-tuned test predictions: {len(dhs_pred)} samples")
print(f"Sub-DHS fine-tuned test predictions: {len(sub_dhs_pred)} samples")
print(f"Zero-shot scores (total): {len(zero_shot)} samples")

# ============================================================
# Prepare zero-shot data
# ============================================================
zero_shot_clean = zero_shot[['PL', 'mendel_name', 'Activity', 'normalized_score']].copy()
zero_shot_clean.columns = ['PL', 'mendel_name', 'activity', 'pred_activity']
zero_shot_clean['key'] = zero_shot_clean['PL'] + '_' + zero_shot_clean['mendel_name']

# Create keys for test sets
dhs_pred['key'] = dhs_pred['PL'] + '_' + dhs_pred['mendel_name']
sub_dhs_pred['key'] = sub_dhs_pred['PL'] + '_' + sub_dhs_pred['mendel_name']

# ============================================================
# Match zero-shot to each test set
# ============================================================
print("\nMatching zero-shot predictions to test sets...")

# Zero-shot filtered to DHS test samples
dhs_keys = set(dhs_pred['key'])
zero_for_dhs = zero_shot_clean[zero_shot_clean['key'].isin(dhs_keys)].copy()
zero_for_dhs = zero_for_dhs.set_index('key').loc[
    zero_for_dhs['key'][zero_for_dhs['key'].isin(dhs_keys)]
].reset_index(drop=False)
# Deduplicate and align
zero_for_dhs = zero_for_dhs.drop_duplicates(subset='key').set_index('key')
dhs_pred_indexed = dhs_pred.drop_duplicates(subset='key').set_index('key')

# Get intersection
common_dhs_keys = sorted(set(dhs_pred_indexed.index).intersection(set(zero_for_dhs.index)))
print(f"  DHS test set: {len(dhs_pred)} samples, matched with zero-shot: {len(common_dhs_keys)} samples")

dhs_matched = dhs_pred_indexed.loc[common_dhs_keys]
zero_dhs_matched = zero_for_dhs.loc[common_dhs_keys]

# Zero-shot filtered to Sub-DHS test samples
sub_dhs_keys = set(sub_dhs_pred['key'])
zero_for_sub = zero_shot_clean[zero_shot_clean['key'].isin(sub_dhs_keys)].copy()
zero_for_sub = zero_for_sub.drop_duplicates(subset='key').set_index('key')
sub_dhs_pred_indexed = sub_dhs_pred.drop_duplicates(subset='key').set_index('key')

# Get intersection
common_sub_keys = sorted(set(sub_dhs_pred_indexed.index).intersection(set(zero_for_sub.index)))
print(f"  Sub-DHS test set: {len(sub_dhs_pred)} samples, matched with zero-shot: {len(common_sub_keys)} samples")

sub_dhs_matched = sub_dhs_pred_indexed.loc[common_sub_keys]
zero_sub_matched = zero_for_sub.loc[common_sub_keys]

# ============================================================
# Compute metrics function
# ============================================================
def compute_metrics(actual, predicted, label=""):
    """Compute comprehensive evaluation metrics."""
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[mask]
    predicted = predicted[mask]
    
    n = len(actual)
    if n == 0:
        return None
    
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    pearson_r, pearson_p = stats.pearsonr(actual, predicted)
    spearman_r, spearman_p = stats.spearmanr(actual, predicted)
    median_ae = np.median(np.abs(actual - predicted))
    max_error = np.max(np.abs(actual - predicted))
    
    metrics = {
        'label': label,
        'n_samples': n,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'Median_AE': median_ae,
        'Max_Error': max_error,
        'R2': r2,
        'Pearson_r': pearson_r,
        'Pearson_p': pearson_p,
        'Spearman_r': spearman_r,
        'Spearman_p': spearman_p,
        'Pred_mean': np.mean(predicted),
        'Pred_std': np.std(predicted),
        'Actual_mean': np.mean(actual),
        'Actual_std': np.std(actual),
    }
    
    return metrics

# ============================================================
# Compute metrics for each comparison
# ============================================================
print("\nComputing metrics...")

# --- DHS Test Set Comparison ---
metrics_dhs_ft = compute_metrics(
    dhs_matched['activity'].values,
    dhs_matched['pred_activity'].values,
    label="DHS Fine-tuned"
)
metrics_zero_on_dhs = compute_metrics(
    zero_dhs_matched['activity'].values,
    zero_dhs_matched['pred_activity'].values,
    label="Zero-shot (on DHS test samples)"
)

# --- Sub-DHS Test Set Comparison ---
metrics_sub_ft = compute_metrics(
    sub_dhs_matched['activity'].values,
    sub_dhs_matched['pred_activity'].values,
    label="Sub-DHS Fine-tuned"
)
metrics_zero_on_sub = compute_metrics(
    zero_sub_matched['activity'].values,
    zero_sub_matched['pred_activity'].values,
    label="Zero-shot (on Sub-DHS test samples)"
)

# ============================================================
# Write summary statistics to file
# ============================================================
summary_file = os.path.join(OUTPUT_DIR, "evaluation_summary.txt")

def write_metrics(f, metrics):
    """Write metrics dictionary to file."""
    if metrics is None:
        f.write("  No overlapping samples found.\n\n")
        return
    f.write(f"  Model: {metrics['label']}\n")
    f.write(f"  Number of samples: {metrics['n_samples']}\n")
    f.write(f"  {'─'*50}\n")
    f.write(f"  MSE:              {metrics['MSE']:.6f}\n")
    f.write(f"  RMSE:             {metrics['RMSE']:.6f}\n")
    f.write(f"  MAE:              {metrics['MAE']:.6f}\n")
    f.write(f"  Median AE:        {metrics['Median_AE']:.6f}\n")
    f.write(f"  Max Error:        {metrics['Max_Error']:.6f}\n")
    f.write(f"  R²:               {metrics['R2']:.6f}\n")
    f.write(f"  Pearson r:        {metrics['Pearson_r']:.6f} (p={metrics['Pearson_p']:.2e})\n")
    f.write(f"  Spearman ρ:       {metrics['Spearman_r']:.6f} (p={metrics['Spearman_p']:.2e})\n")
    f.write(f"  {'─'*50}\n")
    f.write(f"  Actual    - Mean: {metrics['Actual_mean']:.6f}, Std: {metrics['Actual_std']:.6f}\n")
    f.write(f"  Predicted - Mean: {metrics['Pred_mean']:.6f}, Std: {metrics['Pred_std']:.6f}\n")
    f.write(f"\n")

def improvement_str(ft_val, zs_val, higher_better=True):
    """Compute improvement percentage."""
    if higher_better:
        if zs_val != 0:
            pct = ((ft_val - zs_val) / abs(zs_val)) * 100
        else:
            pct = float('inf') if ft_val > 0 else 0
        return f"{pct:+.1f}%"
    else:
        if zs_val != 0:
            pct = ((zs_val - ft_val) / abs(zs_val)) * 100
        else:
            pct = float('inf') if ft_val < 0 else 0
        return f"{pct:+.1f}%"

with open(summary_file, 'w') as f:
    f.write("="*70 + "\n")
    f.write("EVALUATION SUMMARY: Fine-tuned vs Zero-shot Predictions\n")
    f.write("(Zero-shot evaluated ONLY on matching test set samples)\n")
    f.write("="*70 + "\n")
    f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("Files:\n")
    f.write(f"  DHS Fine-tuned:     {DHS_PRED_FILE}\n")
    f.write(f"  Sub-DHS Fine-tuned: {SUB_DHS_PRED_FILE}\n")
    f.write(f"  Zero-shot:          {ZERO_SHOT_FILE}\n\n")
    
    # --- DHS Comparison ---
    f.write("═"*70 + "\n")
    f.write("COMPARISON 1: DHS Fine-tuned vs Zero-shot\n")
    f.write(f"(Evaluated on {len(common_dhs_keys)} matched test samples)\n")
    f.write("═"*70 + "\n\n")
    
    write_metrics(f, metrics_dhs_ft)
    write_metrics(f, metrics_zero_on_dhs)
    
    if metrics_dhs_ft and metrics_zero_on_dhs:
        f.write("  IMPROVEMENT (Fine-tuned over Zero-shot):\n")
        f.write(f"  {'─'*50}\n")
        f.write(f"  RMSE:      {metrics_dhs_ft['RMSE']:.4f} vs {metrics_zero_on_dhs['RMSE']:.4f}  ({improvement_str(metrics_dhs_ft['RMSE'], metrics_zero_on_dhs['RMSE'], higher_better=False)} reduction)\n")
        f.write(f"  MAE:       {metrics_dhs_ft['MAE']:.4f} vs {metrics_zero_on_dhs['MAE']:.4f}  ({improvement_str(metrics_dhs_ft['MAE'], metrics_zero_on_dhs['MAE'], higher_better=False)} reduction)\n")
        f.write(f"  R²:        {metrics_dhs_ft['R2']:.4f} vs {metrics_zero_on_dhs['R2']:.4f}  ({improvement_str(metrics_dhs_ft['R2'], metrics_zero_on_dhs['R2'], higher_better=True)} improvement)\n")
        f.write(f"  Pearson r: {metrics_dhs_ft['Pearson_r']:.4f} vs {metrics_zero_on_dhs['Pearson_r']:.4f}  ({improvement_str(metrics_dhs_ft['Pearson_r'], metrics_zero_on_dhs['Pearson_r'], higher_better=True)} improvement)\n")
        f.write(f"  Spearman:  {metrics_dhs_ft['Spearman_r']:.4f} vs {metrics_zero_on_dhs['Spearman_r']:.4f}  ({improvement_str(metrics_dhs_ft['Spearman_r'], metrics_zero_on_dhs['Spearman_r'], higher_better=True)} improvement)\n")
    f.write("\n")
    
    # --- Sub-DHS Comparison ---
    f.write("═"*70 + "\n")
    f.write("COMPARISON 2: Sub-DHS Fine-tuned vs Zero-shot\n")
    f.write(f"(Evaluated on {len(common_sub_keys)} matched test samples)\n")
    f.write("═"*70 + "\n\n")
    
    write_metrics(f, metrics_sub_ft)
    write_metrics(f, metrics_zero_on_sub)
    
    if metrics_sub_ft and metrics_zero_on_sub:
        f.write("  IMPROVEMENT (Fine-tuned over Zero-shot):\n")
        f.write(f"  {'─'*50}\n")
        f.write(f"  RMSE:      {metrics_sub_ft['RMSE']:.4f} vs {metrics_zero_on_sub['RMSE']:.4f}  ({improvement_str(metrics_sub_ft['RMSE'], metrics_zero_on_sub['RMSE'], higher_better=False)} reduction)\n")
        f.write(f"  MAE:       {metrics_sub_ft['MAE']:.4f} vs {metrics_zero_on_sub['MAE']:.4f}  ({improvement_str(metrics_sub_ft['MAE'], metrics_zero_on_sub['MAE'], higher_better=False)} reduction)\n")
        f.write(f"  R²:        {metrics_sub_ft['R2']:.4f} vs {metrics_zero_on_sub['R2']:.4f}  ({improvement_str(metrics_sub_ft['R2'], metrics_zero_on_sub['R2'], higher_better=True)} improvement)\n")
        f.write(f"  Pearson r: {metrics_sub_ft['Pearson_r']:.4f} vs {metrics_zero_on_sub['Pearson_r']:.4f}  ({improvement_str(metrics_sub_ft['Pearson_r'], metrics_zero_on_sub['Pearson_r'], higher_better=True)} improvement)\n")
        f.write(f"  Spearman:  {metrics_sub_ft['Spearman_r']:.4f} vs {metrics_zero_on_sub['Spearman_r']:.4f}  ({improvement_str(metrics_sub_ft['Spearman_r'], metrics_zero_on_sub['Spearman_r'], higher_better=True)} improvement)\n")
    f.write("\n")
    
    # --- Overall Summary Table ---
    f.write("═"*70 + "\n")
    f.write("OVERALL SUMMARY TABLE\n")
    f.write("═"*70 + "\n\n")
    
    header = f"  {'Model':<40} {'N':<6} {'RMSE':<8} {'MAE':<8} {'R²':<9} {'Pearson':<9} {'Spearman':<9}"
    f.write(header + "\n")
    f.write(f"  {'─'*89}\n")
    
    all_metrics = [
        metrics_dhs_ft, metrics_zero_on_dhs,
        metrics_sub_ft, metrics_zero_on_sub
    ]
    
    for m in all_metrics:
        if m:
            row = f"  {m['label']:<40} {m['n_samples']:<6} {m['RMSE']:<8.4f} {m['MAE']:<8.4f} {m['R2']:<9.4f} {m['Pearson_r']:<9.4f} {m['Spearman_r']:<9.4f}"
            f.write(row + "\n")
    
    f.write("\n" + "="*70 + "\n")
    f.write("END OF EVALUATION\n")
    f.write("="*70 + "\n")

print(f"\nSummary statistics written to: {summary_file}")

# ============================================================
# PLOTTING
# ============================================================
print("\nGenerating plots...")

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# ──────────────────────────────────────────────────────────────
# Plot 1: DHS Test Set - Scatter (Fine-tuned vs Zero-shot)
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

actual = dhs_matched['activity'].values

# DHS Fine-tuned
ax = axes[0]
ax.scatter(actual, dhs_matched['pred_activity'].values, alpha=0.6, s=30, c='#2196F3', edgecolors='white', linewidth=0.3)
lims = [min(actual.min(), dhs_matched['pred_activity'].min()) - 0.02,
        max(actual.max(), dhs_matched['pred_activity'].max()) + 0.02]
ax.plot(lims, lims, 'r--', lw=2, label='Perfect prediction')
ax.set_xlabel('Actual Activity', fontsize=12)
ax.set_ylabel('Predicted Activity', fontsize=12)
ax.set_title(f'DHS Fine-tuned\nr={metrics_dhs_ft["Pearson_r"]:.4f}, R²={metrics_dhs_ft["R2"]:.4f}', fontsize=13)
ax.legend(fontsize=10)
ax.set_xlim(lims)
ax.set_ylim(lims)

# Zero-shot on same samples
ax = axes[1]
ax.scatter(actual, zero_dhs_matched['pred_activity'].values, alpha=0.6, s=30, c='#FF9800', edgecolors='white', linewidth=0.3)
lims2 = [min(actual.min(), zero_dhs_matched['pred_activity'].min()) - 0.02,
         max(actual.max(), zero_dhs_matched['pred_activity'].max()) + 0.02]
ax.plot(lims2, lims2, 'r--', lw=2, label='Perfect prediction')
ax.set_xlabel('Actual Activity', fontsize=12)
ax.set_ylabel('Predicted Activity (Normalized Score)', fontsize=12)
ax.set_title(f'Zero-shot (on DHS test samples)\nr={metrics_zero_on_dhs["Pearson_r"]:.4f}, R²={metrics_zero_on_dhs["R2"]:.4f}', fontsize=13)
ax.legend(fontsize=10)

plt.suptitle(f'DHS Test Set Comparison (n={len(common_dhs_keys)} samples)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "dhs_test_scatter_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: dhs_test_scatter_comparison.png")

# ──────────────────────────────────────────────────────────────
# Plot 2: Sub-DHS Test Set - Scatter (Fine-tuned vs Zero-shot)
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

actual_sub = sub_dhs_matched['activity'].values

# Sub-DHS Fine-tuned
ax = axes[0]
ax.scatter(actual_sub, sub_dhs_matched['pred_activity'].values, alpha=0.6, s=30, c='#4CAF50', edgecolors='white', linewidth=0.3)
lims = [min(actual_sub.min(), sub_dhs_matched['pred_activity'].min()) - 0.02,
        max(actual_sub.max(), sub_dhs_matched['pred_activity'].max()) + 0.02]
ax.plot(lims, lims, 'r--', lw=2, label='Perfect prediction')
ax.set_xlabel('Actual Activity', fontsize=12)
ax.set_ylabel('Predicted Activity', fontsize=12)
ax.set_title(f'Sub-DHS Fine-tuned\nr={metrics_sub_ft["Pearson_r"]:.4f}, R²={metrics_sub_ft["R2"]:.4f}', fontsize=13)
ax.legend(fontsize=10)
ax.set_xlim(lims)
ax.set_ylim(lims)

# Zero-shot on same samples
ax = axes[1]
ax.scatter(actual_sub, zero_sub_matched['pred_activity'].values, alpha=0.6, s=30, c='#FF9800', edgecolors='white', linewidth=0.3)
lims2 = [min(actual_sub.min(), zero_sub_matched['pred_activity'].min()) - 0.02,
         max(actual_sub.max(), zero_sub_matched['pred_activity'].max()) + 0.02]
ax.plot(lims2, lims2, 'r--', lw=2, label='Perfect prediction')
ax.set_xlabel('Actual Activity', fontsize=12)
ax.set_ylabel('Predicted Activity (Normalized Score)', fontsize=12)
ax.set_title(f'Zero-shot (on Sub-DHS test samples)\nr={metrics_zero_on_sub["Pearson_r"]:.4f}, R²={metrics_zero_on_sub["R2"]:.4f}', fontsize=13)
ax.legend(fontsize=10)

plt.suptitle(f'Sub-DHS Test Set Comparison (n={len(common_sub_keys)} samples)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sub_dhs_test_scatter_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: sub_dhs_test_scatter_comparison.png")

# ──────────────────────────────────────────────────────────────
# Plot 3: Metrics Bar Comparison (side by side)
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

metric_names = ['RMSE', 'MAE', 'R2', 'Pearson_r']
metric_labels = ['RMSE (↓ better)', 'MAE (↓ better)', 'R² (↑ better)', 'Pearson r (↑ better)']

# Row 1: DHS comparison
for i, (metric, label) in enumerate(zip(metric_names, metric_labels)):
    ax = axes[0, i]
    vals = [metrics_dhs_ft[metric], metrics_zero_on_dhs[metric]]
    bars = ax.bar(['DHS\nFine-tuned', 'Zero-shot\n(DHS test)'], vals,
                  color=['#2196F3', '#FF9800'], edgecolor='black', linewidth=0.5)
    ax.set_title(label, fontsize=12, fontweight='bold')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{v:.4f}', ha='center', fontsize=10)
    if i == 0:
        ax.set_ylabel('DHS Test Set', fontsize=12, fontweight='bold')

# Row 2: Sub-DHS comparison
for i, (metric, label) in enumerate(zip(metric_names, metric_labels)):
    ax = axes[1, i]
    vals = [metrics_sub_ft[metric], metrics_zero_on_sub[metric]]
    bars = ax.bar(['Sub-DHS\nFine-tuned', 'Zero-shot\n(Sub-DHS test)'], vals,
                  color=['#4CAF50', '#FF9800'], edgecolor='black', linewidth=0.5)
    ax.set_title(label, fontsize=12, fontweight='bold')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{v:.4f}', ha='center', fontsize=10)
    if i == 0:
        ax.set_ylabel('Sub-DHS Test Set', fontsize=12, fontweight='bold')

plt.suptitle('Fine-tuned vs Zero-shot: Metric Comparison per Test Set', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "metrics_bar_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: metrics_bar_comparison.png")

# ──────────────────────────────────────────────────────────────
# Plot 4: Residual Comparison
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# DHS Fine-tuned residuals
ax = axes[0, 0]
residuals = dhs_matched['pred_activity'].values - dhs_matched['activity'].values
ax.scatter(dhs_matched['activity'].values, residuals, alpha=0.5, s=20, c='#2196F3')
ax.axhline(y=0, color='r', linestyle='--', lw=2)
ax.set_xlabel('Actual Activity', fontsize=11)
ax.set_ylabel('Residual (Pred - Actual)', fontsize=11)
ax.set_title('DHS Fine-tuned Residuals', fontsize=12)

# Zero-shot on DHS residuals
ax = axes[0, 1]
residuals = zero_dhs_matched['pred_activity'].values - zero_dhs_matched['activity'].values
ax.scatter(zero_dhs_matched['activity'].values, residuals, alpha=0.5, s=20, c='#FF9800')
ax.axhline(y=0, color='r', linestyle='--', lw=2)
ax.set_xlabel('Actual Activity', fontsize=11)
ax.set_ylabel('Residual (Pred - Actual)', fontsize=11)
ax.set_title('Zero-shot Residuals (DHS test samples)', fontsize=12)

# Sub-DHS Fine-tuned residuals
ax = axes[1, 0]
residuals = sub_dhs_matched['pred_activity'].values - sub_dhs_matched['activity'].values
ax.scatter(sub_dhs_matched['activity'].values, residuals, alpha=0.5, s=20, c='#4CAF50')
ax.axhline(y=0, color='r', linestyle='--', lw=2)
ax.set_xlabel('Actual Activity', fontsize=11)
ax.set_ylabel('Residual (Pred - Actual)', fontsize=11)
ax.set_title('Sub-DHS Fine-tuned Residuals', fontsize=12)

# Zero-shot on Sub-DHS residuals
ax = axes[1, 1]
residuals = zero_sub_matched['pred_activity'].values - zero_sub_matched['activity'].values
ax.scatter(zero_sub_matched['activity'].values, residuals, alpha=0.5, s=20, c='#FF9800')
ax.axhline(y=0, color='r', linestyle='--', lw=2)
ax.set_xlabel('Actual Activity', fontsize=11)
ax.set_ylabel('Residual (Pred - Actual)', fontsize=11)
ax.set_title('Zero-shot Residuals (Sub-DHS test samples)', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "residual_plots.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: residual_plots.png")

# ──────────────────────────────────────────────────────────────
# Plot 5: Distribution of Predictions vs Actual
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# DHS
ax = axes[0, 0]
ax.hist(dhs_matched['activity'].values, bins=25, alpha=0.6, label='Actual', color='gray', edgecolor='black', linewidth=0.5)
ax.hist(dhs_matched['pred_activity'].values, bins=25, alpha=0.6, label='DHS FT Predicted', color='#2196F3', edgecolor='black', linewidth=0.5)
ax.set_xlabel('Activity', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('DHS Fine-tuned: Predictions vs Actual', fontsize=12)
ax.legend()

ax = axes[0, 1]
ax.hist(zero_dhs_matched['activity'].values, bins=25, alpha=0.6, label='Actual', color='gray', edgecolor='black', linewidth=0.5)
ax.hist(zero_dhs_matched['pred_activity'].values, bins=25, alpha=0.6, label='Zero-shot Predicted', color='#FF9800', edgecolor='black', linewidth=0.5)
ax.set_xlabel('Activity', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('Zero-shot (DHS test): Predictions vs Actual', fontsize=12)
ax.legend()

# Sub-DHS
ax = axes[1, 0]
ax.hist(sub_dhs_matched['activity'].values, bins=25, alpha=0.6, label='Actual', color='gray', edgecolor='black', linewidth=0.5)
ax.hist(sub_dhs_matched['pred_activity'].values, bins=25, alpha=0.6, label='Sub-DHS FT Predicted', color='#4CAF50', edgecolor='black', linewidth=0.5)
ax.set_xlabel('Activity', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('Sub-DHS Fine-tuned: Predictions vs Actual', fontsize=12)
ax.legend()

ax = axes[1, 1]
ax.hist(zero_sub_matched['activity'].values, bins=25, alpha=0.6, label='Actual', color='gray', edgecolor='black', linewidth=0.5)
ax.hist(zero_sub_matched['pred_activity'].values, bins=25, alpha=0.6, label='Zero-shot Predicted', color='#FF9800', edgecolor='black', linewidth=0.5)
ax.set_xlabel('Activity', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('Zero-shot (Sub-DHS test): Predictions vs Actual', fontsize=12)
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "distribution_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: distribution_comparison.png")

# ──────────────────────────────────────────────────────────────
# Plot 6: Cumulative Error Distribution
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# DHS test set
ax = axes[0]
errors_dhs_ft = np.sort(np.abs(dhs_matched['pred_activity'].values - dhs_matched['activity'].values))
errors_zero_dhs = np.sort(np.abs(zero_dhs_matched['pred_activity'].values - zero_dhs_matched['activity'].values))
ax.plot(errors_dhs_ft, np.arange(1, len(errors_dhs_ft)+1)/len(errors_dhs_ft),
        label='DHS Fine-tuned', color='#2196F3', lw=2.5)
ax.plot(errors_zero_dhs, np.arange(1, len(errors_zero_dhs)+1)/len(errors_zero_dhs),
        label='Zero-shot', color='#FF9800', lw=2.5, linestyle='--')
ax.set_xlabel('Absolute Error', fontsize=12)
ax.set_ylabel('Cumulative Proportion', fontsize=12)
ax.set_title(f'DHS Test Set (n={len(common_dhs_keys)})', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Sub-DHS test set
ax = axes[1]
errors_sub_ft = np.sort(np.abs(sub_dhs_matched['pred_activity'].values - sub_dhs_matched['activity'].values))
errors_zero_sub = np.sort(np.abs(zero_sub_matched['pred_activity'].values - zero_sub_matched['activity'].values))
ax.plot(errors_sub_ft, np.arange(1, len(errors_sub_ft)+1)/len(errors_sub_ft),
        label='Sub-DHS Fine-tuned', color='#4CAF50', lw=2.5)
ax.plot(errors_zero_sub, np.arange(1, len(errors_zero_sub)+1)/len(errors_zero_sub),
        label='Zero-shot', color='#FF9800', lw=2.5, linestyle='--')
ax.set_xlabel('Absolute Error', fontsize=12)
ax.set_ylabel('Cumulative Proportion', fontsize=12)
ax.set_title(f'Sub-DHS Test Set (n={len(common_sub_keys)})', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.suptitle('Cumulative Error Distribution', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cumulative_error_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: cumulative_error_comparison.png")

# ──────────────────────────────────────────────────────────────
# Plot 7: Box plot of errors per comparison
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# DHS
ax = axes[0]
error_dhs_data = pd.DataFrame({
    'DHS Fine-tuned': np.abs(dhs_matched['pred_activity'].values - dhs_matched['activity'].values),
    'Zero-shot': np.abs(zero_dhs_matched['pred_activity'].values - zero_dhs_matched['activity'].values),
})
error_dhs_melted = error_dhs_data.melt(var_name='Model', value_name='Absolute Error')
sns.boxplot(data=error_dhs_melted, x='Model', y='Absolute Error', ax=ax,
            palette=['#2196F3', '#FF9800'], width=0.5)
ax.set_title(f'DHS Test Set (n={len(common_dhs_keys)})', fontsize=13, fontweight='bold')

# Sub-DHS
ax = axes[1]
error_sub_data = pd.DataFrame({
    'Sub-DHS Fine-tuned': np.abs(sub_dhs_matched['pred_activity'].values - sub_dhs_matched['activity'].values),
    'Zero-shot': np.abs(zero_sub_matched['pred_activity'].values - zero_sub_matched['activity'].values),
})
error_sub_melted = error_sub_data.melt(var_name='Model', value_name='Absolute Error')
sns.boxplot(data=error_sub_melted, x='Model', y='Absolute Error', ax=ax,
            palette=['#4CAF50', '#FF9800'], width=0.5)
ax.set_title(f'Sub-DHS Test Set (n={len(common_sub_keys)})', fontsize=13, fontweight='bold')

plt.suptitle('Absolute Error Distribution', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "error_boxplot_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: error_boxplot_comparison.png")

# ──────────────────────────────────────────────────────────────
# Plot 8: Rank comparison
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# DHS Fine-tuned ranks
ax = axes[0, 0]
actual_ranks = stats.rankdata(dhs_matched['activity'].values)
pred_ranks = stats.rankdata(dhs_matched['pred_activity'].values)
ax.scatter(actual_ranks, pred_ranks, alpha=0.5, s=20, c='#2196F3')
ax.plot([actual_ranks.min(), actual_ranks.max()], [actual_ranks.min(), actual_ranks.max()], 'r--', lw=2)
ax.set_xlabel('Actual Rank', fontsize=11)
ax.set_ylabel('Predicted Rank', fontsize=11)
ax.set_title(f'DHS Fine-tuned (Rank)\nSpearman ρ={metrics_dhs_ft["Spearman_r"]:.4f}', fontsize=12)

# Zero-shot on DHS ranks
ax = axes[0, 1]
pred_ranks = stats.rankdata(zero_dhs_matched['pred_activity'].values)
ax.scatter(actual_ranks, pred_ranks, alpha=0.5, s=20, c='#FF9800')
ax.plot([actual_ranks.min(), actual_ranks.max()], [actual_ranks.min(), actual_ranks.max()], 'r--', lw=2)
ax.set_xlabel('Actual Rank', fontsize=11)
ax.set_ylabel('Predicted Rank', fontsize=11)
ax.set_title(f'Zero-shot on DHS test (Rank)\nSpearman ρ={metrics_zero_on_dhs["Spearman_r"]:.4f}', fontsize=12)

# Sub-DHS Fine-tuned ranks
ax = axes[1, 0]
actual_ranks_sub = stats.rankdata(sub_dhs_matched['activity'].values)
pred_ranks = stats.rankdata(sub_dhs_matched['pred_activity'].values)
ax.scatter(actual_ranks_sub, pred_ranks, alpha=0.5, s=20, c='#4CAF50')
ax.plot([actual_ranks_sub.min(), actual_ranks_sub.max()], [actual_ranks_sub.min(), actual_ranks_sub.max()], 'r--', lw=2)
ax.set_xlabel('Actual Rank', fontsize=11)
ax.set_ylabel('Predicted Rank', fontsize=11)
ax.set_title(f'Sub-DHS Fine-tuned (Rank)\nSpearman ρ={metrics_sub_ft["Spearman_r"]:.4f}', fontsize=12)

# Zero-shot on Sub-DHS ranks
ax = axes[1, 1]
pred_ranks = stats.rankdata(zero_sub_matched['pred_activity'].values)
ax.scatter(actual_ranks_sub, pred_ranks, alpha=0.5, s=20, c='#FF9800')
ax.plot([actual_ranks_sub.min(), actual_ranks_sub.max()], [actual_ranks_sub.min(), actual_ranks_sub.max()], 'r--', lw=2)
ax.set_xlabel('Actual Rank', fontsize=11)
ax.set_ylabel('Predicted Rank', fontsize=11)
ax.set_title(f'Zero-shot on Sub-DHS test (Rank)\nSpearman ρ={metrics_zero_on_sub["Spearman_r"]:.4f}', fontsize=12)

plt.suptitle('Rank-Based Comparison', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rank_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: rank_comparison.png")

# ──────────────────────────────────────────────────────────────
# Plot 9: Combined summary figure
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Row 1: DHS test set
ax = axes[0, 0]
ax.scatter(dhs_matched['activity'].values, dhs_matched['pred_activity'].values,
           alpha=0.5, s=25, c='#2196F3', label='DHS FT')
ax.scatter(dhs_matched['activity'].values, zero_dhs_matched['pred_activity'].values,
           alpha=0.5, s=25, c='#FF9800', marker='^', label='Zero-shot')
lims = [min(dhs_matched['activity'].min(), zero_dhs_matched['pred_activity'].min(), dhs_matched['pred_activity'].min()) - 0.05,
        max(dhs_matched['activity'].max(), zero_dhs_matched['pred_activity'].max(), dhs_matched['pred_activity'].max()) + 0.05]
ax.plot(lims, lims, 'r--', lw=1.5, alpha=0.7)
ax.set_xlabel('Actual Activity')
ax.set_ylabel('Predicted Activity')
ax.set_title(f'DHS Test: Overlay (n={len(common_dhs_keys)})')
ax.legend()

ax = axes[0, 1]
sns.kdeplot(data=pd.DataFrame({
    'DHS FT Error': dhs_matched['pred_activity'].values - dhs_matched['activity'].values,
    'Zero-shot Error': zero_dhs_matched['pred_activity'].values - zero_dhs_matched['activity'].values,
}), ax=ax, fill=True, alpha=0.4, palette=['#2196F3', '#FF9800'])
ax.axvline(x=0, color='r', linestyle='--', lw=1.5)
ax.set_xlabel('Error (Pred - Actual)')
ax.set_title('DHS Test: Error Distribution')

ax = axes[0, 2]
metrics_to_plot = ['RMSE', 'MAE', 'Pearson_r', 'Spearman_r']
x = np.arange(len(metrics_to_plot))
width = 0.35
vals_ft = [metrics_dhs_ft[m] for m in metrics_to_plot]
vals_zs = [metrics_zero_on_dhs[m] for m in metrics_to_plot]
bars1 = ax.bar(x - width/2, vals_ft, width, label='DHS FT', color='#2196F3', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, vals_zs, width, label='Zero-shot', color='#FF9800', edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(['RMSE', 'MAE', 'Pearson r', 'Spearman ρ'])
ax.set_title('DHS Test: Metrics')
ax.legend()

# Row 2: Sub-DHS test set
ax = axes[1, 0]
ax.scatter(sub_dhs_matched['activity'].values, sub_dhs_matched['pred_activity'].values,
           alpha=0.5, s=25, c='#4CAF50', label='Sub-DHS FT')
ax.scatter(sub_dhs_matched['activity'].values, zero_sub_matched['pred_activity'].values,
           alpha=0.5, s=25, c='#FF9800', marker='^', label='Zero-shot')
lims = [min(sub_dhs_matched['activity'].min(), zero_sub_matched['pred_activity'].min(), sub_dhs_matched['pred_activity'].min()) - 0.05,
        max(sub_dhs_matched['activity'].max(), zero_sub_matched['pred_activity'].max(), sub_dhs_matched['pred_activity'].max()) + 0.05]
ax.plot(lims, lims, 'r--', lw=1.5, alpha=0.7)
ax.set_xlabel('Actual Activity')
ax.set_ylabel('Predicted Activity')
ax.set_title(f'Sub-DHS Test: Overlay (n={len(common_sub_keys)})')
ax.legend()

ax = axes[1, 1]
sns.kdeplot(data=pd.DataFrame({
    'Sub-DHS FT Error': sub_dhs_matched['pred_activity'].values - sub_dhs_matched['activity'].values,
    'Zero-shot Error': zero_sub_matched['pred_activity'].values - zero_sub_matched['activity'].values,
}), ax=ax, fill=True, alpha=0.4, palette=['#4CAF50', '#FF9800'])
ax.axvline(x=0, color='r', linestyle='--', lw=1.5)
ax.set_xlabel('Error (Pred - Actual)')
ax.set_title('Sub-DHS Test: Error Distribution')

ax = axes[1, 2]
vals_ft = [metrics_sub_ft[m] for m in metrics_to_plot]
vals_zs = [metrics_zero_on_sub[m] for m in metrics_to_plot]
bars1 = ax.bar(x - width/2, vals_ft, width, label='Sub-DHS FT', color='#4CAF50', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, vals_zs, width, label='Zero-shot', color='#FF9800', edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(['RMSE', 'MAE', 'Pearson r', 'Spearman ρ'])
ax.set_title('Sub-DHS Test: Metrics')
ax.legend()

plt.suptitle('Fine-tuned vs Zero-shot: Complete Comparison', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "combined_summary_figure.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: combined_summary_figure.png")

# ============================================================
# Final printout
# ============================================================
print("\n" + "="*60)
print("EVALUATION COMPLETE")
print("="*60)
print(f"\nAll results saved to: {OUTPUT_DIR}/")
print("\nFiles generated:")
print("  - evaluation_summary.txt")
print("  - dhs_test_scatter_comparison.png")
print("  - sub_dhs_test_scatter_comparison.png")
print("  - metrics_bar_comparison.png")
print("  - residual_plots.png")
print("  - distribution_comparison.png")
print("  - cumulative_error_comparison.png")
print("  - error_boxplot_comparison.png")
print("  - rank_comparison.png")
print("  - combined_summary_figure.png")

print("\n" + "─"*60)
print("QUICK SUMMARY")
print("─"*60)
print(f"\n{'─'*60}")
print(f"DHS Test Set ({len(common_dhs_keys)} matched samples):")
print(f"  {'Model':<30} {'RMSE':<8} {'R²':<9} {'Pearson':<9} {'Spearman':<9}")
print(f"  {'DHS Fine-tuned':<30} {metrics_dhs_ft['RMSE']:<8.4f} {metrics_dhs_ft['R2']:<9.4f} {metrics_dhs_ft['Pearson_r']:<9.4f} {metrics_dhs_ft['Spearman_r']:<9.4f}")
print(f"  {'Zero-shot':<30} {metrics_zero_on_dhs['RMSE']:<8.4f} {metrics_zero_on_dhs['R2']:<9.4f} {metrics_zero_on_dhs['Pearson_r']:<9.4f} {metrics_zero_on_dhs['Spearman_r']:<9.4f}")

print(f"\n{'─'*60}")
print(f"Sub-DHS Test Set ({len(common_sub_keys)} matched samples):")
print(f"  {'Model':<30} {'RMSE':<8} {'R²':<9} {'Pearson':<9} {'Spearman':<9}")
print(f"  {'Sub-DHS Fine-tuned':<30} {metrics_sub_ft['RMSE']:<8.4f} {metrics_sub_ft['R2']:<9.4f} {metrics_sub_ft['Pearson_r']:<9.4f} {metrics_sub_ft['Spearman_r']:<9.4f}")
print(f"  {'Zero-shot':<30} {metrics_zero_on_sub['RMSE']:<8.4f} {metrics_zero_on_sub['R2']:<9.4f} {metrics_zero_on_sub['Pearson_r']:<9.4f} {metrics_zero_on_sub['Spearman_r']:<9.4f}")
print(f"{'─'*60}")
