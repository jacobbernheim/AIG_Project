"""
Zero-shot inference script for Sox2 expression prediction using AlphaGenome.

Expects preprocessed data in data/processed/. Run preprocess_data.py first.

Usage:
    python zero_shot_inference.py [--output-dir results/]
"""

import argparse
import yaml
from pathlib import Path
import pandas as pd
import numpy as np

from src.data_loader import SequenceDataset
from src.model_utils import GenomeModel, ExpressionPredictor
from src.evaluation import evaluate_predictions


def load_config(config_path: str = "config/default_config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_zero_shot_prediction(
    sequences_path: str = None,
    activities_path: str = None,
    config_path: str = "config/default_config.yaml",
    output_dir: str = "results/",
) -> dict:
    """
    Run zero-shot prediction on preprocessed sequences using AlphaGenome.
    
    Args:
        sequences_path: Path to processed sequences (defaults to data/processed/sequences_processed.csv)
        activities_path: Path to processed activities (defaults to data/processed/activities_processed.csv)
        config_path: Path to configuration YAML
        output_dir: Directory to save results
        
    Returns:
        Dictionary with predictions and metrics
    """
    # Use default paths if not provided
    if sequences_path is None:
        sequences_path = "data/processed/sequences_processed.csv"
    if activities_path is None:
        activities_path = "data/processed/activities_processed.csv"
    
    # Load config
    config = load_config(config_path)
    
    # Load data
    print(f"Loading sequences from {sequences_path}...")
    sequences_df = pd.read_csv(sequences_path)
    
    print(f"Loaded {len(sequences_df)} sequences")
    
    # Load true labels if available
    labels = None
    if Path(activities_path).exists():
        activities_df = pd.read_csv(activities_path)
        if "expression" in activities_df.columns:
            labels = activities_df["expression"].values
            print(f"Loaded {len(labels)} expression labels")
    
    # Initialize AlphaGenome model
    print(f"\nInitializing AlphaGenome model...")
    genome_model = GenomeModel(
        model_name=config["model"]["model_checkpoint"],
        organism=config["model"]["organism"],
        cache_dir=config["model"]["cache_dir"],
        device=config["model"]["device"],
    )
    print(f"Model info: {genome_model.get_model_info()}")
    
    # Load model weights
    genome_model.load_model()
    
    # Run inference on all sequences
    print(f"\nRunning predictions on {len(sequences_df)} sequences...")
    all_track_predictions = {}
    aggregated_features_list = []
    
    for idx, row in sequences_df.iterrows():
        sequence = row["sequence"]
        
        if idx % max(1, len(sequences_df) // 10) == 0:
            print(f"  Progress: {idx}/{len(sequences_df)}")
        
        # Get AlphaGenome predictions
        track_preds = genome_model.predict_on_sequence(
            sequence,
            tracks=config["inference"]["available_tracks"],
            resolution=config["inference"]["prediction_resolution"],
        )
        
        # Aggregate features for expression prediction
        expr_predictor = ExpressionPredictor()
        agg_features = expr_predictor.aggregate_track_features(
            track_preds,
            aggregation=config["inference"]["feature_aggregation"],
        )
        aggregated_features_list.append(agg_features)
    
    print("✓ Predictions complete")
    
    # Aggregate all features
    all_features = np.concatenate(aggregated_features_list, axis=0)
    print(f"Aggregated feature shape: {all_features.shape}")
    
    # Predict Sox2 expression
    print(f"\nPredicting Sox2 expression levels...")
    expr_predictor = ExpressionPredictor()
    predictions = expr_predictor.predict(all_features)
    predictions = np.maximum(predictions, 0)  # Ensure non-negative
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    
    # Evaluate (if labels available)
    results = {
        "predictions": predictions,
        "feature_shape": all_features.shape,
    }
    
    if labels is not None and len(labels) == len(predictions):
        print(f"\nEvaluating predictions against {len(labels)} labels...")
        metrics = evaluate_predictions(labels, predictions)
        results.update(metrics)
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
    
    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if config["output"]["save_predictions"]:
        pred_path = output_dir / "predictions.npy"
        np.save(pred_path, predictions)
        print(f"\nSaved predictions to {pred_path}")
        
        # Also save as CSV for easier viewing
        pred_csv = output_dir / "predictions.csv"
        pred_df = pd.DataFrame({
            "sequence_index": range(len(predictions)),
            "sox2_expression_prediction": predictions,
        })
        pred_df.to_csv(pred_csv, index=False)
        print(f"Saved predictions CSV to {pred_csv}")
    
    if config["output"]["save_embeddings"]:
        emb_path = output_dir / "aggregated_features.npy"
        np.save(emb_path, all_features)
        print(f"Saved aggregated features to {emb_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Zero-shot Sox2 expression prediction using AlphaFold"
    )
    parser.add_argument(
        "--sequences-file",
        type=str,
        default=None,
        help="Path to processed sequences file (defaults to data/processed/sequences_processed.csv)"
    )
    parser.add_argument(
        "--activities-file",
        type=str,
        default=None,
        help="Path to processed activities file (defaults to data/processed/activities_processed.csv)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/",
        help="Directory to save results"
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (uses default if not provided)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    results = run_zero_shot_prediction(
        sequences_path=args.sequences_file,
        activities_path=args.activities_file,
        model_checkpoint=args.model_checkpoint,
        config_path=args.config,
        output_dir=args.output_dir,
    )
