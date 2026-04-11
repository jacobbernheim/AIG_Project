"""
Zero-shot inference script for Sox2 expression prediction.

Expects preprocessed data in data/processed/. Run preprocess_data.py first.

Usage:
    python zero_shot_inference.py [--output-dir results/]
"""

import argparse
import yaml
from pathlib import Path
import numpy as np

from src.data_loader import SequenceDataset
from src.model_utils import AlphaFoldPredictor, ExpressionPredictor
from src.evaluation import evaluate_predictions


def load_config(config_path: str = "config/default_config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_zero_shot_prediction(
    sequences_path: str = None,
    activities_path: str = None,
    model_checkpoint: str = None,
    config_path: str = "config/default_config.yaml",
    output_dir: str = "results/",
) -> dict:
    """
    Run zero-shot prediction on preprocessed sequences.
    
    Args:
        sequences_path: Path to processed sequences (defaults to data/processed/sequences_processed.csv)
        activities_path: Path to processed activities (defaults to data/processed/activities_processed.csv)
        model_checkpoint: Path to pretrained model (uses default if None)
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
    dataset = SequenceDataset(sequences_path)
    dataset.load()
    print(f"Loaded {len(dataset.sequences)} sequences")
    print(f"Dataset summary: {dataset.summary()}")
    
    # Initialize model
    print("Initializing AlphaFold model...")
    af_predictor = AlphaFoldPredictor(
        model_path=model_checkpoint,
        use_pretrained=config["zero_shot"]["use_pretrained_embeddings"]
    )
    af_predictor.load_model()
    
    # Get embeddings
    print("Extracting embeddings...")
    embeddings = af_predictor.get_embeddings(dataset.sequences)
    
    # Predict expression
    print("Predicting Sox2 expression levels...")
    expr_predictor = ExpressionPredictor()
    predictions = expr_predictor.predict(embeddings)
    
    # Ensure non-negative predictions
    predictions = np.maximum(predictions, 0)
    
    # Evaluate (if labels available)
    results = {"predictions": predictions}
    if dataset.expressions is not None:
        print("Evaluating predictions...")
        metrics = evaluate_predictions(dataset.expressions, predictions)
        results.update(metrics)
        print(f"Evaluation metrics: {metrics}")
    
    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if config["output"]["save_predictions"]:
        np.save(output_dir / "predictions.npy", predictions)
        print(f"Saved predictions to {output_dir / 'predictions.npy'}")
    
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
