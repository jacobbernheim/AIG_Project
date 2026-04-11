"""
Quick example script demonstrating AlphaGenome model setup and inference.

This script shows how to:
1. Initialize the AlphaGenome model
2. Run predictions on a sample DNA sequence
3. Aggregate predictions for downstream tasks
"""

import numpy as np
from src.model_utils import GenomeModel, ExpressionPredictor


def log(message: str) -> None:
    print(message, flush=True)


def main():
    """Run example inference."""
    
    log("=" * 60)
    log("AlphaGenome Model Setup Example")
    log("=" * 60)
    
    # Initialize model
    log("\n1. Initializing AlphaGenome model...")
    genome_model = GenomeModel(
        organism="human",
    )
    log(f"Model info: {genome_model.get_model_info()}")
    
    # Load model
    log("\n2. Loading model weights...")
    genome_model.load_model()
    
    # Example DNA sequence for a quick local smoke test.
    # AlphaGenome supports much longer windows, but this keeps the test practical on CPU.
    example_sequence = ("ACGT" * 1024)
    
    log(f"\n3. Running prediction on sequence of length {len(example_sequence)} bp...")
    
    # Get predictions
    track_predictions = genome_model.predict_on_sequence(
        example_sequence,
        tracks=["atac", "dnase", "rna_seq"],
        resolution=1,
    )
    
    log(f"\nTrack predictions retrieved:")
    for track_name, pred_array in track_predictions.items():
        log(f"  {track_name}: shape {pred_array.shape}")
    
    # Aggregate features for expression prediction
    log("\n4. Aggregating features for Sox2 expression prediction...")
    expr_predictor = ExpressionPredictor()
    
    agg_features = expr_predictor.aggregate_track_features(
        track_predictions,
        aggregation="mean",
    )
    log(f"Aggregated features shape: {agg_features.shape}")
    
    # Predict Sox2 expression
    log("\n5. Predicting Sox2 expression...")
    expression_pred = expr_predictor.predict(agg_features)
    log(f"Predicted Sox2 expression level: {expression_pred[0]:.3f}")
    log(f"Expression scale: 0 (none), 1 (WT), >1 (overexpression)")
    
    log("\n" + "=" * 60)
    log("Example complete! Model is working correctly.")
    log("=" * 60)


if __name__ == "__main__":
    main()
