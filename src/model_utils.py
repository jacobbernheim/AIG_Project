"""
AlphaGenome model utilities for genomic predictions.
"""

from typing import Dict, Optional, List

import numpy as np
import torch
from alphagenome_pytorch import dna_model


class GenomeModel:
    """Wrapper for AlphaGenome model for sequence-based predictions."""

    # Available output tracks from AlphaGenome
    AVAILABLE_TRACKS = {
        "atac": "ATAC-seq peaks",
        "dnase": "DNase I hypersensitivity",
        "chip_histone": "ChIP-seq histone modifications",
        "chip_tf": "ChIP-seq transcription factors",
        "rna_seq": "RNA-seq expression",
    }
    
    # Organisms: 0=human, 1=mouse
    ORGANISM_MAP = {"human": 0, "mouse": 1}
    RESOLUTIONS = [1, 128]  # Base pair resolutions
    INPUT_LENGTH_MULTIPLE = 128
    MIN_INPUT_LENGTH = 2048

    def __init__(
        self,
        organism: str = "human",
        device: Optional[str] = None,
    ):
        """
        Initialize AlphaGenome model.
        
        Args:
            organism: "human" (default) or "mouse"
            device: "cuda", "cpu", or None for auto-detection
        """
        self.organism = organism.lower()
        self.organism_index = self.ORGANISM_MAP.get(self.organism)
        
        if self.organism_index is None:
            raise ValueError(f"Organism must be 'human' or 'mouse', got {organism}")
        
        # Device selection
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model = None
        self._is_loaded = False

    def load_model(self) -> None:
        """Create AlphaGenome model and move it to the selected device."""
        if self._is_loaded:
            print(f"Model already loaded on {self.device}")
            return

        print(f"Creating AlphaGenome model on {self.device}...")
        self.model = dna_model.create(
            add_reference_heads=True,
            device=self.device,
        )
        self.model.eval()
        
        self._is_loaded = True
        print("Model loaded successfully")

    @staticmethod
    def prepare_sequence_for_model(
        sequence: str,
        minimum_length: int = MIN_INPUT_LENGTH,
        multiple: int = INPUT_LENGTH_MULTIPLE,
    ) -> tuple[str, int]:
        """Pad a sequence so AlphaGenome can process it safely."""
        if minimum_length < 1:
            raise ValueError("minimum_length must be positive")
        if multiple < 1:
            raise ValueError("multiple must be positive")

        sequence = sequence.strip().upper()
        target_length = max(minimum_length, len(sequence))
        remainder = target_length % multiple
        if remainder:
            target_length += multiple - remainder

        pad_length = target_length - len(sequence)
        if pad_length <= 0:
            return sequence, 0

        return sequence + ("N" * pad_length), pad_length

    def predict_on_sequence(
        self,
        sequence: str,
        tracks: Optional[List[str]] = None,
        ontology_terms: Optional[List[str]] = None,
        resolution: int = 1,
    ) -> Dict[str, np.ndarray]:
        """
        Run inference on a single DNA sequence.
        
        Args:
            sequence: DNA sequence to predict on
            tracks: List of tracks to return (e.g., ["atac", "dnase"])
                   If None, returns all available tracks
            resolution: Output resolution (1 or 128 bp)
            
        Returns:
            Dictionary mapping track names to predictions (batch, sequence, tracks)
        """
        if not self._is_loaded:
            self.load_model()
        
        if resolution not in self.RESOLUTIONS:
            raise ValueError(f"Resolution must be 1 or 128, got {resolution}")

        requested_outputs = tracks if tracks is not None else list(self.AVAILABLE_TRACKS.keys())
        sequence, _ = self.prepare_sequence_for_model(sequence)

        # Run prediction directly on the raw DNA string.
        preds = self.model.predict(
            sequence,
            organism=self.organism,
            no_grad=True,
            requested_outputs=requested_outputs,
            ontology_terms=ontology_terms,
        )

        # Unwrap the organism level if present.
        if isinstance(preds, dict) and self.organism in preds:
            preds = preds[self.organism]

        # Convert outputs to numpy arrays for downstream handling.
        results = {}

        for track in requested_outputs:
            track_key = None
            for candidate in (track, track.lower(), track.upper(), self.AVAILABLE_TRACKS.get(track)):
                if candidate in preds:
                    track_key = candidate
                    break

            if track_key is None:
                print(f"Warning: Track '{track}' not available in model predictions")
                continue

            value = preds[track_key]
            if isinstance(value, dict):
                output_name = self.AVAILABLE_TRACKS.get(track)
                value = dna_model._select_head_output(value, output_name or track.upper())

            results[track] = self._to_numeric_array(value)
        
        return results

    @staticmethod
    def _to_numeric_array(value) -> np.ndarray:
        """Convert nested prediction outputs into a flat numeric numpy array."""
        if value is None:
            return np.asarray([])
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, dict):
            flattened_values = []
            for key in sorted(value.keys(), key=str):
                flattened_values.append(GenomeModel._to_numeric_array(value[key]).ravel())
            if not flattened_values:
                return np.asarray([])
            return np.concatenate(flattened_values)
        if isinstance(value, (list, tuple)):
            flattened_values = [GenomeModel._to_numeric_array(item).ravel() for item in value]
            if not flattened_values:
                return np.asarray([])
            return np.concatenate(flattened_values)
        return np.asarray(value)

    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            "organism": self.organism,
            "device": self.device,
            "loaded": self._is_loaded,
            "available_tracks": self.AVAILABLE_TRACKS,
            "resolutions": self.RESOLUTIONS,
        }


class ExpressionPredictor:
    """
    Head model for predicting Sox2 expression from AlphaGenome outputs.
    
    This will be trained/finetuned on Sox2 expression data in future steps.
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 128):
        """
        Initialize expression prediction head.
        
        Args:
            input_dim: Dimension of input features from AlphaGenome
            hidden_dim: Hidden layer dimension
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.model = None

    def aggregate_track_features(
        self,
        track_predictions: Dict[str, np.ndarray],
        aggregation: str = "mean",
    ) -> np.ndarray:
        """
        Aggregate features from multiple AlphaGenome tracks.
        
        Args:
            track_predictions: Dict from predict_on_sequence()
            aggregation: "mean", "concat", or "max"
            
        Returns:
            Aggregated features (batch, feature_dim)
        """
        if not track_predictions:
            raise ValueError("No track predictions provided")
        
        features_list = []
        for track_name, track_data in track_predictions.items():
            if track_data is None:
                continue

            track_data = np.asarray(track_data)
            if track_data.dtype == object:
                flattened = GenomeModel._to_numeric_array(track_data.item() if track_data.shape == () else track_data.tolist())
                track_data = np.asarray(flattened)

            flat = np.asarray(track_data).ravel()
            if flat.size == 0:
                continue

            if aggregation == "mean":
                agg_feat = np.array([[flat.mean()]])
            elif aggregation == "max":
                agg_feat = np.array([[flat.max()]])
            elif aggregation == "concat":
                agg_feat = flat.reshape(1, -1)
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")
            
            features_list.append(agg_feat)
        
        if aggregation == "concat":
            aggregated = np.concatenate(features_list, axis=1)
        else:
            aggregated = np.concatenate(features_list, axis=1)
        
        return aggregated

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict Sox2 expression from features.
        
        Args:
            features: Aggregated features from AlphaGenome
            
        Returns:
            Expression predictions (batch,)
        """
        # Placeholder for trained model prediction
        # For now, returns dummy predictions
        return np.random.uniform(0, 2, size=features.shape[0])
