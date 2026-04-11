"""
Data loading and preprocessing utilities for Sox2 expression prediction.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional


class SequenceDataset:
    """Load and manage DNA sequences with Sox2 expression labels."""

    # Valid DNA characters
    VALID_NUCLEOTIDES = set("ACGTN")

    def __init__(self, data_path: str):
        """
        Initialize dataset from file.
        
        Args:
            data_path: Path to data file (CSV with 'sequence' and 'expression' columns)
        """
        self.data_path = Path(data_path)
        self.data = None
        self.sequences = None
        self.expressions = None

    def load(self):
        """Load data from file."""
        self.data = pd.read_csv(self.data_path)
        self.sequences = self.data["sequence"].values
        
        if "expression" in self.data.columns:
            self.expressions = self.data["expression"].values
        
        return self

    def summary(self) -> Dict:
        """Return summary statistics of dataset."""
        summary_dict = {
            "num_sequences": len(self.sequences),
            "sequence_lengths": {
                "min": min(len(seq) for seq in self.sequences),
                "max": max(len(seq) for seq in self.sequences),
                "mean": np.mean([len(seq) for seq in self.sequences]),
            }
        }
        
        if self.expressions is not None:
            summary_dict.update({
                "expression_min": float(np.min(self.expressions)),
                "expression_max": float(np.max(self.expressions)),
                "expression_mean": float(np.mean(self.expressions)),
                "expression_std": float(np.std(self.expressions)),
            })
        
        return summary_dict

    def validate_sequences(self) -> Tuple[np.ndarray, List[int]]:
        """
        Validate sequences contain only ACGTN.
        
        Returns:
            (valid_sequences, indices_of_invalid)
        """
        invalid_indices = []
        valid_sequences = []
        
        for idx, seq in enumerate(self.sequences):
            seq_upper = seq.upper()
            if all(c in self.VALID_NUCLEOTIDES for c in seq_upper):
                valid_sequences.append(seq_upper)
            else:
                invalid_indices.append(idx)
        
        return np.array(valid_sequences), invalid_indices

    def filter_sequences(
        self,
        min_length: int = 128,
        max_length: int = 1_048_576,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[int]]:
        """
        Filter sequences by length.
        
        Args:
            min_length: Minimum sequence length
            max_length: Maximum sequence length (AlphaGenome max is 1 Mb)
            
        Returns:
            (filtered_sequences, filtered_expressions or None, excluded_indices)
        """
        valid_indices = []
        
        for idx, seq in enumerate(self.sequences):
            if min_length <= len(seq) <= max_length:
                valid_indices.append(idx)
        
        excluded_indices = [i for i in range(len(self.sequences)) if i not in valid_indices]
        
        filtered_sequences = self.sequences[valid_indices]
        filtered_expressions = None
        
        if self.expressions is not None:
            filtered_expressions = self.expressions[valid_indices]
        
        return filtered_sequences, filtered_expressions, excluded_indices

    def to_tensor_format(self, sequences: np.ndarray) -> np.ndarray:
        """
        Convert DNA sequences to one-hot encoded format.
        
        Args:
            sequences: Array of DNA sequences
            
        Returns:
            One-hot encoded array (batch, 4, sequence_length)
        """
        nucleotide_map = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
        
        max_length = max(len(seq) for seq in sequences)
        batch_size = len(sequences)
        
        # One-hot encoding (4 nucleotides + 1 for N)
        one_hot = np.zeros((batch_size, 4, max_length), dtype=np.float32)
        
        for i, seq in enumerate(sequences):
            seq = seq.upper()
            for j, nuc in enumerate(seq):
                if nuc != "N":
                    one_hot[i, nucleotide_map[nuc], j] = 1.0
                else:
                    # For N, set all to low value (ambiguous)
                    one_hot[i, :, j] = 0.25
        
        return one_hot


def parse_sequences(seq_list: List[str]) -> np.ndarray:
    """
    Normalize DNA sequences to uppercase ACGT.
    
    Args:
        seq_list: List of DNA sequences
        
    Returns:
        Normalized sequences
    """
    return np.array([seq.upper() for seq in seq_list])


def filter_sequences(sequences: List[str], min_length: int = 128, max_length: int = 1_048_576) -> Tuple[List[str], List[int]]:
    """
    Filter sequences by valid length range for AlphaGenome.
    
    Args:
        sequences: List of DNA sequences
        min_length: Minimum length
        max_length: Maximum length (AlphaGenome supports up to 1 Mb)
        
    Returns:
        (filtered_sequences, excluded_indices)
    """
    valid_sequences = []
    valid_indices = []
    
    for idx, seq in enumerate(sequences):
        if min_length <= len(seq) <= max_length:
            valid_sequences.append(seq)
            valid_indices.append(idx)
    
    excluded_indices = [i for i in range(len(sequences)) if i not in valid_indices]
    
    return valid_sequences, excluded_indices
