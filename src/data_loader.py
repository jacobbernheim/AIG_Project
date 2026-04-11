"""
Data loading and preprocessing utilities for Sox2 expression prediction.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict


class SequenceDataset:
    """Load and manage DNA sequences with Sox2 expression labels."""

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
        self.expressions = self.data["expression"].values
        return self

    def summary(self) -> Dict:
        """Return summary statistics of dataset."""
        return {
            "num_sequences": len(self.sequences),
            "expression_min": float(np.min(self.expressions)),
            "expression_max": float(np.max(self.expressions)),
            "expression_mean": float(np.mean(self.expressions)),
            "expression_std": float(np.std(self.expressions)),
        }


def parse_sequences(seq_list: List[str]) -> np.ndarray:
    """Convert DNA sequences to normalized format."""
    # Placeholder for sequence parsing/tokenization
    pass


def filter_sequences(sequences: List[str], min_length: int = 50) -> List[str]:
    """Filter sequences by length."""
    return [seq for seq in sequences if len(seq) >= min_length]
