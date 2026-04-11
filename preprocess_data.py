"""
Data preprocessing and formatting script.

Reads raw CSV files from data/raw/ and outputs formatted data to data/processed/.

Usage:
    python preprocess_data.py [--raw-dir data/raw/] [--processed-dir data/processed/]
"""

import argparse
from pathlib import Path
import pandas as pd

from src.data_loader import SequenceDataset


def preprocess_data(
    raw_dir: str = "data/raw/",
    processed_dir: str = "data/processed/",
    sequences_file: str = "Payload Sequences.csv",
    activities_file: str = "Payload Activities.csv",
) -> None:
    """
    Load raw data, format it, and save to processed directory.
    
    Args:
        raw_dir: Directory containing raw CSV files
        processed_dir: Directory to save processed data
        sequences_file: Name of sequences file
        activities_file: Name of activities file
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    sequences_path = raw_dir / sequences_file
    activities_path = raw_dir / activities_file
    
    print(f"Loading raw data from {raw_dir}...")
    
    # Load raw data
    if not sequences_path.exists():
        raise FileNotFoundError(f"Sequences file not found: {sequences_path}")
    if not activities_path.exists():
        raise FileNotFoundError(f"Activities file not found: {activities_path}")
    
    sequences_df = pd.read_csv(sequences_path)
    activities_df = pd.read_csv(activities_path)
    
    print(f"Loaded {len(sequences_df)} sequences")
    print(f"Loaded {len(activities_df)} activity records")
    
    # TODO: Add formatting logic here
    # - Merge/align sequences with activities
    # - Filter/validate sequences
    # - Normalize expression values
    # - Handle missing data
    # - Any other preprocessing steps
    
    # For now, save raw data as processed (placeholder)
    sequences_df.to_csv(processed_dir / "sequences_processed.csv", index=False)
    activities_df.to_csv(processed_dir / "activities_processed.csv", index=False)
    
    print(f"Saved processed data to {processed_dir}")
    print(f"  - sequences_processed.csv")
    print(f"  - activities_processed.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess and format raw data files"
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw/",
        help="Directory containing raw CSV files"
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed/",
        help="Directory to save processed data"
    )
    parser.add_argument(
        "--sequences-file",
        type=str,
        default="Payload Sequences.csv",
        help="Name of sequences file in raw directory"
    )
    parser.add_argument(
        "--activities-file",
        type=str,
        default="Payload Activities.csv",
        help="Name of activities file in raw directory"
    )
    
    args = parser.parse_args()
    
    preprocess_data(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        sequences_file=args.sequences_file,
        activities_file=args.activities_file,
    )
