from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List, Mapping, Any

import numpy as np
import torch
import pandas as pd

from src.track_metadata import TrackMetadata

try:
    from alphagenome_pytorch import AlphaGenome
except ImportError:
    AlphaGenome = None

try:
    from alphagenome_pytorch import dna_model
except ImportError:
    dna_model = None


class GenomeModel:
    """Wrapper for AlphaGenome using raw model() forward pass."""

    AVAILABLE_TRACKS = {
        "atac": "ATAC-seq peaks",
        "dnase": "DNase I hypersensitivity",
        "chip_histone": "ChIP-seq histone modifications",
        "chip_tf": "ChIP-seq transcription factors",
        "rna_seq": "RNA-seq expression",
        "cage": "CAGE",
        "procap": "PRO-cap",
    }

    ORGANISM_MAP = {"human": 0, "mouse": 1}
    RESOLUTIONS = [1, 128]
    INPUT_LENGTH_MULTIPLE = 2048
    MIN_INPUT_LENGTH = 4096

    REAL_TRACK_COUNTS = {
        "atac": 167, "dnase": 305, "procap": 12, "cage": 546,
        "rna_seq": 667, "chip_tf": 1617, "chip_histone": 1116,
        "contact_maps": 28, "splice_sites": 5,
        "splice_junctions": 734, "splice_site_usage": 734,
    }

    OUTPUT_RESOLUTIONS = {
        "atac": [1, 128], "dnase": [1, 128], "procap": [1, 128],
        "cage": [1, 128], "rna_seq": [1, 128], "chip_tf": [128],
        "chip_histone": [128], "contact_maps": [128],
        "splice_sites": [1], "splice_junctions": [1], "splice_site_usage": [1],
    }

    def __init__(self, organism: str = "human", device: Optional[str] = None,
                 track_metadata_path: Optional[str | Path] = None):
        self.organism = organism.lower()
        self.organism_index = self.ORGANISM_MAP.get(self.organism)
        if self.organism_index is None:
            raise ValueError(f"Organism must be 'human' or 'mouse', got {organism}")
        if device is None or device == "cpu":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        print(f"  Using device: {self.device} (CUDA available: {torch.cuda.is_available()})")

        self.model = None
        self._is_loaded = False
        self.track_metadata = TrackMetadata(csv_path=track_metadata_path, organism=self.organism)

    def load_model(self) -> None:
        if self._is_loaded:
            print(f"Model already loaded on {self.device}")
            return
        print(f"Creating AlphaGenome model on {self.device}...")
        if AlphaGenome is not None:
            model = None
            attempts = [
                ("constructor_with_device", lambda: AlphaGenome(device=self.device)),
                ("constructor_no_args", lambda: AlphaGenome()),
            ]
            from_pretrained = getattr(AlphaGenome, "from_pretrained", None)
            if callable(from_pretrained):
                attempts.insert(0, ("from_pretrained",
                                    lambda: from_pretrained("alphagenome.pt", device=self.device)))
            for name, loader in attempts:
                try:
                    model = loader()
                    print(f"  Loaded via: {name}")
                    break
                except Exception:
                    continue
            if model is not None:
                self.model = model
                self.model = self.model.to(self.device)
                print(f"adding {self.organism} reference heads", flush=True)
                self.model.add_reference_heads(organism=self.organism)
                self.model = self.model.to(self.device)
            elif dna_model is not None:
                self.model = dna_model.create(add_reference_heads=True, device=self.device)
            else:
                raise RuntimeError("Unable to load AlphaGenome model")
        else:
            if dna_model is None:
                raise RuntimeError("AlphaGenome not available")
            self.model = dna_model.create(add_reference_heads=True, device=self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()
        self._is_loaded = True
        print("Model loaded successfully")

    def get_model_info(self) -> dict:
        """Return basic model information."""
        info = {
            "organism": self.organism,
            "device": str(self.device),
        }
        if self.model is not None:
            heads = list(self.model.heads.get(self.organism, {}).keys())
            info["available_heads"] = heads
        return info

    def encode_sequence(self, sequence: str, pad_to_multiple_of: int = 2048) -> torch.Tensor:
        """Encode DNA sequence to integer tensor with padding.
        
        A=0, C=1, G=2, T=3, N=-1 (masked to zero internally by model).
        Pads sequence to nearest multiple of pad_to_multiple_of using N (-1).
        
        Args:
            sequence: DNA string
            pad_to_multiple_of: Pad length to be divisible by this value.
                            2048 is safe for the transformer_unet architecture
                            (accounts for multiple downsampling stages of 2x, 4x, 8x, 16x).
        
        Returns:
            Integer tensor of shape (padded_seq_len,)
        """
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        encoded = [mapping.get(base.upper(), -1) for base in sequence]
        
        # Pad to nearest multiple
        seq_len = len(encoded)
        if seq_len % pad_to_multiple_of != 0:
            pad_len = pad_to_multiple_of - (seq_len % pad_to_multiple_of)
            # Pad with -1 (N) — model clamps to 0 then one-hots, so these become zero vectors
            encoded.extend([-1] * pad_len)
        
        return torch.tensor(encoded, dtype=torch.long)

    def predict_on_sequence_raw(
        self,
        sequence: str,
        tracks: list[str] = None,
        resolution: int = 128,
    ) -> dict[str, np.ndarray]:
        """
        Run AlphaGenome on a DNA sequence and return raw predictions.

        Args:
            sequence: DNA string (ACGT characters)
            tracks: List of assay types to return, e.g. ["dnase", "chip_histone", "chip_tf"]
            resolution: Prediction resolution (128 for chip_histone/chip_tf)

        Returns:
            Dict mapping track names to numpy arrays of shape (n_bins, n_tracks_per_assay)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if tracks is None:
            tracks = ["dnase", "chip_histone", "chip_tf"]

        # Encode and prepare input
        input_tensor = self.encode_sequence(sequence).unsqueeze(0).to(self.device)

        # Organism index: 0=human, 1=mouse
        organism_idx = 1 if self.organism == "mouse" else 0
        organism_tensor = torch.tensor([organism_idx], dtype=torch.long).to(self.device)

        # Run inference
        with torch.no_grad():
            self.model.eval()
            output = self.model(input_tensor, organism_tensor)

        # Output is nested under organism key
        organism_output = output[self.organism]

        # Extract requested tracks at the desired resolution
        results = {}
        for track_name in tracks:
            if track_name not in organism_output:
                print(f"  [warn] Track '{track_name}' not found. "
                    f"Available: {list(organism_output.keys()) if isinstance(organism_output, dict) else 'N/A'}")
                continue

            pred = organism_output[track_name]

            if isinstance(pred, dict):
                resolution_key = f"resolution_{resolution}"
                if resolution_key in pred:
                    tensor = pred[resolution_key]
                else:
                    available = list(pred.keys())
                    print(f"  [info] '{track_name}': using {available[-1]} "
                        f"(requested {resolution_key})")
                    tensor = pred[available[-1]]
                results[track_name] = tensor[0].cpu().numpy()
            elif hasattr(pred, 'shape'):
                results[track_name] = pred[0].cpu().numpy()

        return results
