"""
AlphaGenome model utilities for genomic predictions.

Uses the raw model() forward pass to get resolution-keyed dicts.
Cell-type filtering uses track metadata parquet/CSV files.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List, Mapping, Any

import numpy as np
import torch
import pandas as pd

try:
    from alphagenome_pytorch import AlphaGenome
except ImportError:
    AlphaGenome = None

try:
    from alphagenome_pytorch import dna_model
except ImportError:
    dna_model = None


# ---------------------------------------------------------------------------
# Track metadata
# ---------------------------------------------------------------------------

DEFAULT_TRACK_METADATA_DIR = Path(__file__).parent

TRACK_METADATA_FILES = {
    "human": [
        "track_metadata_human.parquet",
        "alphagenome_track_celltypes_human.csv",
    ],
    "mouse": [
        "track_metadata_mouse.parquet",
        "alphagenome_track_celltypes_mouse.csv",
    ],
}

OUTPUT_TYPE_TO_TRACK_KEY = {
    "OutputType.ATAC": "atac",
    "OutputType.DNASE": "dnase",
    "OutputType.CAGE": "cage",
    "OutputType.RNA_SEQ": "rna_seq",
    "OutputType.PROCAP": "procap",
    "OutputType.CHIP_TF": "chip_tf",
    "OutputType.CHIP_HISTONE": "chip_histone",
    "OutputType.CONTACT_MAPS": "contact_maps",
    "OutputType.SPLICE_SITES": "splice_sites",
    "OutputType.SPLICE_JUNCTIONS": "splice_junctions",
    "OutputType.SPLICE_SITE_USAGE": "splice_site_usage",
}


class TrackMetadata:
    """Loads and queries track-to-cell-type mapping from parquet or CSV."""

    def __init__(self, csv_path: str | Path | None = None, organism: str = "human"):
        self.organism = organism.lower()
        if csv_path:
            self.file_path = Path(csv_path)
        else:
            self.file_path = self._find_metadata_file(self.organism)
        self._df: Optional[pd.DataFrame] = None

    @staticmethod
    def _find_metadata_file(organism: str) -> Optional[Path]:
        candidates = TRACK_METADATA_FILES.get(organism, [])
        for filename in candidates:
            path = DEFAULT_TRACK_METADATA_DIR / filename
            if path.exists():
                return path
        return None

    def _load(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df
        if self.file_path is None or not self.file_path.exists():
            if self.file_path:
                print(f"  [warn] Track metadata not found: {self.file_path}")
            else:
                print(f"  [warn] No track metadata for organism '{self.organism}'")
            self._df = pd.DataFrame()
            return self._df

        if self.file_path.suffix == ".parquet":
            self._df = pd.read_parquet(self.file_path)
        else:
            self._df = pd.read_csv(self.file_path)

        if "index" in self._df.columns and "track_index" not in self._df.columns:
            self._df = self._df.rename(columns={"index": "track_index"})
        if "Assay title" in self._df.columns and "assay_title" not in self._df.columns:
            self._df = self._df.rename(columns={"Assay title": "assay_title"})

        if "output_type" in self._df.columns:
            self._df["track_key"] = self._df["output_type"].map(
                lambda x: OUTPUT_TYPE_TO_TRACK_KEY.get(x, x.lower() if isinstance(x, str) else x)
            )
        else:
            self._df["track_key"] = None

        print(f"  Loaded track metadata: {len(self._df)} rows from {self.file_path}")
        return self._df

    def get_channel_indices(self, ontology_curie: str, track_key: str) -> Optional[List[int]]:
        df = self._load()
        if df.empty:
            return None
        mask = (df["ontology_curie"] == ontology_curie) & (df["track_key"] == track_key)
        matches = df.loc[mask]
        if matches.empty:
            return None
        return sorted(matches["track_index"].astype(int).tolist())

    def get_channel_indices_by_mark(self, ontology_curie: str, track_key: str,
                                     histone_mark: str) -> Optional[List[int]]:
        df = self._load()
        if df.empty or "histone_mark" not in df.columns:
            return None
        mask = (
            (df["ontology_curie"] == ontology_curie)
            & (df["track_key"] == track_key)
            & (df["histone_mark"].str.contains(histone_mark, case=False, na=False))
        )
        matches = df.loc[mask]
        if matches.empty:
            return None
        return sorted(matches["track_index"].astype(int).tolist())

    def get_channel_indices_by_tf(self, ontology_curie: str,
                                   transcription_factor: Optional[str] = None) -> Optional[List[int]]:
        df = self._load()
        if df.empty:
            return None
        mask = (df["ontology_curie"] == ontology_curie) & (df["track_key"] == "chip_tf")
        if transcription_factor and "transcription_factor" in df.columns:
            mask = mask & df["transcription_factor"].str.contains(
                transcription_factor, case=False, na=False
            )
        matches = df.loc[mask]
        if matches.empty:
            return None
        return sorted(matches["track_index"].astype(int).tolist())

    def get_all_channels_for_curie(self, ontology_curie: str) -> Dict[str, List[int]]:
        df = self._load()
        if df.empty:
            return {}
        mask = df["ontology_curie"] == ontology_curie
        matches = df.loc[mask]
        if matches.empty:
            return {}
        result: Dict[str, List[int]] = {}
        for track_key, group in matches.groupby("track_key"):
            if pd.isna(track_key):
                continue
            result[track_key] = sorted(group["track_index"].astype(int).tolist())
        return result

    def search_biosample(self, search_term: str) -> pd.DataFrame:
        df = self._load()
        if df.empty:
            return df
        mask = pd.Series(False, index=df.index)
        for col in ["biosample_name", "ontology_curie", "track_name"]:
            if col in df.columns:
                mask = mask | df[col].astype(str).str.contains(search_term, case=False, na=False)
        display_cols = [c for c in ["track_index", "ontology_curie", "biosample_name",
                                     "output_type", "track_key", "strand",
                                     "histone_mark", "transcription_factor"]
                        if c in df.columns]
        return df.loc[mask, display_cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# GenomeModel
# ---------------------------------------------------------------------------

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
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
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

    @staticmethod
    def prepare_sequence_for_model(sequence: str, minimum_length: int = MIN_INPUT_LENGTH,
                                    multiple: int = INPUT_LENGTH_MULTIPLE) -> tuple[str, int]:
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
        left_pad = pad_length // 2
        right_pad = pad_length - left_pad
        return ("N" * left_pad) + sequence + ("N" * right_pad), pad_length

    @staticmethod
    def _sequence_to_one_hot(sequence: str, device: str) -> torch.Tensor:
        base_to_channel = {"A": 0, "C": 1, "G": 2, "T": 3}
        one_hot = np.zeros((1, len(sequence), 4), dtype=np.float32)
        for index, base in enumerate(sequence.upper()):
            channel_index = base_to_channel.get(base)
            if channel_index is not None:
                one_hot[0, index, channel_index] = 1.0
        tensor = torch.from_numpy(one_hot)
        if device:
            tensor = tensor.to(device)
        return tensor

    def _strip_padding_channels(self, tensor: torch.Tensor, track_name: str) -> torch.Tensor:
        real_count = self.REAL_TRACK_COUNTS.get(track_name)
        if real_count is None:
            return tensor
        if tensor.ndim >= 2 and tensor.shape[-1] > real_count:
            return tensor[..., :real_count]
        return tensor

    def _filter_by_ontology(self, tensor: torch.Tensor, track_name: str,
                             ontology_terms: List[str]) -> tuple[torch.Tensor, Optional[List[int]]]:
        for curie in ontology_terms:
            indices = self.track_metadata.get_channel_indices(curie, track_name)
            if indices is not None and len(indices) > 0:
                max_channel = tensor.shape[-1]
                valid = [i for i in indices if 0 <= i < max_channel]
                if valid:
                    return tensor[..., valid], valid
        return tensor, None

    def predict_on_sequence(self, sequence: str, tracks: Optional[List[str]] = None,
                             ontology_terms: Optional[List[str]] = None,
                             resolution: int = 1, preserve_raw: bool = False,
                             ) -> Dict[str, np.ndarray]:
        """Run inference with optional ontology filtering."""
        if not self._is_loaded:
            self.load_model()
        if resolution not in self.RESOLUTIONS:
            raise ValueError(f"Resolution must be 1 or 128, got {resolution}")

        requested_outputs = tracks if tracks is not None else list(self.AVAILABLE_TRACKS.keys())
        sequence, _ = self.prepare_sequence_for_model(sequence)

        if ontology_terms:
            for curie in ontology_terms:
                all_channels = self.track_metadata.get_all_channels_for_curie(curie)
                if all_channels:
                    print(f"  Cell-type filter: {curie}")
                    for tk, idxs in all_channels.items():
                        if tk in requested_outputs:
                            print(f"    {tk}: channel(s) {idxs}")
                else:
                    print(f"  [warn] No channels found for '{curie}'")

        dna_one_hot = self._sequence_to_one_hot(sequence, self.device)
        organism_tensor = torch.tensor(
            [self.organism_index], dtype=torch.long, device=dna_one_hot.device
        )
        with torch.no_grad():
            outputs = self.model(dna_one_hot, organism_tensor)

        results = {}
        for track in requested_outputs:
            if track not in outputs:
                print(f"  [warn] Track '{track}' not in model outputs.")
                continue
            track_output = outputs[track]
            if isinstance(track_output, dict):
                if resolution in track_output:
                    tensor = track_output[resolution]
                elif str(resolution) in track_output:
                    tensor = track_output[str(resolution)]
                else:
                    available = sorted(track_output.keys())
                    tensor = track_output[available[0]]
            elif isinstance(track_output, torch.Tensor):
                tensor = track_output
            else:
                continue
            tensor = self._strip_padding_channels(tensor, track)
            if ontology_terms:
                tensor, _ = self._filter_by_ontology(tensor, track, ontology_terms)
            if preserve_raw:
                results[track] = tensor
            else:
                arr = tensor.detach().cpu().float().numpy()
                print(f"  {track}: shape={arr.shape}, "
                      f"min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")
                results[track] = arr

        return results

    def predict_on_sequence_raw(self, sequence: str, tracks: Optional[List[str]] = None,
                                 resolution: int = 128) -> Dict[str, torch.Tensor]:
        """Run raw forward pass, returning unfiltered tensors (padding channels stripped)."""
        if not self._is_loaded:
            self.load_model()
        if resolution not in self.RESOLUTIONS:
            raise ValueError(f"Resolution must be 1 or 128, got {resolution}")

        requested_outputs = tracks if tracks is not None else list(self.AVAILABLE_TRACKS.keys())
        sequence, _ = self.prepare_sequence_for_model(sequence)

        dna_one_hot = self._sequence_to_one_hot(sequence, self.device)
        organism_tensor = torch.tensor(
            [self.organism_index], dtype=torch.long, device=dna_one_hot.device
        )
        with torch.no_grad():
            outputs = self.model(dna_one_hot, organism_tensor)

        results = {}
        for track in requested_outputs:
            if track not in outputs:
                continue
            track_output = outputs[track]
            if isinstance(track_output, dict):
                if resolution in track_output:
                    tensor = track_output[resolution]
                elif str(resolution) in track_output:
                    tensor = track_output[str(resolution)]
                else:
                    available = sorted(track_output.keys())
                    tensor = track_output[available[0]]
            elif isinstance(track_output, torch.Tensor):
                tensor = track_output
            else:
                continue
            tensor = self._strip_padding_channels(tensor, track)
            results[track] = tensor

        return results

    def get_model_info(self) -> Dict:
        return {
            "organism": self.organism,
            "device": self.device,
            "loaded": self._is_loaded,
            "available_tracks": self.AVAILABLE_TRACKS,
            "resolutions": self.RESOLUTIONS,
        }


# ---------------------------------------------------------------------------
# Expression predictor (placeholder)
# ---------------------------------------------------------------------------

class ExpressionPredictor:
    def __init__(self, input_dim: int = 256, hidden_dim: int = 128):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.model = None

    def aggregate_track_features(self, track_predictions: Dict[str, np.ndarray],
                                  aggregation: str = "mean") -> np.ndarray:
        if not track_predictions:
            raise ValueError("No track predictions provided")
        features_list = []
        for _, v in track_predictions.items():
            if v is None:
                continue
            flat = np.asarray(v).ravel()
            if flat.size == 0:
                continue
            if aggregation == "mean":
                features_list.append(np.array([[flat.mean()]]))
            elif aggregation == "max":
                features_list.append(np.array([[flat.max()]]))
            elif aggregation == "concat":
                features_list.append(flat.reshape(1, -1))
        return np.concatenate(features_list, axis=1)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.random.uniform(0, 2, size=features.shape[0])


# ---------------------------------------------------------------------------
# Zero-shot scorer — mESC-specific (ES-Bruce4 + ES-CJ7 DNase)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ZeroShotScoreWeights:
    """Weights for the zero-shot scoring formula.

    ATAC dropped (no mESC ATAC in mouse model).
    DNase uses ES-CJ7 (EFO:0005916) track_index 23, weight 1.0.
    Histone marks and EP300 use ES-Bruce4 (EFO:0005483).
    TF reduced to EP300 only (track_index 89) — no entropy.
    """
    dnase: float = 1.0
    h3k27ac: float = 1.0
    h3k4me1: float = 0.8
    ep300: float = 0.6


@dataclass(frozen=True)
class ZeroShotScore:
    """Container for the raw score and its component breakdown."""
    raw_score: float
    component_values: Dict[str, float]


# mESC channel indices from track_metadata_mouse.parquet
MESC_DNASE_INDICES = [23]      # ES-CJ7 DNase
MESC_H3K27AC_INDICES = [50]    # ES-Bruce4 chip_histone H3K27ac
MESC_H3K4ME1_INDICES = [51]    # ES-Bruce4 chip_histone H3K4me1
MESC_EP300_INDICES = [89]      # ES-Bruce4 chip_tf EP300


class ZeroShotScorer:
    """mESC-specific Sox2 scorer using explicit channel indices.

    Scoring uses SIGNAL MASS (sum of positive signal across all bins), which
    naturally scales with sequence length:
    - A 30kb sequence with three enhancers scores ~3x a 10kb single enhancer
    - N-padded regions produce near-zero signal and contribute minimally
    - An optional signal_threshold can exclude low background noise

    Components:
    - DNase: ES-CJ7 channel 23 (chromatin accessibility)
    - H3K27ac: ES-Bruce4 channel 50 (active enhancer/promoter mark)
    - H3K4me1: ES-Bruce4 channel 51 (enhancer mark)
    - EP300: ES-Bruce4 channel 89 (enhancer-associated coactivator)
    """

    def __init__(
        self,
        weights: Optional[ZeroShotScoreWeights] = None,
        dnase_indices: Optional[List[int]] = None,
        h3k27ac_indices: Optional[List[int]] = None,
        h3k4me1_indices: Optional[List[int]] = None,
        ep300_indices: Optional[List[int]] = None,
        signal_threshold: float = 0.0,
    ):
        """
        Args:
            weights: Component weights for the scoring formula.
            dnase_indices: Channel indices for DNase (default: ES-CJ7 ch 23).
            h3k27ac_indices: Channel indices for H3K27ac (default: ES-Bruce4 ch 50).
            h3k4me1_indices: Channel indices for H3K4me1 (default: ES-Bruce4 ch 51).
            ep300_indices: Channel indices for EP300 (default: ES-Bruce4 ch 89).
            signal_threshold: Minimum signal value to count as "active".
                Bins below this are treated as zero. Set to 0.0 (default) to
                count all positive signal. Set higher (e.g. 0.01) to ignore
                low-level background from N-padded regions.
        """
        self.weights = weights or ZeroShotScoreWeights()
        self.dnase_indices = dnase_indices or MESC_DNASE_INDICES
        self.h3k27ac_indices = h3k27ac_indices or MESC_H3K27AC_INDICES
        self.h3k4me1_indices = h3k4me1_indices or MESC_H3K4ME1_INDICES
        self.ep300_indices = ep300_indices or MESC_EP300_INDICES
        self.signal_threshold = signal_threshold

    @staticmethod
    def _slice_channels(tensor_or_array, indices: List[int]) -> np.ndarray:
        """Slice specific channel indices from the last dimension."""
        if torch.is_tensor(tensor_or_array):
            arr = tensor_or_array.detach().cpu().float().numpy()
        else:
            arr = np.asarray(tensor_or_array, dtype=float)
        if arr.ndim < 1 or arr.size == 0:
            return arr
        max_ch = arr.shape[-1]
        valid = [i for i in indices if 0 <= i < max_ch]
        if not valid:
            print(f"  [warn] Channel indices {indices} out of range (max={max_ch - 1})")
            return arr
        return arr[..., valid]

    def _signal_mass(self, array: np.ndarray) -> float:
        """Compute total signal mass: sum of all values above threshold.

        This is the key metric that scales with sequence length:
        - Each 128bp bin contributes its signal value to the sum
        - More bins with active signal = higher mass
        - Three enhancers produce ~3x the mass of one enhancer
        - N-padded regions contribute near-zero (below threshold if set)

        Returns:
            Total positive signal summed across all bins.
        """
        flat = array.ravel().astype(float)
        # Apply threshold: zero out bins below threshold
        flat = np.where(flat >= self.signal_threshold, flat, 0.0)
        # Clip negatives (shouldn't happen after threshold but be safe)
        flat = np.clip(flat, 0.0, None)
        return float(flat.sum())

    def _signal_stats(self, array: np.ndarray) -> Dict[str, float]:
        """Compute signal statistics for logging."""
        flat = array.ravel().astype(float)
        positive = flat[flat > self.signal_threshold]
        return {
            "mass": self._signal_mass(array),
            "n_bins_total": int(flat.size),
            "n_bins_active": int(positive.size),
            "frac_active": float(positive.size / max(flat.size, 1)),
            "mean_active": float(positive.mean()) if positive.size > 0 else 0.0,
            "max": float(flat.max()) if flat.size > 0 else 0.0,
        }

    def score(self, raw_track_outputs: Dict[str, Any]) -> ZeroShotScore:
        """Compute the mESC-specific Sox2 score from raw AlphaGenome outputs.

        The score uses SIGNAL MASS (total sum of positive signal) for each
        component, which naturally rewards sequences with more enhancer
        content spread over more bins.

        Args:
            raw_track_outputs: Dict from predict_on_sequence_raw() containing
                full (unfiltered) tensors for 'dnase', 'chip_histone', 'chip_tf'.

        Returns:
            ZeroShotScore with raw_score and component breakdown.
        """
        if self.signal_threshold > 0:
            print(f"  [scorer] Signal threshold: {self.signal_threshold}")

        # --- DNase: ES-CJ7 channel 23 ---
        dnase_raw = raw_track_outputs.get("dnase")
        if dnase_raw is not None:
            dnase_arr = self._slice_channels(dnase_raw, self.dnase_indices)
            dnase_stats = self._signal_stats(dnase_arr)
            dnase_mass = dnase_stats["mass"]
            print(f"  [scorer] DNase (ES-CJ7 ch{self.dnase_indices}): "
                  f"mass={dnase_mass:.4f}, "
                  f"{dnase_stats['n_bins_active']}/{dnase_stats['n_bins_total']} bins active "
                  f"({dnase_stats['frac_active']:.1%}), "
                  f"mean_active={dnase_stats['mean_active']:.4f}")
        else:
            dnase_mass = 0.0
            print("  [scorer] DNase: not available")

        # --- H3K27ac: ES-Bruce4 channel 50 ---
        histone_raw = raw_track_outputs.get("chip_histone")
        if histone_raw is not None:
            h3k27ac_arr = self._slice_channels(histone_raw, self.h3k27ac_indices)
            h3k27ac_stats = self._signal_stats(h3k27ac_arr)
            h3k27ac_mass = h3k27ac_stats["mass"]
            print(f"  [scorer] H3K27ac (ES-Bruce4 ch{self.h3k27ac_indices}): "
                  f"mass={h3k27ac_mass:.4f}, "
                  f"{h3k27ac_stats['n_bins_active']}/{h3k27ac_stats['n_bins_total']} bins active "
                  f"({h3k27ac_stats['frac_active']:.1%})")
        else:
            h3k27ac_mass = 0.0
            print("  [scorer] H3K27ac: not available")

        # --- H3K4me1: ES-Bruce4 channel 51 ---
        if histone_raw is not None:
            h3k4me1_arr = self._slice_channels(histone_raw, self.h3k4me1_indices)
            h3k4me1_stats = self._signal_stats(h3k4me1_arr)
            h3k4me1_mass = h3k4me1_stats["mass"]
            print(f"  [scorer] H3K4me1 (ES-Bruce4 ch{self.h3k4me1_indices}): "
                  f"mass={h3k4me1_mass:.4f}, "
                  f"{h3k4me1_stats['n_bins_active']}/{h3k4me1_stats['n_bins_total']} bins active "
                  f"({h3k4me1_stats['frac_active']:.1%})")
        else:
            h3k4me1_mass = 0.0
            print("  [scorer] H3K4me1: not available")

        # --- EP300: ES-Bruce4 channel 89 ---
        tf_raw = raw_track_outputs.get("chip_tf")
        if tf_raw is not None:
            ep300_arr = self._slice_channels(tf_raw, self.ep300_indices)
            ep300_stats = self._signal_stats(ep300_arr)
            ep300_mass = ep300_stats["mass"]
            print(f"  [scorer] EP300 (ES-Bruce4 ch{self.ep300_indices}): "
                  f"mass={ep300_mass:.4f}, "
                  f"{ep300_stats['n_bins_active']}/{ep300_stats['n_bins_total']} bins active "
                  f"({ep300_stats['frac_active']:.1%})")
        else:
            ep300_mass = 0.0
            print("  [scorer] EP300: not available")

        component_values = {
            "dnase": dnase_mass,
            "h3k27ac": h3k27ac_mass,
            "h3k4me1": h3k4me1_mass,
            "ep300": ep300_mass,
        }

        raw_score = (
            self.weights.dnase * component_values["dnase"]
            + self.weights.h3k27ac * component_values["h3k27ac"]
            + self.weights.h3k4me1 * component_values["h3k4me1"]
            + self.weights.ep300 * component_values["ep300"]
        )

        print(f"  [scorer] Raw score: {raw_score:.6f}")

        return ZeroShotScore(raw_score=float(raw_score), component_values=component_values)