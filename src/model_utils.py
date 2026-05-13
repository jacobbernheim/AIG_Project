"""
scoring utils for the model
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List, Mapping, Any

import numpy as np
import torch
import pandas as pd


@dataclass(frozen=True)
class ZeroShotScoreWeights:
    """ weights for channels """
    dnase: float = 1.0
    h3k27ac: float = 1.0
    h3k4me1: float = 0.8
    ep300: float = 0.6


@dataclass(frozen=True)
class ZeroShotScore:
    """ score with breakdown """
    raw_score: float
    component_values: Dict[str, float]


# mESC channel indices
# ES-CJ7 DNase
MESC_DNASE_INDICES = [23]
# ES-Bruce4 chip_histone H3K27ac    
MESC_H3K27AC_INDICES = [50]
# ES-Bruce4 chip_histone H3K4me1
MESC_H3K4ME1_INDICES = [51]
# ES-Bruce4 chip_tf EP300
MESC_EP300_INDICES = [89]


class ZeroShotScorer:
    """ scores using specific channels """

    def __init__(
        self,
        weights: Optional[ZeroShotScoreWeights] = None,
        dnase_indices: Optional[List[int]] = None,
        h3k27ac_indices: Optional[List[int]] = None,
        h3k4me1_indices: Optional[List[int]] = None,
        ep300_indices: Optional[List[int]] = None,
        signal_threshold: float = 0.0,
    ):
        self.weights = weights or ZeroShotScoreWeights()
        self.dnase_indices = dnase_indices or MESC_DNASE_INDICES
        self.h3k27ac_indices = h3k27ac_indices or MESC_H3K27AC_INDICES
        self.h3k4me1_indices = h3k4me1_indices or MESC_H3K4ME1_INDICES
        self.ep300_indices = ep300_indices or MESC_EP300_INDICES
        self.signal_threshold = signal_threshold

    @staticmethod
    def _slice_channels(tensor_or_array, indices: List[int]) -> np.ndarray:
        """ slice specific channel indices from the last dimension """
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
        """compute total signal mass: sum of all values above threshold """
        flat = array.ravel().astype(float)
        flat = np.where(flat >= self.signal_threshold, flat, 0.0)
        flat = np.clip(flat, 0.0, None)
        return float(flat.sum())

    def _signal_stats(self, array: np.ndarray) -> Dict[str, float]:
        """ compute signal statistics for logging """
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
        """ compute the mESC-specific Sox2 score from raw AlphaGenome outputs """
        if self.signal_threshold > 0:
            print(f"  [scorer] Signal threshold: {self.signal_threshold}")

        # DNase: ES-CJ7 channel 23
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

        # H3K27ac: ES-Bruce4 channel 50
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

        # H3K4me1: ES-Bruce4 channel 51
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

        # EP300: ES-Bruce4 channel 89
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
