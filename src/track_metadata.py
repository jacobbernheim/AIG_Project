from pathlib import Path
from typing import Dict, Optional, List, Mapping, Any

import numpy as np
import pandas as pd


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
    """ loads and queries track-to-cell-type mapping from parquet or CSV """

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
