"""
zero-shot inference for sox2 expression prediction using alphagenome
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import tqdm

from src.model_utils import ZeroShotScorer, ZeroShotScoreWeights
from src.genome_model import GenomeModel

import random
import torch


DEFAULT_ORGANISM = "mouse"
DEFAULT_TRACKS = ["dnase", "chip_histone", "chip_tf"]
DEFAULT_REFERENCE_PL = "PL018"


def _set_seeds(seed: int) -> None:
    """ sets seed """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

_set_seeds(22)

def _load_data(csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ loads sequence and activities data """
    df = pd.read_csv(csv_path)
    df["PL"] = df["PL"].astype(str)
    mendel_col = "MenDel.Name"
    df = df.rename(columns={mendel_col: "mendel_name"})
    df["mendel_name"] = df["mendel_name"].astype(str).str.strip()
    df = df.rename(columns={"Sequence": "sequence"})
    
    activities_df = df.copy()
    
    n_seq = df["sequence"].notna().sum()
    n_total = len(df)
    print(f"  Sequences {n_seq}/{n_total} not NA", flush=True)
    sequences_df = df.dropna(subset=["sequence"]).reset_index(drop=True)
    
    if len(sequences_df) == 0:
        raise ValueError("no sequences")
    
    sequences_df["sequence"] = sequences_df["sequence"].astype(str).str.strip().str.upper()
    
    for idx, row in sequences_df.iterrows():
        seq = row["sequence"]
        if len(seq) < 100:
            print(f"  [warn] {row['PL']} ({row['mendel_name']}): "
                  f"very short sequence ({len(seq)} bp)", flush=True)
    
    print(f"  Final: {len(sequences_df)} sequences ready for scoring", flush=True)
    print(f"  Activities table: {len(activities_df)} records", flush=True)
    
    return sequences_df, activities_df


def _extract_score_components(prefix: str, component_values: dict[str, float]) -> dict[str, float]:
    """ gets the score components """
    return {f"{prefix}_{name}": float(value) for name, value in component_values.items()}


def load_config(config_path: str = "config/default_config.yaml") -> dict:
    """ loads the config """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _build_zero_shot_weights(config: dict) -> ZeroShotScoreWeights:
    """ gets the zero-shot weights """
    zs = config.get("zero_shot", {})
    w = zs.get("weights", {})
    return ZeroShotScoreWeights(
        dnase=float(w.get("dnase", ZeroShotScoreWeights.dnase)),
        h3k27ac=float(w.get("h3k27ac", ZeroShotScoreWeights.h3k27ac)),
        h3k4me1=float(w.get("h3k4me1", ZeroShotScoreWeights.h3k4me1)),
        ep300=float(w.get("ep300", ZeroShotScoreWeights.ep300)),
    )


def run_zero_shot_prediction(
    data_path: str = None,
    config_path: str = "config/default_config.yaml",
    output_dir: str = "results/",
) -> dict:
    """ run zero-shot xox2 expression prediction """

    if data_path is None:
        data_path = "data/processed/merged_payloads.csv"

    config = load_config(config_path)

    print(f"Loading data from {data_path}...", flush=True)
    sequences_df, activities_df = _load_data(data_path)
    print(f"Loaded {len(sequences_df)} sequences, {len(activities_df)} activity records", flush=True)

    zs_config = config.get("zero_shot", {})
    organism = zs_config.get("organism",
                              config.get("model", {}).get("organism", DEFAULT_ORGANISM))
    reference_pl = str(zs_config.get("reference_pl", DEFAULT_REFERENCE_PL))
    requested_tracks = zs_config.get("requested_tracks", DEFAULT_TRACKS)
    prediction_resolution = int(
        config.get("inference", {}).get("prediction_resolution", 128)
    )
    signal_threshold = float(zs_config.get("signal_threshold", 0.0))

    if any(t in requested_tracks for t in ["chip_histone", "chip_tf"]):
        if prediction_resolution != 128:
            print(f"  [info] Forcing resolution to 128bp "
                  f"(chip_histone/chip_tf only at 128bp)", flush=True)
            prediction_resolution = 128

    print("\nInitializing AlphaGenome model...", flush=True)
    genome_model = GenomeModel(
        organism=organism,
        device=config.get("model", {}).get("device"),
    )
    print(f"Model info: {genome_model.get_model_info()}", flush=True)
    genome_model.load_model()

    scorer = ZeroShotScorer(
        weights=_build_zero_shot_weights(config),
        signal_threshold=signal_threshold,
    )

    print(f"\n{'='*60}", flush=True)
    print(f"mESC Zero-Shot Scoring Configuration", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Organism:         {organism}", flush=True)
    print(f"  Resolution:       {prediction_resolution}bp", flush=True)
    print(f"  Tracks:           {requested_tracks}", flush=True)
    print(f"  Reference:        {reference_pl}", flush=True)
    print(f"  Signal threshold: {signal_threshold}", flush=True)
    print(f"  DNase:            ES-CJ7 ch{scorer.dnase_indices}", flush=True)
    print(f"  H3K27ac:          ES-Bruce4 ch{scorer.h3k27ac_indices}", flush=True)
    print(f"  H3K4me1:          ES-Bruce4 ch{scorer.h3k4me1_indices}", flush=True)
    print(f"  EP300:            ES-Bruce4 ch{scorer.ep300_indices}", flush=True)
    print(f"  Weights:          dnase={scorer.weights.dnase}, "
          f"h3k27ac={scorer.weights.h3k27ac}, "
          f"h3k4me1={scorer.weights.h3k4me1}, "
          f"ep300={scorer.weights.ep300}", flush=True)
    print(f"  Scoring method:   signal mass (sum of positive signal)", flush=True)
    print(f"{'='*60}\n", flush=True)

    print(f"Scoring {len(sequences_df)} sequences...", flush=True)
    score_rows = []

    for index, row in tqdm.tqdm(sequences_df.iterrows(), total=len(sequences_df)):
        sequence = str(row["sequence"]).strip().upper()
        pl_value = str(row["PL"])
        mendel_name = str(row.get("mendel_name", ""))

        if index % max(1, len(sequences_df) // 10) == 0:
            print(f"\n  [{index}/{len(sequences_df)}] {pl_value} "
                  f"({mendel_name}, {len(sequence)} bp)", flush=True)

        raw_outputs = genome_model.predict_on_sequence_raw(
            sequence,
            tracks=requested_tracks,
            resolution=prediction_resolution,
        )

        score = scorer.score(raw_outputs)

        score_rows.append({
            "PL": pl_value,
            "mendel_name": mendel_name,
            "sequence_length": len(sequence),
            "raw_score": score.raw_score,
            **_extract_score_components("component", score.component_values),
        })

    score_df = pd.DataFrame(score_rows)

    ref_match = score_df.loc[score_df["PL"] == reference_pl]
    if ref_match.empty:
        print(f"\n  [warn] Reference PL '{reference_pl}' not found in scored sequences.",
              flush=True)
        print(f"  Available PLs: {score_df['PL'].tolist()}", flush=True)
        print(f"  Skipping normalization — raw scores only.", flush=True)
        score_df["normalized_score"] = np.nan
        score_df["is_reference"] = False
        ref_raw = None
    else:
        ref_raw = float(ref_match.iloc[0]["raw_score"])
        if ref_raw == 0.0:
            print(f"  [warn] Reference {reference_pl} has zero raw score. "
                  f"Skipping normalization.", flush=True)
            score_df["normalized_score"] = np.nan
        else:
            score_df["normalized_score"] = score_df["raw_score"] / ref_raw
        score_df["is_reference"] = score_df["PL"] == reference_pl

    if activities_df is not None:
        score_df = score_df.merge(activities_df, on="PL", how="left",
                                   suffixes=("", "_activity"))

    print(f"\n{'='*60}", flush=True)
    print("Scoring complete", flush=True)
    if ref_raw is not None and ref_raw != 0.0:
        print(f"Reference {reference_pl} raw score: {ref_raw:.6f}", flush=True)
        print(f"Normalized reference score: 1.000000", flush=True)
    print(f"Score range (raw): {score_df['raw_score'].min():.4f} — "
          f"{score_df['raw_score'].max():.4f}", flush=True)
    if score_df["normalized_score"].notna().any():
        print(f"Score range (norm): {score_df['normalized_score'].min():.4f} — "
              f"{score_df['normalized_score'].max():.4f}", flush=True)
    print(f"{'='*60}", flush=True)

    results = {
        "predictions": score_df["normalized_score"].to_numpy(),
        "raw_scores": score_df["raw_score"].to_numpy(),
        "score_table": score_df,
        "reference_pl": reference_pl,
        "reference_raw_score": ref_raw,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.get("output", {}).get("save_predictions", True):
        np.save(output_dir / "zero_shot_normalized_scores.npy",
                score_df["normalized_score"].to_numpy())
        np.save(output_dir / "zero_shot_raw_scores.npy",
                score_df["raw_score"].to_numpy())
        score_df.to_csv(output_dir / "zero_shot_scores.csv", index=False)

        summary = {
            "reference_pl": reference_pl,
            "reference_raw_score": ref_raw,
            "organism": organism,
            "resolution": prediction_resolution,
            "requested_tracks": requested_tracks,
            "signal_threshold": signal_threshold,
            "n_sequences_scored": len(score_df),
            "scoring_method": "signal_mass (sum of positive signal, scales with length)",
            "mesc_channels": {
                "dnase": {"cell_line": "ES-CJ7", "curie": "EFO:0005916",
                          "indices": scorer.dnase_indices},
                "h3k27ac": {"cell_line": "ES-Bruce4", "curie": "EFO:0005483",
                            "indices": scorer.h3k27ac_indices},
                "h3k4me1": {"cell_line": "ES-Bruce4", "curie": "EFO:0005483",
                            "indices": scorer.h3k4me1_indices},
                "ep300": {"cell_line": "ES-Bruce4", "curie": "EFO:0005483",
                          "indices": scorer.ep300_indices},
            },
            "weights": {
                "dnase": scorer.weights.dnase,
                "h3k27ac": scorer.weights.h3k27ac,
                "h3k4me1": scorer.weights.h3k4me1,
                "ep300": scorer.weights.ep300,
            },
        }
        (output_dir / "zero_shot_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n"
        )

        print(f"\nSaved to {output_dir}:", flush=True)
        print(f"  zero_shot_scores.csv", flush=True)
        print(f"  zero_shot_normalized_scores.npy", flush=True)
        print(f"  zero_shot_raw_scores.npy", flush=True)
        print(f"  zero_shot_summary.json", flush=True)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="zero-shot sox2 expression prediction using alphagenome"
    )
    parser.add_argument("--data-file", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/")
    parser.add_argument("--config", type=str, default="config/default_config.yaml")

    args = parser.parse_args()

    results = run_zero_shot_prediction(
        data_path=args.data_file,
        config_path=args.config,
        output_dir=args.output_dir,
    )
