"""
Generate AlphaGenome tracks for one sequence or a CSV of sequences.

Uses the raw model() forward pass (as in the demo notebook) to get
resolution-keyed dicts. Ontology-based cell-type filtering is NOT
supported — the alphagenome_pytorch package does not ship ontology
metadata. All channels are returned; use --plot-track-indices to
select specific channels for visualization.

Example:
    python generate_tracks.py \
        --input-csv examples.csv \
        --output-dir results/tracks/batch_run

    python generate_tracks.py \
        --sequence ACGTACGT... \
        --output-dir results/tracks/sample_run \
        --resolution 1 \
        --plot-track-indices 'atac:0,1,2;dnase:0,1,2'

CSV expects columns:
    - sample_name   : unique identifier used for output directory naming
    - sequence      : DNA sequence string
    - organism      : "human" or "mouse"
    - cell_type     : ontology term (logged but NOT used for filtering)

Example CSV:
    sample_name,sequence,organism,cell_type
    mesc_sample1,ACGTACGT...,mouse,CL:0002322
    hepg2_sample1,ACGTACGT...,human,EFO:0001187
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List

import numpy as np
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.model_utils import GenomeModel

DEFAULT_TRACKS = ["atac", "dnase"]
DEFAULT_ORGANISM = "mouse"
DEFAULT_ONTOLOGY_TERMS = ["CL:0002322"]

DEFAULT_PLOT_WINDOW_SIZES = {
    1: 131_072,
    128: 1024,
}


def parse_tracks(track_arg: str | None) -> List[str]:
    if not track_arg:
        return DEFAULT_TRACKS
    tracks = [t.strip() for t in track_arg.split(",") if t.strip()]
    return tracks or DEFAULT_TRACKS


def normalize_column_name(column_name: str) -> str:
    return "".join(ch for ch in column_name.lower() if ch.isalnum())


def pick_column(dataframe: pd.DataFrame, candidates: List[str]) -> str:
    lookup = {normalize_column_name(c): c for c in dataframe.columns}
    for candidate in candidates:
        norm = normalize_column_name(candidate)
        if norm in lookup:
            return lookup[norm]
    raise KeyError(f"Could not find any of these columns: {candidates}")


def parse_optional_column_name(arg: str | None) -> str | None:
    if arg is None:
        return None
    return arg.strip() or None


def parse_ontology_terms(arg: str | None) -> List[str] | None:
    if not arg:
        return None
    terms = [t.strip() for t in arg.split(",") if t.strip()]
    return terms or None


def parse_track_indices(arg: str | None) -> dict[str, List[int]] | None:
    if not arg:
        return None
    parsed = {}
    for block in arg.split(";"):
        block = block.strip()
        if not block:
            continue
        if ":" not in block:
            raise ValueError("Track indices must look like 'atac:0,1,2;dnase:0,1,2'")
        name, idx_text = block.split(":", 1)
        indices = [int(v.strip()) for v in idx_text.split(",") if v.strip()]
        if indices:
            parsed[name.strip()] = indices
    return parsed or None


def load_input_table(
    input_csv: str,
    sequence_column: str | None = None,
    organism_column: str | None = None,
    cell_type_column: str | None = None,
    name_column: str | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    seq_col = sequence_column or pick_column(df, ["sequence", "seq", "dna_sequence"])
    org_col = organism_column or pick_column(df, ["organism", "species"])

    # cell_type is optional
    try:
        ct_col = cell_type_column or pick_column(
            df, ["ontology_term", "ontology_terms", "cell_type_ontology_term",
                 "cell_type", "celltype"]
        )
    except KeyError:
        ct_col = None

    rename_map = {seq_col: "sequence", org_col: "organism"}
    if ct_col:
        rename_map[ct_col] = "cell_type"
    renamed = df.rename(columns=rename_map).copy()

    if "cell_type" not in renamed.columns:
        renamed["cell_type"] = "unknown"

    if name_column:
        renamed = renamed.rename(columns={name_column: "sample_name"})
    else:
        for candidate in ["sample_name", "name", "id", "example_id", "row_id"]:
            try:
                found = pick_column(renamed, [candidate])
                if found != "sample_name":
                    renamed = renamed.rename(columns={found: "sample_name"})
                break
            except KeyError:
                continue

    renamed["sequence"] = renamed["sequence"].astype(str)
    renamed["organism"] = renamed["organism"].astype(str)
    renamed["cell_type"] = renamed["cell_type"].astype(str)
    if "sample_name" not in renamed.columns:
        renamed["sample_name"] = [f"row_{i+1}" for i in range(len(renamed))]
    renamed["sample_name"] = renamed["sample_name"].astype(str)
    return renamed


def save_track_outputs(
    track_outputs: dict[str, np.ndarray],
    output_dir: Path,
    sequence: str,
    sample_name: str,
    cell_type: str,
    organism: str,
    tracks: List[str],
    ontology_terms: List[str] | None,
    resolution: int,
    pad_length: int,
    padded_sequence_length: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_payload = {}
    summary = {}

    for track_name, values in track_outputs.items():
        arr = np.asarray(values)
        npz_payload[track_name] = arr
        summary[track_name] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
        np.save(output_dir / f"{track_name}.npy", arr)

    np.savez_compressed(output_dir / "tracks.npz", **npz_payload)

    metadata = {
        "sample_name": sample_name,
        "sequence_length": len(sequence),
        "padded_sequence_length": padded_sequence_length,
        "padding_added": pad_length,
        "organism": organism,
        "cell_type": cell_type,
        "requested_tracks": tracks,
        "ontology_terms": ontology_terms,
        "resolution": resolution,
        "available_tracks": list(track_outputs.keys()),
        "track_summary": summary,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    (output_dir / "sequence.txt").write_text(sequence + "\n")


def generate_cell_type_tracks(
    model: GenomeModel,
    sequence: str,
    cell_type: str,
    tracks: List[str],
    resolution: int,
) -> dict[str, np.ndarray]:
    """Generate tracks for a sequence. Returns all requested tracks."""
    ontology_terms = parse_ontology_terms(cell_type) or [cell_type]
    padded_sequence, _ = GenomeModel.prepare_sequence_for_model(sequence)

    raw_predictions = model.predict_on_sequence(
        padded_sequence,
        tracks=tracks,
        ontology_terms=ontology_terms,
        resolution=resolution,
    )

    results = {}
    for track_name in tracks:
        if track_name in raw_predictions:
            results[track_name] = np.asarray(raw_predictions[track_name])
        else:
            print(f"  [warn] Track '{track_name}' not returned by model")
    return results


def get_model_for_organism(
    model_cache: dict[str, GenomeModel],
    organism: str,
) -> GenomeModel:
    norm = organism.strip().lower()
    if norm not in model_cache:
        model = GenomeModel(organism=norm)
        model.load_model()
        model_cache[norm] = model
    return model_cache[norm]


def format_bp(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f} Mb"
    elif value >= 1_000:
        return f"{value / 1_000:.1f} kb"
    return f"{value} bp"


def save_track_plots(
    track_outputs: dict[str, np.ndarray],
    output_dir: Path,
    sequence_length: int,
    resolution: int,
    window_size: int | None = None,
    start: int = 0,
    track_indices: dict[str, List[int]] | None = None,
) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not track_outputs:
        return

    if window_size is None:
        window_size = DEFAULT_PLOT_WINDOW_SIZES.get(resolution, 1024)

    track_items = list(track_outputs.items())
    n_tracks = len(track_items)
    n_cols = 1 if n_tracks == 1 else 2
    n_rows = math.ceil(n_tracks / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3.5 * n_rows), squeeze=False)
    axes_flat = axes.ravel()

    for ax in axes_flat[n_tracks:]:
        ax.axis("off")

    for ax, (track_name, values) in zip(axes_flat, track_items):
        arr = np.asarray(values).squeeze()
        if arr.size == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(track_name)
            continue

        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        elif arr.ndim > 2:
            pos_ax = int(np.argmax(arr.shape))
            reduce = tuple(i for i in range(arr.ndim) if i != pos_ax)
            arr = arr.mean(axis=reduce)
            if arr.ndim == 1:
                arr = arr[:, np.newaxis]

        if arr.ndim != 2:
            ax.text(0.5, 0.5, "Unsupported shape", ha="center", va="center")
            ax.set_title(track_name)
            continue

        seq_bins = arr.shape[0]
        ws = max(0, int(start))
        effective = min(int(window_size), seq_bins - ws)
        we = ws + max(1, effective)
        if ws >= seq_bins:
            ws = max(0, (seq_bins - min(int(window_size), seq_bins)) // 2)
            we = min(seq_bins, ws + int(window_size))

        window = arr[ws:we, :]
        if window.size == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(track_name)
            continue

        x = np.arange(ws, we) * resolution
        genomic_span = (we - ws) * resolution

        sel = track_indices.get(track_name) if track_indices else None
        if sel:
            valid = [i for i in sel if 0 <= i < window.shape[1]]
            for i in valid:
                ax.plot(x, window[:, i], alpha=0.75,
                        linewidth=0.5 if resolution == 1 else 1.0,
                        label=f"Track {i}")
            if len(valid) > 1:
                ax.legend(loc="upper right", fontsize=8)
        else:
            mean = window.mean(axis=1)
            std = window.std(axis=1)
            ax.plot(x, mean, color="#2563eb", alpha=0.9,
                    linewidth=0.5 if resolution == 1 else 1.2)
            ax.fill_between(x, mean - std, mean + std, color="#2563eb", alpha=0.25)

        ax.set_title(
            f"{track_name.upper()} at {resolution}bp "
            f"({format_bp(genomic_span)}, {we - ws:,} bins, "
            f"{arr.shape[1]} ch)"
        )
        ax.set_xlabel("Position (bp)")
        ax.set_ylabel(track_name.upper())
        ax.grid(True, alpha=0.3)

        # Individual track figure
        fw = 18 if resolution == 1 else 12
        tfig, tax = plt.subplots(figsize=(fw, 3.5))
        if sel:
            valid = [i for i in sel if 0 <= i < window.shape[1]]
            for i in valid:
                tax.plot(x, window[:, i], alpha=0.75,
                         linewidth=0.5 if resolution == 1 else 1.0,
                         label=f"Track {i}")
            if len(valid) > 1:
                tax.legend(loc="upper right", fontsize=8)
        else:
            tax.plot(x, mean, color="#2563eb", alpha=0.9,
                     linewidth=0.5 if resolution == 1 else 1.2)
            tax.fill_between(x, mean - std, mean + std, color="#2563eb", alpha=0.25)
        tax.set_title(
            f"{track_name.upper()} at {resolution}bp "
            f"({format_bp(genomic_span)}, {we - ws:,} bins, "
            f"{arr.shape[1]} ch)"
        )
        tax.set_xlabel("Position (bp)")
        tax.set_ylabel(track_name.upper())
        tax.grid(True, alpha=0.3)
        tfig.tight_layout()
        tfig.savefig(plots_dir / f"{track_name}.png", dpi=200, bbox_inches="tight")
        plt.close(tfig)

    fig.suptitle(f"AlphaGenome track profiles ({resolution}bp resolution)", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "tracks_overview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate AlphaGenome tracks for DNA sequences"
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--sequence", type=str, help="Single DNA sequence")
    input_group.add_argument("--input-csv", type=str, help="CSV with sequences")

    parser.add_argument("--output-dir", type=str, default="results/tracks")
    parser.add_argument("--tracks", type=str, default=",".join(DEFAULT_TRACKS))
    parser.add_argument("--resolution", type=int, default=1, choices=[1, 128])
    parser.add_argument("--plot-start", type=int, default=0)
    parser.add_argument("--plot-window-size", type=int, default=None)
    parser.add_argument("--plot-track-indices", type=str, default=None)
    parser.add_argument("--sequence-column", type=str, default=None)
    parser.add_argument("--organism-column", type=str, default=None)
    parser.add_argument("--cell-type-column", type=str, default=None)
    parser.add_argument("--name-column", type=str, default=None)
    args = parser.parse_args()

    tracks = parse_tracks(args.tracks)
    plot_track_indices = parse_track_indices(args.plot_track_indices)
    output_dir = Path(args.output_dir)

    plot_window_size = args.plot_window_size
    if plot_window_size is None:
        plot_window_size = DEFAULT_PLOT_WINDOW_SIZES.get(args.resolution, 1024)

    print("=" * 60)
    print("AlphaGenome Track Generation")
    print("=" * 60)
    print(f"Tracks        : {tracks}")
    print(f"Resolution    : {args.resolution}bp")
    print(f"Plot window   : {plot_window_size:,} bins = {format_bp(plot_window_size * args.resolution)}")
    print(f"Output dir    : {output_dir}")
    if plot_track_indices:
        print(f"Track indices : {plot_track_indices}")
    else:
        print(f"Track indices : all (mean ± std). Use --plot-track-indices for individual channels.")
    print()

    if args.input_csv:
        input_table = load_input_table(
            args.input_csv,
            sequence_column=parse_optional_column_name(args.sequence_column),
            organism_column=parse_optional_column_name(args.organism_column),
            cell_type_column=parse_optional_column_name(args.cell_type_column),
            name_column=parse_optional_column_name(args.name_column),
        )
        print(f"Loaded {len(input_table)} rows from {args.input_csv}\n")

        manifest_rows = []
        model_cache: dict[str, GenomeModel] = {}

        for index, row in input_table.iterrows():
            sequence = str(row["sequence"]).strip().upper()
            organism = str(row["organism"]).strip().lower()
            cell_type = str(row["cell_type"]).strip()
            sample_name = str(row["sample_name"]).strip() or f"row_{index+1}"
            sample_dir = output_dir / sample_name

            padded, pad_len = GenomeModel.prepare_sequence_for_model(sequence)

            print(f"{'='*60}")
            print(f"[{index+1}/{len(input_table)}] {sample_name}")
            print(f"  sequence : {len(sequence)} bp → padded {len(padded)} bp")
            print(f"  organism : {organism}")
            print(f"  cell_type: {cell_type} (logged only — not used for filtering)")

            model = get_model_for_organism(model_cache, organism)

            track_outputs = generate_cell_type_tracks(
                model=model,
                sequence=sequence,
                cell_type=cell_type,
                tracks=tracks,
                resolution=args.resolution,
            )

            save_track_outputs(
                track_outputs=track_outputs,
                output_dir=sample_dir,
                sequence=sequence,
                sample_name=sample_name,
                cell_type=cell_type,
                organism=organism,
                tracks=tracks,
                ontology_terms=parse_ontology_terms(cell_type) or [cell_type],
                resolution=args.resolution,
                pad_length=pad_len,
                padded_sequence_length=len(padded),
            )
            save_track_plots(
                track_outputs,
                sample_dir,
                sequence_length=len(sequence),
                resolution=args.resolution,
                window_size=plot_window_size,
                start=args.plot_start,
                track_indices=plot_track_indices,
            )

            manifest_rows.append({
                "sample_name": sample_name,
                "organism": organism,
                "cell_type": cell_type,
                "output_dir": str(sample_dir),
                "sequence_length": len(sequence),
                "padded_sequence_length": len(padded),
                "padding_added": pad_len,
            })
            print(f"  saved to {sample_dir}\n")

        manifest = pd.DataFrame(manifest_rows)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(output_dir / "manifest.csv", index=False)
        print(f"Saved batch manifest to {output_dir / 'manifest.csv'}")
        return

    # Single sequence mode
    model = GenomeModel(organism=DEFAULT_ORGANISM)
    model.load_model()

    sequence = args.sequence.strip().upper()
    padded, pad_len = GenomeModel.prepare_sequence_for_model(sequence)

    print(f"Sequence: {len(sequence)} bp → padded {len(padded)} bp")

    track_outputs = generate_cell_type_tracks(
        model=model,
        sequence=sequence,
        cell_type=DEFAULT_ONTOLOGY_TERMS[0],
        tracks=tracks,
        resolution=args.resolution,
    )

    save_track_outputs(
        track_outputs=track_outputs,
        output_dir=output_dir,
        sequence=sequence,
        sample_name="single_sequence",
        cell_type=DEFAULT_ONTOLOGY_TERMS[0],
        organism=DEFAULT_ORGANISM,
        tracks=tracks,
        ontology_terms=DEFAULT_ONTOLOGY_TERMS,
        resolution=args.resolution,
        pad_length=pad_len,
        padded_sequence_length=len(padded),
    )
    save_track_plots(
        track_outputs,
        output_dir,
        sequence_length=len(sequence),
        resolution=args.resolution,
        window_size=plot_window_size,
        start=args.plot_start,
        track_indices=plot_track_indices,
    )

    print("\nSaved outputs:")
    for tn in track_outputs:
        print(f"  {output_dir / (tn + '.npy')}")
        print(f"  {output_dir / 'plots' / (tn + '.png')}")
    print(f"  {output_dir / 'tracks.npz'}")
    print(f"  {output_dir / 'tracks_overview.png'}")
    print(f"  {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()