"""
Generate AlphaGenome tracks for a single input DNA sequence and save them to disk.

Example:
    python generate_tracks.py \
        --sequence ACGTACGT... \
        --output-dir results/tracks/sample_run
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.model_utils import GenomeModel

DEFAULT_TRACKS = ["atac", "dnase", "rna_seq"]
DEFAULT_ORGANISM = "mouse"
DEFAULT_ONTOLOGY_TERMS = ["CL:0002322"]


def track_profile(values: np.ndarray) -> np.ndarray:
    """Reduce a track array to a 1D profile that is easy to plot."""
    array = np.asarray(values)
    if array.size == 0:
        return np.asarray([])

    array = np.squeeze(array)
    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim == 1:
        return array.astype(float, copy=False)

    if array.ndim >= 2:
        position_axis = int(np.argmax(array.shape))
        reduce_axes = tuple(axis for axis in range(array.ndim) if axis != position_axis)
        profile = array.mean(axis=reduce_axes)
        return np.asarray(profile, dtype=float).reshape(-1)

    return np.asarray(array, dtype=float).reshape(-1)


def parse_tracks(track_arg: str | None) -> List[str]:
    """Parse a comma-separated track list from the command line."""
    if not track_arg:
        return DEFAULT_TRACKS
    tracks = [track.strip() for track in track_arg.split(",") if track.strip()]
    return tracks or DEFAULT_TRACKS


def parse_ontology_terms(ontology_arg: str | None) -> List[str] | None:
    """Parse a comma-separated ontology term list from the command line."""
    if not ontology_arg:
        return None
    terms = [term.strip() for term in ontology_arg.split(",") if term.strip()]
    return terms or None


def save_track_outputs(
    track_outputs: dict[str, np.ndarray],
    output_dir: Path,
    sequence: str,
    organism: str,
    tracks: List[str],
    ontology_terms: List[str] | None,
    resolution: int,
    pad_length: int,
    padded_sequence_length: int,
) -> None:
    """Save track outputs and metadata to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_payload = {}
    summary = {}

    for track_name, values in track_outputs.items():
        array = np.asarray(values)
        npz_payload[track_name] = array
        summary[track_name] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }

        np.save(output_dir / f"{track_name}.npy", array)

    np.savez_compressed(output_dir / "tracks.npz", **npz_payload)

    metadata = {
        "sequence_length": len(sequence),
        "padded_sequence_length": padded_sequence_length,
        "padding_added": pad_length,
        "organism": organism,
        "requested_tracks": tracks,
        "ontology_terms": ontology_terms,
        "resolution": resolution,
        "available_tracks": list(track_outputs.keys()),
        "track_summary": summary,
    }

    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )

    (output_dir / "sequence.txt").write_text(sequence + "\n")


def save_track_plots(track_outputs: dict[str, np.ndarray], output_dir: Path) -> None:
    """Save a plot for each track and one combined overview figure."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not track_outputs:
        return

    track_items = list(track_outputs.items())
    n_tracks = len(track_items)
    n_cols = 1 if n_tracks == 1 else 2
    n_rows = math.ceil(n_tracks / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3.5 * n_rows), squeeze=False)
    axes_flat = axes.ravel()

    for axis in axes_flat[n_tracks:]:
        axis.axis("off")

    for axis, (track_name, values) in zip(axes_flat, track_items):
        profile = track_profile(values)
        if profile.size == 0:
            axis.text(0.5, 0.5, "No data", ha="center", va="center")
            axis.set_title(track_name)
            axis.set_axis_off()
            continue

        axis.plot(profile, color="#0f766e", linewidth=1.5)
        axis.set_title(track_name)
        axis.set_xlabel("Position")
        axis.set_ylabel("Signal")
        axis.grid(alpha=0.2)

        track_fig, track_ax = plt.subplots(figsize=(12, 3.5))
        track_ax.plot(profile, color="#0f766e", linewidth=1.5)
        track_ax.set_title(f"{track_name} track")
        track_ax.set_xlabel("Position")
        track_ax.set_ylabel("Signal")
        track_ax.grid(alpha=0.2)
        track_fig.tight_layout()
        track_fig.savefig(plots_dir / f"{track_name}.png", dpi=200, bbox_inches="tight")
        plt.close(track_fig)

    fig.suptitle("AlphaGenome track profiles", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "tracks_overview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate AlphaGenome tracks for a single DNA sequence"
    )
    parser.add_argument(
        "--sequence",
        type=str,
        required=True,
        help="DNA sequence to run through AlphaGenome",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/tracks",
        help="Directory where track outputs will be saved",
    )
    parser.add_argument(
        "--tracks",
        type=str,
        default=",".join(DEFAULT_TRACKS),
        help="Comma-separated list of tracks to generate (default: atac,dnase,rna_seq)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1,
        choices=[1, 128],
        help="Output resolution to request from AlphaGenome",
    )
    args = parser.parse_args()

    sequence = args.sequence.strip().upper()
    tracks = parse_tracks(args.tracks)
    ontology_terms = DEFAULT_ONTOLOGY_TERMS
    output_dir = Path(args.output_dir)
    padded_sequence, pad_length = GenomeModel.prepare_sequence_for_model(sequence)

    print("=" * 60)
    print("AlphaGenome Track Generation")
    print("=" * 60)
    print(f"Sequence length: {len(sequence)} bp")
    if pad_length:
        print(f"Padding added: {pad_length} bp of N")
        print(f"Padded sequence length: {len(padded_sequence)} bp")
    print(f"Organism: {DEFAULT_ORGANISM}")
    print(f"Requested tracks: {tracks}")
    print(f"Ontology terms: {ontology_terms}")
    print(f"Resolution: {args.resolution}")
    print(f"Output directory: {output_dir}")

    model = GenomeModel(organism=DEFAULT_ORGANISM)
    model.load_model()

    track_outputs = model.predict_on_sequence(
        padded_sequence,
        tracks=tracks,
        ontology_terms=ontology_terms,
        resolution=args.resolution,
    )

    if not track_outputs:
        raise RuntimeError(
            "No track outputs were returned by AlphaGenome. "
            "Try a different sequence length or track list."
        )

    save_track_outputs(
        track_outputs=track_outputs,
        output_dir=output_dir,
        sequence=sequence,
        organism=DEFAULT_ORGANISM,
        tracks=tracks,
        ontology_terms=ontology_terms,
        resolution=args.resolution,
        pad_length=pad_length,
        padded_sequence_length=len(padded_sequence),
    )
    save_track_plots(track_outputs, output_dir)

    print("")
    print("Saved outputs:")
    for track_name in track_outputs:
        print(f"  - {output_dir / (track_name + '.npy')}")
        print(f"  - {output_dir / 'plots' / (track_name + '.png')}")
    print(f"  - {output_dir / 'tracks.npz'}")
    print(f"  - {output_dir / 'tracks_overview.png'}")
    print(f"  - {output_dir / 'metadata.json'}")
    print(f"  - {output_dir / 'sequence.txt'}")


if __name__ == "__main__":
    main()
