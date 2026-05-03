# Changelog

All notable changes to this project will be documented in this file.

## [0.8.0] - 2026-XX-XX (Zero-Shot Scoring with mESC-Specific Channels)

### Added
- FASTA sequence loading in `zero_shot_inference.py` — sequences are loaded from `data/raw/Payload Sequences - Ribeiro-dos-Santos_Supplemental_Data_S1.fasta` and joined to the input CSV by MenDel.Name
- `predict_on_sequence_raw()` method in `GenomeModel` for unfiltered tensor output used by the scorer
- `_signal_mass()` method in `ZeroShotScorer` — sum of positive signal that scales with sequence length
- `_signal_stats()` method for detailed per-component logging (bins active, fraction active, mean active signal)
- `signal_threshold` parameter for excluding low-level background noise from N-padded regions
- Per-component logging showing exact channel indices, cell lines, and signal statistics

### Changed
- **Scoring formula completely rewritten** to use explicit mESC channel indices from track metadata:
  - DNase: ES-CJ7 channel 23 (weight 1.0)
  - H3K27ac: ES-Bruce4 channel 50 (weight 1.0)
  - H3K4me1: ES-Bruce4 channel 51 (weight 0.8)
  - EP300: ES-Bruce4 channel 89 (weight 0.6)
- **ATAC removed** from scoring — no mESC ATAC-seq data exists in the mouse AlphaGenome model
- **DNase weight reduced** from 2.0 to 1.0 (ES-CJ7, not BL6 ES-Bruce4)
- **TF scoring simplified** to EP300 only (dropped CTCF, POLR2A); TF entropy removed (meaningless with 1 channel)
- Scorer now slices specific channel indices from full output tensors instead of searching nested dicts by name
- `ZeroShotScoreWeights` fields updated: removed `atac`, `tf_abundance`, `tf_entropy`; added `ep300`
- Graceful handling when reference PL is missing from subset — warns and saves raw scores without normalization
- `mendel_name` and `sequence_length` included in output CSV

### Fixed
- Scorer no longer averages across all ~1100+ histone channels from every cell type
- Scorer no longer attempts `_resolve_named_component()` name-search on flat tensors (which always returned None)
- `preserve_raw=True` no longer returns `NamedTrackTensor` objects that crash on format strings

## [0.7.0] - 2026-XX-XX (Track Metadata System)

### Added
- `TrackMetadata` class in `model_utils.py` for cell-type → channel-index mapping
- Support for both parquet (`track_metadata_human.parquet`, `track_metadata_mouse.parquet`) and CSV metadata files
- Organism-aware metadata file selection — mouse sequences automatically load mouse metadata
- `get_channel_indices()` — look up channels by ontology CURIE and track type
- `get_channel_indices_by_mark()` — look up specific histone marks (e.g., H3K27ac) by channel index
- `get_channel_indices_by_tf()` — look up specific transcription factors
- `get_all_channels_for_curie()` — get all tracks available for a cell type
- `search_biosample()` — search metadata by biosample name (for discovery)
- Column normalization to handle different naming conventions between human CSV (`index`, `OutputType.ATAC`) and mouse parquet (`track_index`, `atac`)

### Changed
- `GenomeModel.__init__()` now accepts `track_metadata_path` parameter and passes `organism` to `TrackMetadata`
- `_filter_by_ontology()` uses `TrackMetadata` for channel lookup instead of the non-functional `.select(ontology_curie=...)` API

## [0.6.0] - 2026-XX-XX (Raw Forward Pass Migration)

### Added
- `_strip_padding_channels()` method — removes padding channels using known real track counts (e.g., ATAC 256 → 167)
- `predict_on_sequence_raw()` method — returns unfiltered tensors for scorer use

### Changed
- **Switched from `.predict(named_outputs=True)` to raw `model(dna_one_hot, organism_index)` forward pass**
  - The `.predict()` API returns `NamedTrackTensor` objects that only contain 128bp resolution data
  - Ontology-based `.select(ontology_curie=...)` filtering requires metadata annotations not shipped with `alphagenome_pytorch`
  - The raw forward pass returns plain `{1: Tensor, 128: Tensor}` dicts with both resolutions, matching the demo notebook approach
- Removed `_build_metadata_catalog()` and `TrackMetadataCatalog` imports (no longer needed)
- Resolution selection now uses direct dict key lookup instead of the buggy `.get(resolution, track_container)` fallback

### Fixed
- **Resolution bug**: The old code used `track_container.get(resolution, track_container)` which fell back to returning the entire `{1: tensor, 128: tensor}` dict when the key wasn't found, causing `_to_numeric_array()` to concatenate both resolutions into garbled data
- 1bp resolution data now correctly returned (previously always got 128bp data regardless of requested resolution)

## [0.5.0] - 2026-XX-XX (Track Generation Improvements)

### Added
- Resolution-aware default plot window sizes: 131,072 bins for 1bp, 1,024 bins for 128bp (both ≈131 kb genomic span)
- `format_bp()` helper for human-readable position formatting in plot titles
- Plot titles now show genomic span, bin count, and channel count
- Thinner line widths and wider figures for 1bp resolution plots
- `generate_cell_type_tracks()` now returns all requested tracks (not just hardcoded atac/dnase)
- Output summary with min/max/mean per track for quick validation

### Changed
- Default `--plot-window-size` depends on resolution instead of fixed 1024
- Individual track figures are 18 inches wide at 1bp resolution (vs 12 at 128bp) to resolve dense signal
- Removed hardcoded `{"atac": ..., "dnase": ...}` return in `generate_cell_type_tracks()`

### Fixed
- Plots at 1bp resolution no longer look identical