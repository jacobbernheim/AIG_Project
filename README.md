# Sox2 Expression Prediction from Regulatory Sequences

Zero-shot and fine-tuned AlphaGenome-based prediction of Sox2 expression levels from DNA sequences containing regulatory elements.

**Final project for AI in Genomics 2026**

## Project Structure

```
data/
├── raw/              # Raw input data (DNA sequences + expression labels)
│   ├── Payload Sequences - Ribeiro-dos-Santos_Supplemental_Data_S1.fasta
│   ├── example_subset_sequences.csv
│   └── generate_tracks_example.csv
└── processed/        # Processed/prepared data

src/
├── __init__.py
├── model_utils.py                        # AlphaGenome wrapper, TrackMetadata, ZeroShotScorer
├── track_metadata_human.parquet          # Human track-to-biosample channel mapping
├── track_metadata_mouse.parquet          # Mouse track-to-biosample channel mapping
├── alphagenome_track_celltypes_human.csv # Human track metadata (CSV fallback)
├── data_loader.py                        # Data loading and preprocessing
└── evaluation.py                         # Evaluation metrics

config/
└── default_config.yaml  # Configuration for experiments

notebooks/           # Jupyter notebooks for exploration
results/             # Predictions, metrics, and visualizations

preprocess_data.py      # Data preprocessing and formatting
generate_tracks.py      # Generate and save AlphaGenome track outputs
zero_shot_inference.py  # Zero-shot prediction script
example_inference.py    # Example script to verify setup
requirements.txt        # Python dependencies (pip)
environment.yml        # Conda environment specification (CPU)
environment-gpu.yml    # Conda environment specification (GPU/HPC)
```

## Expression Scale

- **0**: No expression
- **1**: Wild-type (WT) expression (PL018, Sox2 distal LCR)
- **>1**: Overexpression

## Getting Started

### Quick Setup

```bash
conda env create -f environment.yml
conda activate sox2-alphagenome
```

### Verify Installation

```bash
python example_inference.py
```

This downloads the AlphaGenome model (~680 MB, cached after first run) and runs a test inference.

## Track Metadata

AlphaGenome predicts all cell types simultaneously as separate channels in each track output tensor [1]. For example, at 128bp resolution ATAC has 256 channels (167 real + 89 padding), where each channel corresponds to a specific biosample/cell type [1].

Cell-type-specific channel filtering uses track metadata parquet files that map ontology CURIEs (e.g., `EFO:0001187` for HepG2, `EFO:0005483` for ES-Bruce4) to channel indices. These metadata files are stored in `src/`:

- `track_metadata_human.parquet` — Human track-to-biosample mapping
- `track_metadata_mouse.parquet` — Mouse track-to-biosample mapping
- `alphagenome_track_celltypes_human.csv` — CSV fallback for human

The `TrackMetadata` class in `model_utils.py` automatically selects the correct file based on the organism and handles both parquet and CSV formats with different column naming conventions.

### Available mESC Tracks (Mouse)

Not all track types are available for every cell type. For mouse embryonic stem cells, the following tracks are available:

| Cell Line | Ontology CURIE | Available Tracks |
|---|---|---|
| **ES-Bruce4** (BL6) | `EFO:0005483` | chip_histone (H3K27ac, H3K4me1, H3K4me3, H3K9ac), chip_tf (CTCF, EP300, POLR2A), rna_seq (+/-), splice_site_usage |
| **ES-E14** | `EFO:0007075` | dnase, chip_tf (5 TFs), rna_seq (+/-), splice_site_usage |
| **ES-CJ7** | `EFO:0005916` | dnase |
| **E14TG2a.4** | `EFO:0007751` | chip_tf (CTCF, NANOG), chip_histone (5 marks) |

**Notable limitations:**
- No mESC ATAC-seq tracks exist in the mouse AlphaGenome model
- Histone marks (H3K27ac, H3K4me1) are only available for ES-Bruce4
- DNase is only available for ES-E14 and ES-CJ7 (not ES-Bruce4)

## Workflow

### 1. Generate AlphaGenome Tracks

Uses the raw `model(dna_one_hot, organism_index)` forward pass exactly as shown in the demo notebook [1], which returns resolution-keyed dicts like `{1: Tensor, 128: Tensor}` for each track. Cell-type filtering uses the track metadata parquet files to map ontology CURIEs to specific channel indices.

**Human HepG2 example at 1bp resolution:**

```bash
python generate_tracks.py \
    --input-csv data/raw/generate_tracks_example.csv \
    --output-dir results/tracks/hepg2_1bp \
    --tracks atac,dnase \
    --resolution 1
```

The CSV should contain columns: `sample_name`, `sequence`, `organism`, `cell_type` (ontology term).

**Example CSV:**
```csv
sample_name,sequence,organism,cell_type
hepg2_sample1,ACGTACGT...,human,EFO:0001187
mesc_sample1,ACGTACGT...,mouse,EFO:0005483
```

**Plot specific channels (matching the demo notebook style [1]):**
```bash
python generate_tracks.py \
    --input-csv data/raw/generate_tracks_example.csv \
    --output-dir results/tracks/hepg2_1bp \
    --tracks atac,dnase \
    --resolution 1 \
    --plot-track-indices 'atac:0,1,2;dnase:0,1,2'
```

Plot window sizes default to resolution-appropriate values: 131,072 bins for 1bp (~131 kb genomic span) and 1,024 bins for 128bp (~131 kb genomic span), ensuring comparable visualization regardless of resolution.

**Output files:**
- `tracks.npz` — all tracks in a single compressed archive
- `{track_name}.npy` — individual track arrays
- `plots/{track_name}.png` — individual track plots
- `tracks_overview.png` — combined overview plot
- `metadata.json` — scoring configuration and track shapes
- `sequence.txt` — input sequence

### 2. Data Preprocessing

```bash
python preprocess_data.py
```

Reads raw data from `data/raw/` and outputs formatted data to `data/processed/`.

### 3. Zero-Shot Sox2 Expression Prediction

Predicts Sox2 expression levels using mESC-specific AlphaGenome track outputs with a deterministic, WT-normalized scoring formula.

```bash
python zero_shot_inference.py \
    --sequences-file data/raw/example_subset_sequences.csv \
    --output-dir results/zero_shot/
```

The sequences CSV requires `PL` and `MenDel.Name` columns. Sequences are loaded from the FASTA file `data/raw/Payload Sequences - Ribeiro-dos-Santos_Supplemental_Data_S1.fasta` and joined to the CSV by matching the `MenDel.Name` column to FASTA headers.

#### Scoring Formula

The zero-shot score uses **signal mass** (sum of all positive signal across all bins at 128bp resolution), which naturally scales with sequence length — a 30kb insert with three enhancers scores approximately 3× higher than a 10kb insert with a single enhancer.

**Components and weights:**

| Component | Weight | Source Cell Line | Channel Index | Track |
|---|---|---|---|---|
| DNase | 1.0 | ES-CJ7 (`EFO:0005916`) | 23 | `dnase` |
| H3K27ac | 1.0 | ES-Bruce4 (`EFO:0005483`) | 50 | `chip_histone` |
| H3K4me1 | 0.8 | ES-Bruce4 (`EFO:0005483`) | 51 | `chip_histone` |
| EP300 | 0.6 | ES-Bruce4 (`EFO:0005483`) | 89 | `chip_tf` |

**Design decisions:**
- **ATAC dropped**: No mESC ATAC-seq data exists in the mouse AlphaGenome model
- **DNase weight reduced** from 2.0 to 1.0 to match histone mark weights, since DNase comes from ES-CJ7 (not the preferred BL6 ES-Bruce4 line)
- **EP300 only** for TF binding (CTCF and POLR2A dropped as less directly relevant to enhancer activity)
- **TF entropy removed**: Meaningless with a single TF channel
- **ES-Bruce4 preferred** for all tracks where available, as it corresponds to BL6 mice matching the experimental setting

The raw score is normalized so that PL018 (the WT Sox2 distal LCR sequence) equals 1.0. Weights are configurable in `config/default_config.yaml` under `zero_shot.weights`.

#### Output Files

- `zero_shot_scores.csv` — full table with PL, MenDel name, sequence length, raw score, normalized score, and per-component breakdowns
- `zero_shot_normalized_scores.npy` — numpy array of normalized scores
- `zero_shot_raw_scores.npy` — numpy array of raw scores
- `zero_shot_summary.json` — complete metadata including channel indices, cell lines, weights, and scoring method

## Configuration

`config/default_config.yaml`:

```yaml
model:
  organism: mouse

inference:
  prediction_resolution: 128

zero_shot:
  organism: mouse
  reference_pl: "PL018"
  requested_tracks:
    - dnase
    - chip_histone
    - chip_tf
  signal_threshold: 0.0
  weights:
    dnase: 1.0
    h3k27ac: 1.0
    h3k4me1: 0.8
    ep300: 0.6

output:
  save_predictions: true
```

## Technical Details

### Model Architecture

This project uses [AlphaGenome PyTorch](https://github.com/genomicsxai/alphagenome-pytorch) (`alphagenome-pytorch==0.3.1`), a faithful port of Google DeepMind's AlphaGenome [1]. The model predicts hundreds of genomic tracks at both 1bp and 128bp resolution from DNA sequences.

The forward pass uses the raw `model(dna_one_hot, organism_index)` call which returns plain dicts `{1: Tensor, 128: Tensor}` for each track [1]. This approach was chosen over the `.predict(named_outputs=True)` API because:
- The named outputs API (`NamedTrackTensor`) only returns 128bp resolution data
- Ontology-based cell-type filtering via `.select(ontology_curie=...)` requires metadata annotations not shipped with `alphagenome_pytorch`
- The raw forward pass provides both resolutions and clean tensor outputs

### Channel Filtering

Cell-type-specific channel selection is performed by the `TrackMetadata` class, which reads parquet or CSV metadata files mapping ontology CURIEs to channel indices. After the raw forward pass, padding channels are stripped (e.g., ATAC 256 → 167 real channels), and cell-type channels are selected by index (e.g., HepG2 ATAC = channel 56 for human, ES-CJ7 DNase = channel 23 for mouse).

### Sequence Handling

Input sequences are center-padded with N characters to the nearest multiple of 2048 bp (minimum 4096 bp) to satisfy AlphaGenome's internal pooling requirements. N-padded regions produce near-zero model predictions, so the signal mass scoring naturally discounts padding.

## GPU / HPC Setup

The model runs on CPU but is significantly faster on GPU. The demo notebook completed a forward pass in ~1.7 seconds on CUDA vs. 10+ minutes on CPU [1].

```bash
# GPU environment
conda env create -f environment-gpu.yml
conda activate sox2-alphagenome-gpu
python -c "import torch; print('GPU available:', torch.cuda.is_available())"
```

## Roadmap

- [x] Set up AlphaGenome model loading and inference
- [x] Create project structure and configuration system
- [x] Implement track generation pipeline with cell-type filtering
- [x] Build track metadata system (parquet/CSV) for channel mapping
- [x] Complete zero-shot inference with mESC-specific scoring
- [x] Validate resolution selection (1bp vs 128bp)
- [ ] Implement data preprocessing pipeline
- [ ] Run zero-shot predictions on full dataset
- [ ] Set up fine-tuning training loop for Sox2 expression
- [ ] Add visualization and analysis tools
- [ ] Evaluate on benchmark datasets