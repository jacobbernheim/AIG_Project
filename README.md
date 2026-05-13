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
├── model_utils.py                        # Scoring utils for the model
├── model_utils.py                        # Wrapper for the model
├── track_metadata_human.parquet          # Human track-to-biosample channel mapping
├── track_metadata_mouse.parquet          # Mouse track-to-biosample channel mapping
├── track_metadata.py                     # Helps access track metadata
└── alphagenome_track_celltypes_human.csv # Human track metadata (CSV fallback)

config/
└── default_config.yaml  # Configuration for experiments

notebooks/           # Jupyter notebooks for exploration
results/             # Predictions, metrics, and visualizations

preprocess_data.py      # Data preprocessing and formatting
zero_shot_inference.py  # Zero-shot prediction script
zero_shot_inference.sh  # Zero-shot prediction script for SLURM
finetune.py             # Fine-tuning and prediction script
finetune.sh             # Fine-tuning, prediction, and final eval script for SLURM
final_eval.py           # Final evaluation on the test set script
requirements.txt        # Python dependencies (pip)
environment.yml         # Conda environment specification (CPU)
environment-gpu.yml     # Conda environment specification (GPU/HPC)
install-pip-packages.sh # Package installation helper
setup-gpu.sh            # Package installation helper with conda (GPU)
setup.sh                # Package installation helper with conda
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

### 1. Data Preprocessing

```bash
python preprocess_data.py
```

Reads raw data from `data/raw/` and outputs formatted data to `data/processed/`.

### 2. Zero-Shot Sox2 Expression Prediction

Predicts Sox2 expression levels using mESC-specific AlphaGenome track outputs with a deterministic, WT-normalized scoring formula.

```bash
sbatch zero_shot_inference.sh
```

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
- **DNase weight reduced** from 2.0 to 1.0 to match histone mark weights, since DNase comes from ES-CJ7 (not the preferred BL6 ES-Bruce4 line)
- **EP300 only** for TF binding (CTCF and POLR2A dropped as less directly relevant to enhancer activity)
- **ES-Bruce4 preferred** for all tracks where available, as it corresponds to BL6 mice matching the experimental setting

The raw score is normalized so that PL018 (the WT Sox2 distal LCR sequence) equals 1.0. Weights are configurable in `config/default_config.yaml` under `zero_shot.weights`.

### 2. Fine-tune head for Sox2 Expression Prediction and Evaluate

Fine-tunes a head to predicts Sox2 expression levels using mESC-specific AlphaGenome track outputs. Evaluates performance compared to zero-shot.

```bash
sbatch finetune.sh
```

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
