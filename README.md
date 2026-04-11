# Sox2 Expression Prediction from Regulatory Sequences

Zero-shot and fine-tuned AlphaFold-based prediction of Sox2 expression levels from DNA sequences containing regulatory elements.

**Final project for AI in Genomics 2026**

## Project Structure

```
data/
├── raw/              # Raw input data (DNA sequences + expression labels)
└── processed/        # Processed/prepared data

models/
├── alphafold/        # AlphaFold model checkpoints
└── expression/       # Fine-tuned expression prediction heads

src/
├── __init__.py
├── data_loader.py    # Data loading and preprocessing
├── model_utils.py    # AlphaFold and prediction model wrappers
└── evaluation.py     # Evaluation metrics

config/
└── default_config.yaml  # Configuration for experiments

notebooks/           # Jupyter notebooks for exploration
results/             # Predictions, metrics, and visualizations

preprocess_data.py      # Data preprocessing and formatting
zero_shot_inference.py  # Zero-shot prediction script
requirements.txt        # Python dependencies
```

## Expression Scale

- **0**: No expression
- **1**: Wild-type (WT) expression
- **>1**: Overexpression

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Workflow

This project uses a two-step pipeline:

#### 1. Data Preprocessing

Reads raw data from `data/raw/` and formats it into `data/processed/`:

```bash
python preprocess_data.py
```

This takes:
- `data/raw/Payload Sequences.csv` — DNA sequences with regulatory elements
- `data/raw/Payload Activities.csv` — Sox2 expression levels

And outputs:
- `data/processed/sequences_processed.csv`
- `data/processed/activities_processed.csv`

#### 2. Zero-shot Prediction

Runs inference on preprocessed data:

```bash
python zero_shot_inference.py --output-dir results/
```

This reads from `data/processed/` and saves predictions to `results/`

## Roadmap

- [ ] Implement zero-shot inference pipeline
- [ ] Add data preprocessing utilities
- [ ] Set up fine-tuning training loop
- [ ] Add visualization tools
- [ ] Evaluate on benchmark datasets
