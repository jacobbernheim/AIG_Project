# Sox2 Expression Prediction from Regulatory Sequences

Zero-shot and fine-tuned AlphaGenome-based prediction of Sox2 expression levels from DNA sequences containing regulatory elements.

**Final project for AI in Genomics 2026**

## Project Structure

```
data/
├── raw/              # Raw input data (DNA sequences + expression labels)
└── processed/        # Processed/prepared data

models/               # AlphaGenome and expression model checkpoints

src/
├── __init__.py
├── data_loader.py    # Data loading and preprocessing
├── model_utils.py    # AlphaGenome and expression prediction wrappers
└── evaluation.py     # Evaluation metrics

config/
└── default_config.yaml  # Configuration for experiments

notebooks/           # Jupyter notebooks for exploration
results/             # Predictions, metrics, and visualizations

preprocess_data.py      # Data preprocessing and formatting
zero_shot_inference.py  # Zero-shot prediction script
example_inference.py    # Example script to verify setup
setup.sh               # Setup script (macOS/Linux)
setup-gpu.sh           # Setup script (GPU/HPC)
setup.bat              # Setup script (Windows)
requirements.txt       # Python dependencies (pip)
environment.yml        # Conda environment specification (CPU)
environment-gpu.yml    # Conda environment specification (GPU/HPC)
```

## Expression Scale

- **0**: No expression
- **1**: Wild-type (WT) expression
- **>1**: Overexpression

## Getting Started

### Quick Setup (Recommended)

We provide setup scripts for easy environment creation:

**macOS/Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

**Windows:**
```bash
setup.bat
```

These scripts will create the conda environment automatically.

### Manual Setup

Alternatively, create the environment manually:

**Create the environment (one-time setup):**
```bash
conda env create -f environment.yml
```

**Activate the environment:**
```bash
conda activate sox2-alphagenome
```

PyTorch is automatically installed from conda's pytorch channel during environment creation. The setup scripts then install alphagenome-pytorch and other dependencies via pip.

### Verify Installation

Run the example script to verify everything works:

```bash
python example_inference.py
```

This will download the AlphaGenome model (~680 MB, cached after first run) and run a test inference.

### Alternative: Install with pip only

If you prefer pip without conda:

```bash
pip install -r requirements.txt
```

However, **conda is recommended** for this project due to PyTorch dependencies and ease of sharing.

## GPU / HPC Setup

### Local Development (CPU)

The default `environment.yml` uses CPU-only PyTorch, which is perfect for macOS and local development.

### Training on HPC with GPU

When you're ready to train on HPC GPU nodes, use the GPU environment and dedicated setup script:

**Create GPU environment on HPC:**
```bash
chmod +x setup-gpu.sh
./setup-gpu.sh
```

**HPC Setup Instructions:**

1. **Clone repository on HPC:**
   ```bash
   git clone <repo-url>
   cd AIG_Project
   ```

2. **Load HPC modules (example for typical HPC):**
   ```bash
   module load conda
   # or: module load miniconda/latest
   ```

3. **Create GPU environment:**
   ```bash
   chmod +x setup-gpu.sh
   ./setup-gpu.sh
   ```

4. **Activate and verify GPU access:**
   ```bash
   conda activate sox2-alphagenome-gpu
   python -c "import torch; print('GPU available:', torch.cuda.is_available())"
   ```

5. **Submit training job (example SLURM script):**
   ```bash
   #!/bin/bash
   #SBATCH --nodes=1
   #SBATCH --gpus=1
   #SBATCH --time=04:00:00
   
   conda activate sox2-alphagenome-gpu
   
   # Your training command here
   python your_training_script.py
   ```

**CUDA Version Note:** `environment-gpu.yml` uses CUDA 12.1 with PyTorch 2.5.1. If your HPC uses a different CUDA version:
- Check available CUDA: `module avail cuda`
- Contact HPC support to confirm CUDA version
- We can adjust `environment-gpu.yml` if needed

**Note:** If your HPC admin requires a different CUDA module version, we may need to pin a different `pytorch-cuda` build in `environment-gpu.yml`.

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

### Sharing with Collaborators

The `environment.yml` and setup scripts make it easy for collaborators to replicate your exact setup:

**Collaborator setup (macOS/Linux):**
```bash
git clone <repo-url>
cd AIG_Project
chmod +x setup.sh
./setup.sh
conda activate sox2-alphagenome
python example_inference.py
```

**Collaborator setup (Windows):**
```bash
git clone <repo-url>
cd AIG_Project
setup.bat
conda activate sox2-alphagenome
python example_inference.py
```

This ensures:
- ✓ Same Python version
- ✓ Identical package versions (reproducible)
- ✓ Same PyTorch configuration
- ✓ No conflicts with other projects
- ✓ Works across macOS, Linux, and Windows

## About AlphaGenome

This project uses [AlphaGenome PyTorch](https://github.com/genomicsxai/alphagenome-pytorch), a faithful port of Google DeepMind's AlphaGenome to PyTorch. AlphaGenome predicts hundreds of genomic tracks (ATAC-seq, DNase, ChIP-seq, RNA-seq) at single base-pair resolution from DNA sequences up to 1 Mb long.

## Roadmap

- [x] Set up AlphaGenome model loading and inference
- [x] Create project structure and configuration system
- [ ] Implement data preprocessing pipeline
- [ ] Complete zero-shot inference on full dataset
- [ ] Set up fine-tuning training loop for Sox2 expression
- [ ] Add visualization and analysis tools
- [ ] Evaluate on benchmark datasets
