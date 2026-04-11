# Changelog

All notable changes to this project will be documented in this file.

## [0.4.2] - 2026-04-11 (PyTorch Installation Fix)

### Fixed
- PyTorch installation issue by using conda pytorch channel instead of pip
- alphagenome-pytorch now compatible with PyTorch installed from conda
- Resolved "torch not available" error on macOS by using pytorch channel
- Installation now works without dependency resolution conflicts

### Changed
- `environment.yml`: Added pytorch channel, restored pytorch/torchvision from pytorch channel
- Setup scripts: Removed torch from pip installation (comes from conda)
- PyTorch is now installed via conda, alphagenome-pytorch via pip
- Installation order: conda creates env with PyTorch → pip installs alphagenome-pytorch

### Purpose
Ensure reliable installation on macOS by leveraging conda's pytorch channel for PyTorch.

## [0.4.1] - 2026-04-11 (PyTorch Dependency Fix)

### Fixed
- PyTorch installation issue by moving PyTorch to pip installation
- alphagenome-pytorch now uses pip for PyTorch (>= 2.4.0) instead of conda
- Conda no longer locks PyTorch version, allowing pip to install latest compatible version
- Resolves dependency conflicts with PyTorch 2.4+ requirements

### Changed
- `environment.yml`: Removed PyTorch from conda dependencies
- PyTorch now installed via pip (setup.sh, setup.bat, install-pip-packages.sh)
- Updated requirements.txt to reflect pip-first installation strategy

### Purpose
Ensure alphagenome-pytorch gets the necessary PyTorch 2.4+ without conda channel limitations.

## [0.4.0] - 2026-04-11 (GPU/HPC Support)

### Added
- `environment-gpu.yml` for GPU-enabled training on HPC systems
- GPU/HPC setup instructions in README with SLURM job script example
- Support for both CPU (local) and GPU (HPC) workflows
- CUDA 12.1 support for modern GPU clusters

### Purpose
Enable seamless transition from local CPU development to HPC GPU training without code changes.

## [0.3.0] - 2026-04-11 (Conda Environment Setup)

### Added
- `environment.yml` for reproducible conda environment
- `setup.sh` for automated environment creation (macOS/Linux)
- `setup.bat` for automated environment creation (Windows)
- Enhanced README with conda and collaboration guidance

### Purpose
Make it easy to:
- Create isolated environments without affecting other projects
- Share exact configuration with collaborators
- Ensure reproducible results across different machines

## [0.2.0] - 2026-04-11 (AlphaGenome Integration)

### Added
- AlphaGenome model wrapper (`GenomeModel` class) with full inference support
- Download and caching of AlphaGenome weights from HuggingFace
- Support for multiple genomic tracks (ATAC, DNase, ChIP-seq, RNA-seq)
- Feature aggregation utilities for downstream prediction
- Enhanced `SequenceDataset` with sequence validation and filtering
- Example inference script demonstrating model usage
- PyTorch and alphagenome-pytorch to requirements

### Changed
- Replaced AlphaFold reference with AlphaGenome implementation
- Updated model_utils.py with AlphaGenome-specific classes
- Enhanced configuration system for AlphaGenome parameters
- Improved data loading for genomic sequences
- Updated zero_shot_inference.py to use AlphaGenome predictions

### Updated
- requirements.txt with new dependencies
- README with AlphaGenome documentation and roadmap

## [0.1.0] - 2026-04-11

### Added
- Initial project structure
- Core modules: `data_loader.py`, `model_utils.py`, `evaluation.py`
- Zero-shot inference script
- Configuration system (YAML-based)
- Basic documentation and requirements
