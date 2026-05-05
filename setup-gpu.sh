#!/bin/bash
# Setup script for Sox2 expression prediction project (GPU/HPC)
# Detects if the GPU conda environment exists, creates it if needed

set -e  # Exit on error

echo "=========================================="
echo "Sox2 Expression Prediction - GPU Setup"
echo "=========================================="
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda is not installed or not in PATH"
    echo "Please load your HPC conda module or install Miniconda/Anaconda first."
    exit 1
fi

echo "✓ conda found"
echo ""

# Check if environment already exists
ENV_NAME="sox2-alphagenome-gpu"
if conda env list | grep -q "^$ENV_NAME "; then
    echo "✓ Environment '$ENV_NAME' already exists"
    echo ""
    read -r -p "Recreate it from scratch? [r]emake / [e]xit: " choice
    case "$choice" in
        r|R|remake|Remake|REMAKE)
            echo ""
            echo "Removing existing environment..."
            conda env remove -n $ENV_NAME --yes

            echo ""
            echo "Creating GPU conda environment from environment-gpu.yml..."
            echo ""
            conda env create -f environment-gpu.yml --yes

            echo ""
            echo "Installing pip packages..."
            conda run -n $ENV_NAME pip install --upgrade pip setuptools wheel
            conda run -n $ENV_NAME pip install -r requirements.txt

            echo ""
            echo "=========================================="
            echo "GPU environment recreated successfully!"
            echo "=========================================="
            echo ""
            echo "To activate the environment, run:"
            echo "  conda activate $ENV_NAME"
            echo ""
            echo "To verify GPU access, run:"
            echo "  conda activate $ENV_NAME"
            echo "  python -c \"import torch; print('GPU available:', torch.cuda.is_available())\""
            echo ""
            ;;
        *)
            echo "Keeping the existing GPU environment unchanged."
            echo ""
            echo "To activate the environment, run:"
            echo "  conda activate $ENV_NAME"
            echo ""
            exit 0
            ;;
    esac
else
    echo "Creating GPU conda environment from environment-gpu.yml..."
    echo ""
    conda env create -f environment-gpu.yml --yes
    
    echo ""
    echo "Installing pip packages..."
    conda run -n $ENV_NAME pip install --upgrade pip setuptools wheel
    conda run -n $ENV_NAME pip install -r requirements.txt
    
    echo ""
    echo "=========================================="
    echo "GPU environment created successfully!"
    echo "=========================================="
    echo ""
    echo "To activate the environment, run:"
    echo "  conda activate $ENV_NAME"
    echo ""
    echo "To verify GPU access, run:"
    echo "  conda activate $ENV_NAME"
    echo "  python -c \"import torch; print('GPU available:', torch.cuda.is_available())\""
    echo ""
    echo "To deactivate the environment when done, run:"
    echo "  conda deactivate"
    echo ""
fi