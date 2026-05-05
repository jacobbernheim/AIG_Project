#!/bin/bash
# Setup script for Sox2 expression prediction project
# Detects if conda environment exists, creates if needed

set -e  # Exit on error

echo "=========================================="
echo "Sox2 Expression Prediction - Setup Script"
echo "=========================================="
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda from: https://www.anaconda.com/download"
    exit 1
fi

echo "✓ conda found"
echo ""

# Check if environment already exists
ENV_NAME="sox2-alphagenome"
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
            echo "Creating conda environment from environment.yml..."
            echo ""
            conda env create -f environment.yml --yes

            echo ""
            echo "Installing pip packages..."
            conda run -n $ENV_NAME pip install --upgrade pip setuptools wheel
            conda run -n $ENV_NAME pip install -r requirements.txt

            echo ""
            echo "=========================================="
            echo "Environment recreated successfully!"
            echo "=========================================="
            echo ""
            echo "To activate the environment, run:"
            echo "  conda activate $ENV_NAME"
            echo ""
            echo "To verify installation, run:"
            echo "  conda activate $ENV_NAME"
            echo "  python example_inference.py"
            echo ""
            ;;
        *)
            echo "Keeping the existing environment unchanged."
            echo ""
            echo "To activate the environment, run:"
            echo "  conda activate $ENV_NAME"
            echo ""
            exit 0
            ;;
    esac
else
    echo "Creating conda environment from environment.yml..."
    echo ""
    conda env create -f environment.yml --yes
    
    echo ""
    echo "Installing pip packages..."
    conda run -n $ENV_NAME pip install --upgrade pip setuptools wheel
    conda run -n $ENV_NAME pip install -r requirements.txt
    
    echo ""
    echo "=========================================="
    echo "Environment created successfully!"
    echo "=========================================="
    echo ""
    echo "To activate the environment, run:"
    echo "  conda activate $ENV_NAME"
    echo ""
    echo "To verify installation, run:"
    echo "  conda activate $ENV_NAME"
    echo "  python example_inference.py"
    echo ""
    echo "To deactivate the environment when done, run:"
    echo "  conda deactivate"
    echo ""
fi
