#!/bin/bash
# Install pip packages for sox2-alphagenome environment
# Run this after: conda activate sox2-alphagenome

set -e

ENV_NAME="sox2-alphagenome"

# Check if environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
    echo "ERROR: Please activate the environment first:"
    echo "  conda activate $ENV_NAME"
    exit 1
fi

echo "Installing pip packages in $ENV_NAME environment..."
echo ""

pip install --upgrade pip setuptools wheel

echo "Installing alphagenome-pytorch dependencies..."
pip install huggingface-hub pyfaidx

echo "Installing AlphaGenome PyTorch..."
pip install alphagenome-pytorch==0.2.8

echo ""
echo "✓ All pip packages installed successfully!"
echo ""
echo "To verify installation, run:"
echo "  python example_inference.py"
