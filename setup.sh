#!/bin/bash

echo "Setting up Medical VQA Project..."

# Create directory structure
mkdir -p data/raw data/processed/splits
mkdir -p models preprocessing training evaluation scripts configs
mkdir -p results/baseline results/blip2 results/figures
mkdir -p notebooks

# Create __init__.py files for Python packages
touch models/__init__.py
touch preprocessing/__init__.py
touch training/__init__.py
touch evaluation/__init__.py

echo "✓ Directory structure created"

# Install Python dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo "✓ Dependencies installed"

echo ""
echo "Setup complete! Next steps:"
echo "1. Download dataset: python scripts/download_dataset.py"
echo "2. Preprocess data: python scripts/preprocess_data.py"
echo "3. Train baseline: python training/train_baseline.py"
echo "4. Or run quick test: python scripts/quick_start.py"