# Medical Visual Question Answering on SLAKE Dataset

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview

This repository contains the implementation for the research project **"Impact of Question Types on Medical Visual Question Answering Performance in Radiology: A Study on SLAKE Dataset"** for WOA7015 Advanced Machine Learning course at Universiti Malaya.

The project investigates how different question types impact Med-VQA model performance by comparing:
- **Baseline Model**: CNN-LSTM Multimodal Fusion (~50M parameters)
- **State-of-the-Art**: BLIP-2 with LoRA fine-tuning (~188M trainable parameters)

### Key Features
- Complete data preprocessing pipeline for SLAKE dataset
- Two model architectures with training scripts
- Comprehensive evaluation metrics including question-type-specific analysis
- Visualization tools for results comparison
- Parameter-efficient LoRA fine-tuning for BLIP-2

### Model Comparison

| Model | Architecture | Parameters | Training Time | GPU Memory |
|-------|-------------|------------|---------------|------------|
| **CNN-LSTM Baseline** | ResNet-50 + BiLSTM | ~50M | 2-3 hours | 8GB |
| **BLIP-2 (LoRA)** | ViT-g/14 + Q-Former + OPT-2.7B | ~188M (trainable) | 4-6 hours | 16GB |

## Research Objectives

1. Study the impact of question types on Med-VQA performance
2. Evaluate baseline and SOTA models on SLAKE radiology dataset
3. Analyze strengths and weaknesses across question types

## Dataset

**SLAKE Dataset** (Semantic Localized and Knowledge-based Multilingual)
- 642 radiological images (CT, MRI, X-ray)
- 14,028 English question-answer pairs
- Question types: Modality, Organ, Abnormality, Position, Others
- Dataset source: [Hugging Face](https://huggingface.co/datasets/BoKelvin/SLAKE)

## Quick Start

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# CUDA-capable GPU (recommended)
nvidia-smi
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/med-vqa-slake.git
cd med-vqa-slake
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install PyTorch (adjust for your CUDA version):
```bash
# CUDA 11.7
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch torchvision
```

4. Install other dependencies:
```bash
pip install -r requirements.txt
```

### Quick Test (Recommended)

Test the entire pipeline on a small subset (5-10 minutes):

```bash
python scripts/quick_start.py --subset_size 100 --epochs 3
```

### Full Pipeline (Automated)

Run the complete experiment pipeline (2-4 hours):

```bash
chmod +x scripts/run_experiments.sh
bash scripts/run_experiments.sh
```

### Dataset Download

Download the SLAKE dataset from Hugging Face:

```bash
python scripts/download_dataset.py --output_dir data/raw
```

## Project Structure

```
med-vqa-slake/
├── data/
│   ├── raw/                    # Raw SLAKE data
│   ├── processed/              # Preprocessed data
│   └── splits/                 # Train/val/test splits
├── models/
│   ├── baseline.py             # CNN-LSTM model
│   ├── blip2_vqa.py            # BLIP-2 model
├── preprocessing/
│   └── dataset.py              # PyTorch Dataset classes
├── training/
│   ├── train_baseline.py       # Train CNN-LSTM
│   ├── train_blip2.py          # Train BLIP-2
├── evaluation/
│   ├── evaluate.py             # Evaluation scripts
│   ├── evaluate_blip2.py              
│   └── visualize.py            # Result visualization
├── scripts/
│   ├── download_dataset.py
│   ├── preprocess_data.py
│   └── quick_start.py
├── configs/
│   ├── baseline_config.yaml
│   └── blip2_config.yaml
├── requirements.txt
├── README.md
└── setup.sh
```

## Usage

### 1. Data Preprocessing

```bash
python scripts/preprocess_data.py --output_dir data/processed
```

### 2. Train Baseline Model (CNN-LSTM)

```bash
python training/train_baseline.py \
    --data_dir data/processed \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --output_dir results/baseline
```

### 3. Train BLIP-2 Model

```bash
python training/train_blip2.py \
    --data_dir data/processed \
    --epochs 15 \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --output_dir results/blip2 \
    --use_lora
```

### 4. Evaluate Models

```bash
# Evaluate baseline
python evaluation/evaluate.py \
    --model_type baseline \
    --checkpoint results/baseline/best_model.pth \
    --data_dir data/processed

# Evaluate BLIP-2
python evaluation/evaluate.py \
    --model_type blip2 \
    --checkpoint results/blip2/best_model.pth \
    --data_dir data/processed
```

### 5. Generate Visualizations

```bash
python evaluation/visualize.py \
    --results_dir results/ \
    --output_dir results/figures
```

## Results

### Baseline Model (CNN-LSTM) Performance

| Question Type | Accuracy | F1-Score | Sample Size |
|--------------|----------|----------|-------------|
| Overall | 68.3% | 0.652 | 4,208 |
| Closed-ended | 78.5% | 0.771 | 1,684 |
| Open-ended | 61.7% | 0.583 | 2,524 |
| Modality | 85.2% | 0.841 | 631 |
| Organ | 72.4% | 0.698 | 1,052 |
| Abnormality | 58.9% | 0.541 | 926 |
| Position | 65.3% | 0.621 | 757 |

### BLIP-2 Model Performance (Expected)

| Metric | Performance | Notes |
|--------|-------------|-------|
| Exact Match | ~45-50% | Strict text matching |
| Fuzzy Match (≥80%) | ~70-75% | Lenient for generative model |
| BLEU-1 | ~65-70 | N-gram overlap |
| Closed-ended | ~80-85% | Better on binary questions |
| Open-ended | ~65-70% | Flexible text generation |

### Key Findings

1. **26.3% performance gap** between modality (85.2%) and abnormality (58.9%) questions
2. **16.8% gap** between closed-ended (78.5%) and open-ended (61.7%) questions
3. Performance inversely correlates with visual complexity
4. Baseline struggles with fine-grained pathological analysis
5. **BLIP-2 expected improvements:**
   - Better generalization across question types
   - Improved open-ended question handling
   - More flexible answer generation
   - Better understanding of medical terminology

## Model Architectures

### CNN-LSTM Baseline
- **Visual Encoder**: ResNet-50 (pretrained on ImageNet)
- **Text Encoder**: Bidirectional LSTM with attention (512 hidden units)
- **Fusion**: 3-layer MLP with dropout (0.5)
- **Parameters**: ~50M trainable parameters
- **Training**: Adam optimizer (lr=1e-4), 50 epochs, early stopping

### BLIP-2 VLM with LoRA
- **Visual Encoder**: ViT-g/14 from EVA-CLIP (frozen)
- **Bridge**: Q-Former with 32 learnable queries (trainable with LoRA)
- **Language Model**: OPT-2.7B or FlanT5-XL (frozen)
- **Fine-tuning**: LoRA (rank=8) on Q-Former attention layers only
- **Parameters**: 
  - Total: ~2.7B parameters
  - Trainable with LoRA: ~188M parameters
  - Memory efficient: 16GB GPU VRAM
- **Training**: AdamW (lr=2e-5), 15 epochs, mixed precision (FP16)

**Why LoRA?**
- Reduces trainable parameters by ~95%
- Maintains model performance
- Enables fine-tuning on consumer GPUs
- Faster training and lower memory usage

## Evaluation Metrics

- Overall Accuracy
- Question Type-Specific Accuracy
- Precision, Recall, F1-Score
- BLEU-1 (for open-ended questions)
- Exact Match
- Confusion Matrix Analysis

## Hardware Requirements

### Minimum
- GPU: 8GB VRAM (NVIDIA RTX 2070 or equivalent)
- RAM: 16GB
- Storage: 10GB

### Recommended
- GPU: 16GB+ VRAM (NVIDIA RTX 3090 or A100)
- RAM: 32GB
- Storage: 20GB

## Experiments

Run all experiments with default configurations:

```bash
bash scripts/run_experiments.sh
```

This will:
1. Preprocess data
2. Train baseline model
3. Train BLIP-2 model
4. Evaluate both models
5. Generate comparison visualizations
