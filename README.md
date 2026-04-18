# GT to MLP Distillation

A knowledge distillation framework for Graph Transformers using the Nagphormer architecture. This project trains a teacher model and distills its knowledge to student models of varying complexity for efficient node classification on graph datasets.

## Overview

**Nagphormer** is a transformer-based GNN that leverages multi-head attention and k-hop neighborhood aggregation for node classification. This repository implements:

- **Teacher Model Training**: Full Nagphormer architecture training
- **Knowledge Distillation**: Multiple student model variants (MLP, SVD-based attention)
- **Representation Analysis**: CKA (Centered Kernel Alignment) for measuring knowledge transfer
- **Multiple Datasets**: Support for Cora, Pubmed, Citeseer, and other graph datasets

## Architecture

### Teacher Model
- Multi-head transformer with self-attention
- k-hop neighborhood aggregation
- Learnable position embeddings
- Layer normalization and FFN blocks

### Student Models
- **NORMAL_MLP**: Simple 2-layer baseline
- **MLP_ATT_U**: Predicts U matrix from SVD decomposition of attention
- **MLP_ATT_V**: Predicts V matrix from SVD decomposition of attention
- **MLP_ATT_UV**: Combined U and V prediction (low-rank attention approximation)
- **MLP_ATT_GATED**: Gated attention with intermediate features

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.10+
- CUDA (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/aditi-04/GraphTransformer_MLP_distillation.git
cd GraphTransformer_MLP_distillation
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Train Teacher Model
```bash
python -m src.train --dataset pubmed --epochs 2000 --device cuda:0
```

### Train Student Model with Knowledge Distillation
```bash
python src/student_code.py --student normal_mlp --dataset pubmed --device cuda:0
```

### Supported Arguments

**Common Parameters:**
- `--dataset`: Dataset name (pubmed, cora, citeseer, etc.)
- `--device`: Device (cuda:0, cpu, etc.)
- `--seed`: Random seed (default: 3407)
- `--epochs`: Training epochs (default: 2000)

**Model Parameters:**
- `--hops`: Number of neighbor hops (default: 7)
- `--hidden_dim`: Hidden dimension (default: 512)
- `--n_layers`: Number of transformer layers (default: 1)
- `--n_heads`: Number of attention heads (default: 8)

**Training Parameters:**
- `--batch_size`: Batch size (default: 1000)
- `--peak_lr`: Peak learning rate (default: 0.001)
- `--weight_decay`: Weight decay (default: 0.00001)

## Project Structure

```
nagphormer-distillation/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── model.py             # Core transformer architecture
│   ├── train.py             # Teacher model training
│   ├── student_code.py      # Student models and distillation
│   ├── data.py              # Dataset loading and preprocessing
│   ├── utils.py             # Utility functions and metrics
│   ├── early_stop.py        # Early stopping mechanism
│   ├── lr.py                # Learning rate scheduling
│   └── CKA.py               # Representation analysis
├── data/                    # Dataset directory (.gitkeep)
├── .gitignore               # Git ignore rules
├── README.md                # This file
├── requirements.txt         # Python dependencies
└── CODE_OVERVIEW.md         # Code documentation
```

## Key Features

### Multi-Objective Distillation
- **Output Distillation**: Students learn final predictions
- **Feature Distillation**: Students learn intermediate representations
- **SVD Attention Distillation**: Students learn U and V matrices from attention decomposition

### Representation Analysis
- CKA scores to measure knowledge transfer quality
- Monitor alignment between teacher and student features
- Track convergence during training

### Early Stopping
- Patience-based stopping on validation metrics
- Automatic best model checkpoint loading
- Configurable stopping criteria

## Results

The distillation framework achieves:
- Significant parameter reduction (90%+ for simple MLPs)
- Minimal accuracy loss compared to teacher
- Fast inference time for deployed students
- Good generalization to unseen data
