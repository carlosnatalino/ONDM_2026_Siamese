# DAS Event Classification: Comparative Study

This repository contains implementations of multiple neural network approaches for event classification on Distributed Acoustic Sensing (DAS) signals from Φ-OTDR systems.

## Overview

This project compares several distinct approaches to DAS event classification:

### Traditional Classifiers
1. **CNN Classifier** (`train_cnn_classifier.py`): 1D Convolutional Neural Network
2. **MLP Classifier** (`train_mlp_classifier.py`): Feed-Forward Neural Network

### Metric Learning Approaches  
3. **Siamese Network** (`siamese.py`): Contrastive learning with embedding networks
4. **Siamese with Triplet Loss** (`siamese_triplet.py`): Prototypical Networks with triplet loss
5. **Multi-Similarity Siamese** (`siamese_multisim/`): Advanced Siamese with 5 similarity metrics for few-shot learning

### Foundation Models
6. **DAS Foundation Model** (`das_foundation_model.py`): Transfer learning approach

## Repository Structure

```
.
├── siamese_multisim/          # Multi-similarity Siamese network package
│   ├── models.py              # Network architectures
│   ├── training.py            # Training components
│   ├── evaluation.py          # Evaluation utilities
│   ├── visualization.py       # Plotting functions
│   ├── main.py                # Main training script
│   └── README.md              # Detailed documentation
├── train_cnn_classifier.py    # CNN training script
├── train_mlp_classifier.py    # MLP training script
├── siamese.py                 # Siamese with contrastive loss
├── siamese_triplet.py         # Prototypical Networks with triplet loss
├── das_foundation_model.py    # Foundation model approach
├── compare_approaches.py      # Generate comparison plots for paper
├── data_loader.py             # DAS dataset loader
├── requirements.txt           # Python dependencies
└── PROJECT_README.md          # This file
```

## Dataset

The code expects the DAS dataset with the following structure:

```
/path/to/DAS-dataset/data/
├── regular/
├── walk/
├── car/
├── construction/
├── fence/
├── longboard/
├── manipulation/
├── openclose/
└── running/
```

Each directory contains `.h5` files with DAS signal recordings.

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 1.10+
- NumPy
- scikit-learn
- matplotlib
- seaborn
- h5py

## Quick Start

### 1. CNN Classifier

```bash
python train_cnn_classifier.py \
    --data_dir /path/to/DAS-dataset/data \
    --epochs 100 \
    --batch_size 256 \
    --lr 1e-3 \
    --early_stopping_patience 20
```

**Expected Results:**
- Test Accuracy: ~78-79%
- Training time: ~30-40 minutes on RTX 2080 Ti

### 2. FFNN Classifier

```bash
python train_ffnn_classifier.py \
    --data_dir /path/to/DAS-dataset/data \
    --epochs 100 \
    --batch_size 256 \
    --lr 1e-3 \
    --early_stopping 20
```

**Expected Results:**
- Test Accuracy: ~89-90%
- Training time: ~25-35 minutes on RTX 2080 Ti

### 3. Siamese Network

```bash
python -m siamese_multisim.main \
    --data_dir /path/to/DAS-dataset/data \
    --epochs 100 \
    --batch_size 64 \
    --learning_rate 1e-3 \
    --early_stopping_patience 15
```

**Expected Results:**
- 5-way 5-shot accuracy: ~58-60%
- Anomaly detection F1: ~99%
- Training time: ~40-50 minutes on RTX 2080 Ti

See `siamese_multisim/README.md` for detailed documentation.

## Key Features

### CNN Classifier
- 1D Convolutional architecture
- Based on Tomasov et al. (2023) OFS paper
- Handles class imbalance with class weights
- StepLR scheduler with decay at epoch 50

### FFNN Classifier
- Fully-connected baseline
- 4-layer architecture (2048→1024→512→256→9)
- BatchNorm and Dropout regularization
- Achieves best overall accuracy

### Siamese Network
- **Few-shot learning**: Works with minimal examples
- **Novelty detection**: Identifies unknown classes
- **Multi-similarity**: 5 complementary metrics (L1, L2, Cosine, Product, Attention)
- **Class-balanced training**: Episodic sampling for imbalanced data

## Comparison

| Approach | Accuracy | F1 Score | Training Time | Few-shot | Novelty Detection |
|----------|----------|----------|---------------|----------|-------------------|
| CNN      | 78.65%   | 0.79     | ~35 min       | ✗        | ✗                 |
| FFNN     | 89.47%   | 0.89     | ~30 min       | ✗        | ✗                 |
| Siamese  | 74.71%*  | -        | ~45 min       | ✓        | ✓                 |

*Siamese: Trained on 5 classes, evaluated on all 9 classes in few-shot setting

## Common Arguments

| Argument | CNN | FFNN | Siamese | Description |
|----------|-----|------|---------|-------------|
| `--data_dir` | ✓ | ✓ | ✓ | Path to dataset |
| `--epochs` | ✓ | ✓ | ✓ | Number of epochs |
| `--batch_size` | ✓ | ✓ | ✓ | Batch size |
| `--lr` | ✓ | ✓ | ✓ | Learning rate |
| `--dropout` | ✓ | ✓ | ✓ | Dropout rate |
| `--output_dir` | ✓ | ✓ | ✓ | Output directory |
| `--seed` | ✓ | ✓ | ✓ | Random seed |

## Output

Each training script produces:
- Training curves (loss, accuracy, F1)
- Confusion matrix
- Classification report
- Model checkpoint (.pth file)
- Training logs

## Data Preprocessing

All models use:
- **FFT transformation**: 8192 → 2048 frequency bins
- **Z-score normalization**: Mean 0, std 1
- **Noise filtering**: Removes low-energy samples
- **Class balancing**: Via sampling (Siamese) or class weights (CNN/FFNN)

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{campeanu2026das,
  title={Comparative Analysis of Neural Network Approaches for DAS Event Classification},
  author={Campeanu, Andrei and Natalino, Carlos},
  booktitle={International Conference on Machine Learning and Computing Networks},
  year={2026}
}
```

## References

1. Tomasov et al., "Enhancing Perimeter Protection using Φ-OTDR and CNN for Event Classification", Optical Fiber Sensors (OFS) 2023
2. Bromley et al., "Signature Verification using a Siamese Time Delay Neural Network" (1993)
3. Koch et al., "Siamese Neural Networks for One-shot Image Recognition" (2015)
4. Snell et al., "Prototypical Networks for Few-shot Learning" (2017)

## License

This code is provided for research purposes. Please cite appropriately if used in publications.

## Authors

- Andrei Ribeiro
- Carlos Natalino

## Contact

For questions or issues, please open a GitHub issue or contact the authors.
