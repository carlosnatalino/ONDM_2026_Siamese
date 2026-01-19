# Multi-Similarity Siamese Network for DAS Event Classification

A PyTorch implementation of a multi-similarity Siamese neural network designed for novelty detection and few-shot learning on Distributed Acoustic Sensing (DAS) signals from Φ-OTDR systems.

## Overview

This implementation combines five complementary similarity metrics to enable robust few-shot classification and novelty detection in imbalanced datasets. The network is particularly designed for scenarios where:
- Only a few labeled examples are available per class (few-shot learning)
- New unknown classes may appear during deployment (novelty detection)
- Training data is highly imbalanced
- Real-time classification decisions are required

## Features

### Multi-Similarity Architecture
The network computes similarity using five different metrics:
1. **L1 Distance (Manhattan)**: Robust to outliers, emphasizes sparse differences
2. **L2 Distance (Euclidean)**: Standard geometric distance
3. **Cosine Similarity**: Direction-based, magnitude-invariant comparison
4. **Element-wise Product**: Captures correlation patterns between embeddings
5. **Learned Attention Fusion**: Adaptive weighting of metrics based on input

### Training Strategy
- **Episodic Training**: Class-balanced pair sampling ensures equal representation regardless of class imbalance
- **Data Augmentation**: Frequency masking, noise injection, scaling, and dropout for robustness
- **Multi-Similarity Loss**: BCE loss combined with contrastive regularization

### Evaluation Capabilities
- **N-way K-shot Classification**: Evaluate few-shot performance (5-way 5-shot, 9-way 10-shot, etc.)
- **Novelty Detection**: Identify samples from unseen classes
- **Real-world Simulation**: Incremental class deployment scenarios
- **Comprehensive Metrics**: Accuracy, F1, precision, recall, balanced accuracy

## Architecture

```
Input (2048-dim FFT) → Embedding Network (CNN) → 128-dim Embedding
                                                        ↓
                                    Compute 5 Similarity Metrics
                                                        ↓
                                    Attention-weighted Fusion
                                                        ↓
                                    Comparison Head → Similarity Score [0,1]
```

### Embedding Network
- 3 Conv1D blocks (64, 128, 256 filters)
- BatchNorm + LeakyReLU activation
- MaxPooling for dimensionality reduction
- Dense layers for embedding projection
- L2 normalization for cosine similarity compatibility

## Module Structure

```
siamese_multisim/
├── __init__.py          # Package exports
├── models.py            # Network architectures
├── training.py          # Training components (sampler, dataset, loss, trainer)
├── evaluation.py        # Evaluation utilities (N-way K-shot, novelty detection)
├── visualization.py     # Plotting functions
├── main.py              # Main training/evaluation script
└── README.md            # This file
```

## Usage

### Training

```bash
python -m siamese_multisim.main \
    --data_dir /path/to/DAS-dataset/data \
    --output_dir siamese_results \
    --epochs 100 \
    --batch_size 64 \
    --embedding_dim 128 \
    --learning_rate 1e-3 \
    --dropout 0.3 \
    --early_stopping_patience 15
```

### Training Classes

The network is trained on 5 classes:
- `regular`: Normal baseline conditions
- `walk`, `car`, `manipulation`, `openclose`: Anomaly classes

### Testing

All 9 classes are used for testing, including 4 novel classes not seen during training:
- `fence`, `longboard`, `running`, `construction`

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | Required | Path to DAS dataset |
| `--output_dir` | `siamese_multisim` | Output directory |
| `--epochs` | 100 | Number of training epochs |
| `--batch_size` | 64 | Number of pairs per batch |
| `--embedding_dim` | 128 | Embedding dimensionality |
| `--learning_rate` | 1e-3 | Initial learning rate |
| `--dropout` | 0.3 | Dropout rate |
| `--positive_ratio` | 0.5 | Fraction of same-class pairs |
| `--n_batches_per_epoch` | 100 | Episodes per epoch |
| `--early_stopping_patience` | 15 | Early stopping patience |
| `--use_mlp` | False | Use MLP instead of CNN |
| `--debug` | False | Debug mode with reduced dataset |

## Results

Typical results on the DAS dataset:

### Classification Performance
- **5-way 5-shot**: ~60% accuracy
- **9-way 10-shot**: ~48% accuracy
- **Pair classification**: ~90% accuracy

### Novelty Detection
- **Anomaly F1**: 99%+ (detecting regular vs anomaly classes)
- **Novel class detection**: High precision with threshold tuning

### Advantages
- Handles class imbalance through episodic sampling
- Works with very few examples per class
- Can detect completely unknown classes
- Multiple similarity metrics provide robustness

## Implementation Details

### Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `embedding_dim` | 128 | Compact representation |
| `margin` | 0.5 | Contrastive loss margin |
| `alpha` | 2.0 | Positive pair scaling |
| `beta` | 50.0 | Negative pair scaling |
| `lambda_ms` | 0.1 | Multi-similarity weight |

### Data Augmentation

Applied with 50% probability:
- Gaussian noise: std=0.05
- Random scaling: [0.9, 1.1]
- Frequency masking: SpecAugment-style
- Element dropout: 5% probability

### Loss Function

```python
Total_Loss = BCE_Loss + λ × Multi_Similarity_Regularization

where:
  BCE_Loss = Binary Cross-Entropy on similarity predictions
  MS_Reg = log(1 + exp(α(d_pos - m))) + log(1 + exp(-β(d_neg - m)))
```

## Dependencies

- Python 3.8+
- PyTorch 1.10+
- NumPy
- scikit-learn
- matplotlib
- seaborn

## References

1. Bromley et al., "Signature Verification using a Siamese Time Delay Neural Network" (1993)
2. Koch et al., "Siamese Neural Networks for One-shot Image Recognition" (2015)
3. Snell et al., "Prototypical Networks for Few-shot Learning" (2017)
4. Wang et al., "Multi-Similarity Loss with General Pair Weighting" (2019)
5. Tomasov et al., "Enhancing Perimeter Protection using Φ-OTDR and CNN for Event Classification", OFS 2023

## License

This code is part of research for DAS event classification. Please cite appropriately if used in publications.

## Authors

- Andrei Campeanu
- Carlos Natalino

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{campeanu2026siamese,
  title={Multi-Similarity Siamese Networks for DAS Event Classification},
  author={Campeanu, Andrei and Natalino, Carlos},
  booktitle={International Conference on Machine Learning and Computing Networks},
  year={2026}
}
```
