# Towards Generalizable ML-Based Event Recognition Using Φ-OTDR and Siamese Networks

This repository contains the code for the paper:

> **"Towards Generalizable ML-Based Event Recognition Using Φ-OTDR and Siamese Networks"**  
> Published at ONDM 2026
> 
> Authors: Andrei Ribeiro, Fabrício Rossy, João C. W. A. Costa, Paolo Monti, and Carlos Natalino

```bibtex
@inproceedings{Ribeiro_2026_GeneralizableMLBasedEvent,
  title = {Towards Generalizable {ML}-Based Event Recognition Using {P}hi-{OTDR} and {S}iamese Networks},
  booktitle = {International Conference on Optical Network Design and Modeling (ONDM)},
  author = {Ribeiro, Andrei Nogueira and Lobato, Fabr{\'i}cio Rossy de Lima and Costa, Jo{\~a}o Cris{\'o}stomo Weyl Albuquerque and Monti, Paolo and Natalino, Carlos},
  year = 2026,
  month = may,
}
```

We compare three approaches for classifying events in Distributed Acoustic Sensing (DAS) signals across 9 classes: a CNN classifier, an MLP classifier, and a proposed Multi-Similarity Siamese Network (MS-SNN) evaluated under few-shot (N-way K-shot) settings.

## Repository Structure

```
├── README.md
├── requirements.in / requirements.txt
├── data_loader.py                        # DAS HDF5 dataset loader
├── train_cnn.py                          # Train the CNN baseline
├── train_mlp.py                          # Train the MLP baseline
├── train_siamese.py                      # Train the MS-SNN
├── evaluate_siamese.py                   # Evaluate MS-SNN (N-way K-shot)
├── visualize_results.ipynb               # Generate all paper figures and tables
└── siamese_multisim/                     # MS-SNN package
    ├── __init__.py
    ├── models.py                         # Network architectures
    ├── training.py                       # Training loop
    ├── evaluation.py                     # Evaluation routines
    └── visualization.py                  # Plotting utilities
```

## Dataset

This repository does not contain the original dataset. You must download it by yourself from [^1].

The dataset has been described in [^2].

It contains 9 event classes recorded with an OptaSense ODH interrogator:
`car`, `construction`, `fence`, `longboard`, `manipulation`, `openclose`, `regular`, `running`, `walk`

Each class directory contains `*.h5` raw measurement files and `*.npy` bitmap files marking valid windows.

You must use the path to where the dataset is stored in the steps to reproduce the results below.

## Setup

**Requirements:** Python 3.10+ (results generated with Python 3.13), PyTorch (CUDA recommended for training)

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

The `requirements.txt` file contains the specific versions used to generate the results in the paper. However, you can also build a new file from the `requirements.in` one.

## Reproducing Results

Follow steps 1–4 in order. All scripts accept `--data_dir` pointing to the dataset root (which contains the `data` and `python` folders).

### Step 1 — Train CNN Classifier

```bash
python train_cnn.py \
    --data_dir /path/to/DAS-dataset/data \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-4 \
    --dropout 0.5 \
    --seed 42 \
    --save_dir cnn_results
```

Outputs saved to `cnn_results`:
- `history.npy` — training curves
- `test_results.npy` — accuracy, balanced accuracy, F1, confusion matrix
- figures and weights

### Step 2 — Train MLP Classifier

```bash
python train_mlp.py \
    --data_dir /path/to/DAS-dataset/data \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-4 \
    --dropout 0.5 \
    --seed 42
```

Outputs saved to `mlp_results/`:
- `history.npy` — training curves
- `test_results.npy` — accuracy, balanced accuracy, F1, confusion matrix
- figures and weights

### Step 3 — Train MS-SNN

The MS-SNN is trained on 5 seen classes (`regular`, `walk`, `car`, `manipulation`, `openclose`) and evaluated on all 9 classes (including 4 unseen: `fence`, `longboard`, `construction`, `running`) in a few-shot setting.

```bash
python train_siamese.py \
    --data_dir /path/to/DAS-dataset/data \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-4 \
    --embedding_dim 128 \
    --dropout 0.3 \
    --seed 42
```

Outputs saved to `siamese_multisim/`:
- `best_model.pth` — best model checkpoint
- `history.npy` — training curves
- figures and weights

### Step 4 — Evaluate MS-SNN on Full Test Set (N-way K-shot)

This script classifies all test samples using prototype-based few-shot evaluation across K={1,5,10,15,20} support shots and both 5-way and 9-way settings.

```bash
python evaluate_siamese.py \
    --checkpoint siamese_multisim_<timestamp>/best_model.pth \
    --data_dir /path/to/DAS-dataset/data \
    --output_dir results_siamese_no_decimation_results \
    --seed 42
```

Output: `results_siamese_no_decimation_results/siamese_full_test_results.pkl`

### Step 5 — Generate Figures and Tables

Update the result directory names at the top of `visualize_results.ipynb` to match those created in steps 1–4:

```python
CNN_DIR     = "results_cnn_<timestamp>"
MLP_DIR     = "results_mlp_<timestamp>"
SIAMESE_DIR = "results_siamese_<timestamp>"
```

The Siamese N-way K-shot results are read from the fixed path `results_siamese_no_decimation_results/siamese_full_test_results.pkl` as written by Step 4. Run all cells to generate all paper figures (PDF) and LaTeX tables into `paper_figures/`.

---

## Models

### CNN Classifier
1D convolutional network following the architecture from the OFS paper in [^3]:
`Conv1D(64) → LeakyReLU → MaxPool → Conv1D(256) → LeakyReLU → MaxPool → Dense(1024) → Dense(9)`

### MLP Classifier
Three-layer MLP:  
`Dense(2048→1024) → Dense(1024→512) → Dense(512→9)` with batch normalization and dropout.

### MS-SNN (Multi-Similarity Siamese Network)
Siamese network with a shared 1D-CNN embedding backbone and a multi-similarity comparison head combining L1, L2, cosine, and element-wise product distances with learned attention weights. Evaluated in a prototypical few-shot framework (9-way, K-shot).

## References

[^1]: "Comprehensive Dataset for Event Classification Using Distributed Acoustic Sensing (DAS) Systems" available at: https://doi.org/10.6084/m9.figshare.27004732

[^2]: Tomasov, A., Zaviska, P., Dejdar, P. et al. Comprehensive Dataset for Event Classification Using Distributed Acoustic Sensing (DAS) Systems. Sci Data 12, 793 (2025). https://doi.org/10.1038/s41597-025-05088-4

[^3]: A. Tomasov, P. Zaviska, V. Spurny, P. Dejdar, P. Munster, T. Horvath, and O. Klicnik, "Enhancing Perimeter Protection using Φ-OTDR and CNN for Event Classification," in 28th International Conference on Optical Fiber Sensors, Technical Digest Series (Optica Publishing Group, 2023), paper W4.39. https://doi.org/10.1364/OFS.2023.W4.39