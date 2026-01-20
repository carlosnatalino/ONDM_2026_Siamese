#!/bin/bash
# Training script for all three models: CNN, MLP, and Siamese
# Results will be saved for the visualization notebook

set -e  # Exit on error

echo "========================================================================"
echo "DAS Event Classification - Complete Training Pipeline"
echo "========================================================================"
echo ""

# Activate virtual environment if needed
# source .venv/bin/activate

# Dataset directory
DATA_DIR="/nobackup/carda/datasets/DAS-dataset/data"

# Training parameters
EPOCHS=100
BATCH_SIZE=256
LEARNING_RATE=1e-4
EARLY_STOPPING=20
SEED=42

echo "Training parameters:"
echo "  Data directory: $DATA_DIR"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Early stopping patience: $EARLY_STOPPING"
echo "  Random seed: $SEED"
echo ""

# ============================================================================
# 1. Train CNN Classifier
# ============================================================================
echo "========================================================================"
echo "1/3 Training CNN Classifier"
echo "========================================================================"

CNN_DIR="cnn_results_$(date +%Y%m%d_%H%M%S)"

uv run train_cnn_classifier.py \
    --data_dir $DATA_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --early_stopping_patience $EARLY_STOPPING \
    --dropout 0.5 \
    --seed $SEED \
    --save_dir $CNN_DIR

echo "✓ CNN training complete. Results saved to: $CNN_DIR"
echo ""

# ============================================================================
# 2. Train MLP Classifier
# ============================================================================
echo "========================================================================"
echo "2/3 Training MLP Classifier"
echo "========================================================================"

uv run train_mlp_classifier.py \
    --data_dir $DATA_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --early_stopping_patience $EARLY_STOPPING \
    --dropout 0.3 \
    --seed $SEED

# MLP creates its own timestamped directory
MLP_DIR=$(ls -td mlp_results_* | head -1)
echo "✓ MLP training complete. Results saved to: $MLP_DIR"
echo ""

# ============================================================================
# 3. Train Siamese Multi-Similarity Network
# ============================================================================
echo "========================================================================"
echo "3/3 Training Siamese Multi-Similarity Network"
echo "========================================================================"

SIAMESE_DIR="siamese_multisim_$(date +%Y%m%d_%H%M%S)"

uv run -m siamese_multisim.main \
    --data_dir $DATA_DIR \
    --epochs $EPOCHS \
    --batch_size 64 \
    --lr 1e-4 \
    --embedding_dim 128 \
    --dropout 0.3 \
    --early_stopping 30 \
    --augment \
    --output_dir $SIAMESE_DIR \
    --seed $SEED

echo "✓ Siamese training complete. Results saved to: $SIAMESE_DIR"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "========================================================================"
echo "Training Complete! All Results Saved"
echo "========================================================================"
echo ""
echo "Result Directories:"
echo "  CNN:     $CNN_DIR"
echo "  MLP:     $MLP_DIR"
echo "  Siamese: $SIAMESE_DIR"
echo ""
echo "Next Steps:"
echo "  1. Update visualize_results.ipynb with these directory names:"
echo "     CNN_DIR = \"$CNN_DIR\""
echo "     MLP_DIR = \"$MLP_DIR\""
echo "     SIAMESE_DIR = \"$SIAMESE_DIR\""
echo ""
echo "  2. Run the visualization notebook to generate all figures"
echo ""
echo "========================================================================"
