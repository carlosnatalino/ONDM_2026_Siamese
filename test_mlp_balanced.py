#!/usr/bin/env python3
"""
Test MLP Classifier on Full Dataset (No Decimation)

Re-evaluates the trained MLP model on the full test set
(no decimation, 80/10/10 split) for fair comparison with CNN.

Author: Andrei Ribeiro
Date: January 2026
"""
import os
import sys
import logging
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, classification_report, confusion_matrix
)

# Import data loader and model
from data_loader import DASDataLoader, fft
from train_mlp_classifier import DASEventClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_device():
    """Get best available device."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def load_model(checkpoint_path: str, num_classes: int, device: torch.device):
    """Load trained MLP model."""
    logger.info(f"Loading model from: {checkpoint_path}")
    
    model = DASEventClassifier(
        input_dim=2048,
        num_classes=num_classes,
        dropout=0.3
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    logger.info("✓ Model loaded successfully")
    return model


def load_dataset(data_dir: str, seed: int = 42):
    """Load full dataset (no decimation, matching CNN configuration)."""
    logger.info(f"Loading dataset from {data_dir}")
    
    # No decimation - same as CNN baseline for fair comparison
    decim_dict = {}
    
    loader = DASDataLoader(
        data_dir=data_dir,
        sample_len=2048,
        transform=fft,
        fsize=8192,
        shift=2048,
        decimate=decim_dict,
        drop_noise=True,
    )
    
    x, y_onehot = loader.parse_dataset()
    
    # Normalize
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True) + 1e-8
    x_normalized = (x - mean) / std
    
    # Convert one-hot to indices
    y = y_onehot.argmax(axis=1)
    class_names = list(loader.encoder.classes_)
    
    logger.info(f"✓ Loaded {len(x)} samples, {len(class_names)} classes")
    logger.info(f"Class distribution:")
    for i, cls in enumerate(class_names):
        count = np.sum(y == i)
        logger.info(f"  {cls:15} {count:6}")
    
    return x_normalized, y, class_names


def create_balanced_splits(x, y, seed=42, train_ratio=0.8, val_ratio=0.1):
    """Create balanced train/val/test splits (80/10/10 - matching CNN)."""
    np.random.seed(seed)
    
    classes = np.unique(y)
    x_train, x_val, x_test = [], [], []
    y_train, y_val, y_test = [], [], []
    
    for cls in classes:
        mask = y == cls
        x_cls = x[mask]
        y_cls = y[mask]
        
        n = len(x_cls)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        indices = np.random.permutation(n)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        x_train.append(x_cls[train_idx])
        x_val.append(x_cls[val_idx])
        x_test.append(x_cls[test_idx])
        y_train.append(y_cls[train_idx])
        y_val.append(y_cls[val_idx])
        y_test.append(y_cls[test_idx])
    
    return (np.vstack(x_train), np.vstack(x_val), np.vstack(x_test),
            np.concatenate(y_train), np.concatenate(y_val), np.concatenate(y_test))


@torch.no_grad()
def evaluate_model(model, x_test, y_test, device, batch_size=256):
    """Evaluate model on test set."""
    model.eval()
    
    n_samples = len(x_test)
    all_predictions = []
    
    logger.info(f"Evaluating on {n_samples} test samples...")
    
    for i in range(0, n_samples, batch_size):
        batch_x = x_test[i:i+batch_size]
        batch_x_tensor = torch.FloatTensor(batch_x).to(device)
        
        outputs = model(batch_x_tensor)
        predictions = outputs.argmax(dim=1).cpu().numpy()
        all_predictions.append(predictions)
    
    predictions = np.concatenate(all_predictions)
    
    # Compute metrics
    accuracy = accuracy_score(y_test, predictions)
    balanced_acc = balanced_accuracy_score(y_test, predictions)
    f1_macro = f1_score(y_test, predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, predictions, average='weighted', zero_division=0)
    precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
    recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
    
    cm = confusion_matrix(y_test, predictions)
    
    return {
        'predictions': predictions,
        'targets': y_test,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm
    }


def main():
    parser = argparse.ArgumentParser(description='Test MLP on Balanced Dataset')
    parser.add_argument('--checkpoint', type=str,
                        default='mlp_results_20260128_070258/mlp_best_classifier_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str,
                        default='/nobackup/carda/datasets/DAS-dataset/data',
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str,
                        default='mlp_no_decimation_results',
                        help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Load dataset
    x, y, class_names = load_dataset(args.data_dir, args.seed)
    
    # Create splits
    logger.info("Creating balanced train/val/test splits...")
    x_train, x_val, x_test, y_train, y_val, y_test = create_balanced_splits(
        x, y, seed=args.seed
    )
    
    logger.info(f"Split sizes:")
    logger.info(f"  Train: {len(x_train)}")
    logger.info(f"  Val:   {len(x_val)}")
    logger.info(f"  Test:  {len(x_test)}")
    
    # Load model
    num_classes = len(class_names)
    model = load_model(args.checkpoint, num_classes, device)
    
    # Evaluate
    logger.info("\n" + "="*70)
    logger.info("EVALUATING MLP ON FULL TEST SET (No Decimation, 80/10/10 Split)")
    logger.info("="*70)
    
    results = evaluate_model(model, x_test, y_test, device)
    
    # Print results
    logger.info("\n" + "="*70)
    logger.info("RESULTS")
    logger.info("="*70)
    logger.info(f"Test samples:      {len(x_test)}")
    logger.info(f"Accuracy:          {results['accuracy']:.4f} ({100*results['accuracy']:.2f}%)")
    logger.info(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}")
    logger.info(f"F1 Macro:          {results['f1_macro']:.4f}")
    logger.info(f"F1 Weighted:       {results['f1_weighted']:.4f}")
    logger.info(f"Precision:         {results['precision']:.4f}")
    logger.info(f"Recall:            {results['recall']:.4f}")
    
    # Per-class metrics
    logger.info("\n" + "="*70)
    logger.info("PER-CLASS METRICS")
    logger.info("="*70)
    report = classification_report(y_test, results['predictions'],
                                   target_names=class_names, zero_division=0)
    logger.info("\n" + report)
    
    # Save results
    results['class_names'] = class_names
    np.save(f'{args.output_dir}/test_results.npy', results)
    logger.info(f"\n✓ Results saved to {args.output_dir}/test_results.npy")


if __name__ == '__main__':
    main()
