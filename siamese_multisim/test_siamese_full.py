#!/usr/bin/env python3
"""
Test Siamese Network on FULL Test Set

Instead of episodic evaluation, this script:
1. Uses K support samples per class to build prototypes (from training set)
2. Classifies ALL test samples using these prototypes
3. Tests multiple configurations: 5-way and 9-way, with K=[1,5,10,15,20]

This allows fair comparison with CNN/MLP traditional evaluation.

Author: Andrei Ribeiro
Date: January 2026
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, classification_report, confusion_matrix
)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader import DASDataLoader, fft
from models import MultiSimilaritySiameseNetwork

# Dataset configuration
ALL_CLASSES = ['car', 'construction', 'fence', 'longboard', 'manipulation', 
               'openclose', 'regular', 'running', 'walk']
TRAIN_CLASSES = ['regular', 'walk', 'car', 'manipulation', 'openclose']

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, device: torch.device) -> MultiSimilaritySiameseNetwork:
    """Load trained Siamese model from checkpoint."""
    logger.info(f"Loading model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check if it's a wrapped checkpoint or direct state_dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        model_args = checkpoint.get('args', {})
        embedding_dim = model_args.get('embedding_dim', 128)
    else:
        # Direct state_dict
        state_dict = checkpoint
        embedding_dim = 128  # Default
    
    # Infer input_dim from first conv layer
    # For CNN-based embedding: embedding_net.conv_layers.0.weight has shape (out_channels, in_channels, kernel)
    # For FC-based: embedding_net.fc1.weight has shape (out_dim, in_dim)
    if 'embedding_net.fc1.weight' in state_dict:
        input_dim = state_dict['embedding_net.fc1.weight'].shape[1]
    elif 'embedding_net.conv_layers.0.weight' in state_dict:
        # For CNN, input will be determined by the model itself
        input_dim = 2048  # Standard DAS input
    else:
        input_dim = 2048  # Default
    
    # Create model
    model = MultiSimilaritySiameseNetwork(
        input_dim=input_dim,
        embedding_dim=embedding_dim
    ).to(device)
    
    # Load weights
    model.load_state_dict(state_dict)
    model.eval()
    
    logger.info(f"✓ Model loaded (input_dim={input_dim}, embedding_dim={embedding_dim})")
    return model


@torch.no_grad()
def build_prototypes(
    model: torch.nn.Module,
    x_support: np.ndarray,
    y_support: np.ndarray,
    classes: np.ndarray,
    device: torch.device
) -> torch.Tensor:
    """
    Build class prototypes from support set.
    
    Args:
        model: Trained Siamese network
        x_support: Support features (N_support, feature_dim)
        y_support: Support labels (N_support,)
        classes: Class IDs to build prototypes for
        device: Torch device
        
    Returns:
        prototypes: Class prototypes (n_classes, embedding_dim)
    """
    model.eval()
    prototypes = []
    
    for cls in classes:
        # Get samples of this class
        cls_mask = y_support == cls
        cls_samples = x_support[cls_mask]
        
        if len(cls_samples) == 0:
            raise ValueError(f"No samples for class {cls}")
        
        # Embed samples
        cls_tensor = torch.FloatTensor(cls_samples).to(device)
        cls_embeddings = model.forward_one(cls_tensor)
        
        # Compute prototype (mean embedding)
        prototype = cls_embeddings.mean(dim=0)
        prototypes.append(prototype)
    
    prototypes = torch.stack(prototypes, dim=0)  # (n_classes, embedding_dim)
    return prototypes


@torch.no_grad()
def classify_with_prototypes(
    model: torch.nn.Module,
    x_query: np.ndarray,
    prototypes: torch.Tensor,
    device: torch.device,
    batch_size: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classify query samples using prototypes via cosine similarity.
    
    Args:
        model: Trained Siamese network
        x_query: Query features (N_query, feature_dim)
        prototypes: Class prototypes (n_classes, embedding_dim)
        device: Torch device
        batch_size: Batch size for processing
        
    Returns:
        predictions: Predicted class indices (N_query,)
        distances: Distances to nearest prototype (N_query,)
    """
    model.eval()
    
    n_samples = len(x_query)
    all_predictions = []
    all_distances = []
    
    # Process in batches
    for i in range(0, n_samples, batch_size):
        batch = x_query[i:i + batch_size]
        batch_tensor = torch.FloatTensor(batch).to(device)
        
        # Embed queries
        query_embeddings = model.forward_one(batch_tensor)  # (batch, embedding_dim)
        
        # Compute cosine similarity to all prototypes
        similarities = F.cosine_similarity(
            query_embeddings.unsqueeze(1),  # (batch, 1, embedding_dim)
            prototypes.unsqueeze(0),         # (1, n_classes, embedding_dim)
            dim=2
        )
        
        # Predict class with highest similarity
        max_similarities, max_indices = similarities.max(dim=1)
        
        # Convert to distances (1 - similarity for normalized embeddings)
        distances = 1 - max_similarities
        
        all_predictions.append(max_indices.cpu().numpy())
        all_distances.append(distances.cpu().numpy())
    
    predictions = np.concatenate(all_predictions)
    distances = np.concatenate(all_distances)
    
    return predictions, distances


def evaluate_full_test(
    model: torch.nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    n_way: int,
    k_shot: int,
    device: torch.device,
    class_names: list = None,
    seed: int = 42
) -> Dict:
    """
    Evaluate model on full test set using K-shot prototypes.
    
    Args:
        model: Trained Siamese network
        x_train: Training features (for support set)
        y_train: Training labels
        x_test: Test features (all will be classified)
        y_test: Test labels
        n_way: Number of classes to use
        k_shot: Number of support samples per class
        device: Torch device
        class_names: List of class names
        seed: Random seed for class/sample selection
        
    Returns:
        Dictionary with evaluation results
    """
    np.random.seed(seed)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"EVALUATING {n_way}-way {k_shot}-shot on FULL TEST SET")
    logger.info(f"{'='*70}")
    
    all_classes = np.unique(y_train)
    n_total_classes = len(all_classes)
    
    # Select n_way classes
    if n_way < n_total_classes:
        selected_classes = np.random.choice(all_classes, n_way, replace=False)
    else:
        selected_classes = all_classes
    
    logger.info(f"Selected classes: {selected_classes}")
    
    # Build support set (K samples per class from training set)
    x_support_list = []
    y_support_list = []
    
    for cls in selected_classes:
        cls_mask = y_train == cls
        cls_samples = x_train[cls_mask]
        
        if len(cls_samples) < k_shot:
            logger.warning(f"Class {cls} has only {len(cls_samples)} samples, using all")
            n_samples = len(cls_samples)
        else:
            n_samples = k_shot
        
        # Sample K shots
        indices = np.random.choice(len(cls_samples), n_samples, replace=False)
        x_support_list.append(cls_samples[indices])
        y_support_list.extend([cls] * n_samples)
    
    x_support = np.vstack(x_support_list)
    y_support = np.array(y_support_list)
    
    # Filter test set to only include selected classes
    test_mask = np.isin(y_test, selected_classes)
    x_test_filtered = x_test[test_mask]
    y_test_filtered = y_test[test_mask]
    
    logger.info(f"Support Set: {len(x_support)} samples ({k_shot} per class × {n_way} classes)")
    logger.info(f"Test Set: {len(x_test_filtered)} samples (filtered to {n_way} classes)")
    
    # Build prototypes
    prototypes = build_prototypes(model, x_support, y_support, selected_classes, device)
    logger.info(f"✓ Built {len(selected_classes)} prototypes")
    
    # Classify all test samples
    logger.info(f"Classifying {len(x_test_filtered)} test samples...")
    pred_indices, distances = classify_with_prototypes(
        model, x_test_filtered, prototypes, device
    )
    
    # Map indices back to class IDs
    predictions = selected_classes[pred_indices]
    
    # Compute metrics
    accuracy = accuracy_score(y_test_filtered, predictions)
    balanced_acc = balanced_accuracy_score(y_test_filtered, predictions)
    f1_macro = f1_score(y_test_filtered, predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test_filtered, predictions, average='weighted', zero_division=0)
    precision = precision_score(y_test_filtered, predictions, average='weighted', zero_division=0)
    recall = recall_score(y_test_filtered, predictions, average='weighted', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test_filtered, predictions, labels=selected_classes)
    
    # Results
    results = {
        'n_way': n_way,
        'k_shot': k_shot,
        'n_support': len(x_support),
        'n_test': len(x_test_filtered),
        'selected_classes': selected_classes,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision': precision,
        'recall': recall,
        'predictions': predictions,
        'targets': y_test_filtered,
        'distances': distances,
        'confusion_matrix': cm,
    }
    
    # Print results
    logger.info(f"\n{'='*70}")
    logger.info("RESULTS")
    logger.info(f"{'='*70}")
    logger.info(f"Accuracy:          {accuracy:.4f} ({100*accuracy:.2f}%)")
    logger.info(f"Balanced Accuracy: {balanced_acc:.4f}")
    logger.info(f"F1 Macro:          {f1_macro:.4f}")
    logger.info(f"F1 Weighted:       {f1_weighted:.4f}")
    logger.info(f"Precision:         {precision:.4f}")
    logger.info(f"Recall:            {recall:.4f}")
    
    if class_names is not None:
        selected_class_names = [class_names[cls] for cls in selected_classes]
        logger.info("\nPer-class Report:")
        logger.info("\n" + classification_report(
            y_test_filtered, predictions,
            labels=selected_classes,
            target_names=selected_class_names,
            zero_division=0
        ))
    
    return results


def load_dataset(data_dir: str = '/mnt/hdd/andrei/DAS-Dataset/Dataset3-2018-all') -> Tuple[np.ndarray, np.ndarray, list]:
    """Load DAS dataset."""
    logger.info(f"Loading dataset from {data_dir}")
    
    # Use NO decimation to match original CNN evaluation
    # This gives ~32,447 test samples (same as original CNN test set)
    # Regular class has full ~21K samples
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
    
    # Z-score normalization
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True) + 1e-8
    x_normalized = (x - mean) / std
    
    # Convert one-hot to indices
    y = y_onehot.argmax(axis=1)
    class_names = list(loader.encoder.classes_)
    
    logger.info(f"✓ Loaded {len(x)} samples, {len(class_names)} classes")
    
    return x_normalized, y, class_names


def create_balanced_splits(x, y, seed=42, train_ratio=0.8, val_ratio=0.1):
    """Create balanced train/val/test splits.
    
    Using 80/10/10 split to match CNN training configuration.
    """
    np.random.seed(seed)
    
    classes = np.unique(y)
    x_train, x_val, x_test = [], [], []
    y_train, y_val, y_test = [], [], []
    
    for cls in classes:
        cls_mask = y == cls
        cls_x = x[cls_mask]
        cls_y = y[cls_mask]
        
        n = len(cls_x)
        indices = np.random.permutation(n)
        
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        x_train.append(cls_x[train_idx])
        x_val.append(cls_x[val_idx])
        x_test.append(cls_x[test_idx])
        
        y_train.append(cls_y[train_idx])
        y_val.append(cls_y[val_idx])
        y_test.append(cls_y[test_idx])
    
    return (np.vstack(x_train), np.vstack(x_val), np.vstack(x_test),
            np.concatenate(y_train), np.concatenate(y_val), np.concatenate(y_test))


def evaluate_anomaly_detection(
    model: torch.nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    k_shot: int,
    device: torch.device,
    regular_class_idx: int = 6,
    seed: int = 42
) -> Dict:
    """
    Evaluate anomaly detection (regular vs all other classes).
    
    Args:
        model: Trained Siamese network
        x_train: Training features
        y_train: Training labels
        x_test: Test features
        y_test: Test labels
        k_shot: Number of support samples per class
        device: Torch device
        regular_class_idx: Index of "regular" class (normal)
        seed: Random seed
        
    Returns:
        Dictionary with anomaly detection metrics
    """
    np.random.seed(seed)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"ANOMALY DETECTION ({k_shot}-shot): Regular vs Others")
    logger.info(f"{'='*70}")
    
    all_classes = np.unique(y_train)
    
    # Build support set (K samples per class)
    x_support_list = []
    y_support_list = []
    
    for cls in all_classes:
        cls_mask = y_train == cls
        cls_samples = x_train[cls_mask]
        
        if len(cls_samples) < k_shot:
            n_samples = len(cls_samples)
        else:
            n_samples = k_shot
        
        indices = np.random.choice(len(cls_samples), n_samples, replace=False)
        x_support_list.append(cls_samples[indices])
        y_support_list.extend([cls] * n_samples)
    
    x_support = np.vstack(x_support_list)
    y_support = np.array(y_support_list)
    
    logger.info(f"Support Set: {len(x_support)} samples ({k_shot} per class × {len(all_classes)} classes)")
    logger.info(f"Test Set: {len(x_test)} samples")
    
    # Build prototypes
    prototypes = build_prototypes(model, x_support, y_support, all_classes, device)
    
    # Classify all test samples
    pred_indices, distances = classify_with_prototypes(model, x_test, prototypes, device)
    predictions = all_classes[pred_indices]
    
    # Convert to binary: 0 = regular (normal), 1 = anomaly (any other class)
    y_binary_true = (y_test != regular_class_idx).astype(int)
    y_binary_pred = (predictions != regular_class_idx).astype(int)
    
    # Compute metrics
    accuracy = accuracy_score(y_binary_true, y_binary_pred)
    balanced_acc = balanced_accuracy_score(y_binary_true, y_binary_pred)
    f1 = f1_score(y_binary_true, y_binary_pred, average='binary', zero_division=0)
    precision = precision_score(y_binary_true, y_binary_pred, zero_division=0)
    recall = recall_score(y_binary_true, y_binary_pred, zero_division=0)
    
    # Confusion matrix for anomaly detection
    cm = confusion_matrix(y_binary_true, y_binary_pred)
    
    results = {
        'k_shot': k_shot,
        'n_test': len(x_test),
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm,
        'predictions': y_binary_pred,
        'targets': y_binary_true,
    }
    
    logger.info(f"\n{'='*70}")
    logger.info("ANOMALY DETECTION RESULTS")
    logger.info(f"{'='*70}")
    logger.info(f"Accuracy:          {accuracy:.4f} ({100*accuracy:.2f}%)")
    logger.info(f"Balanced Accuracy: {balanced_acc:.4f}")
    logger.info(f"F1 Score:          {f1:.4f}")
    logger.info(f"Precision:         {precision:.4f}")
    logger.info(f"Recall:            {recall:.4f}")
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    logger.info(f"  FN={cm[1,0]}, TP={cm[1,1]}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Test Siamese on Full Test Set')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (relative to project root)')
    parser.add_argument('--data_dir', type=str, 
                        default='/nobackup/carda/datasets/DAS-dataset/data',
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='../paper_figures',
                        help='Output directory for results (relative to this script)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    logger.info("Loading data...")
    x_full, y_full, class_names = load_dataset(args.data_dir)
    logger.info(f"Classes: {class_names}")
    
    # Split data (same as training)
    x_train, x_val, x_test, y_train, y_val, y_test = create_balanced_splits(
        x_full, y_full, seed=args.seed
    )
    
    logger.info(f"\nData splits:")
    logger.info(f"  Train: {len(x_train)} samples")
    logger.info(f"  Val:   {len(x_val)} samples")
    logger.info(f"  Test:  {len(x_test)} samples")
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Test configurations
    n_ways = [5, 9]
    k_shots = [1, 5, 10, 15, 20]
    
    all_results = {}
    
    logger.info(f"\n{'='*70}")
    logger.info("STARTING COMPREHENSIVE EVALUATION")
    logger.info(f"Configurations: {len(n_ways)} n-way × {len(k_shots)} k-shot = {len(n_ways)*len(k_shots)} total")
    logger.info(f"{'='*70}")
    
    # Run all configurations
    for n_way in n_ways:
        for k_shot in k_shots:
            config_name = f"{n_way}way_{k_shot}shot"
            
            try:
                results = evaluate_full_test(
                    model=model,
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    y_test=y_test,
                    n_way=n_way,
                    k_shot=k_shot,
                    device=device,
                    class_names=class_names,
                    seed=args.seed
                )
                
                all_results[config_name] = results
                
            except Exception as e:
                logger.error(f"Error in {config_name}: {e}")
                continue
    
    # Anomaly Detection Evaluation (separate from N-way K-shot)
    logger.info(f"\n{'='*70}")
    logger.info("ANOMALY DETECTION EVALUATION")
    logger.info(f"Testing on full test set for multiple K-shot configurations")
    logger.info(f"{'='*70}")
    
    anomaly_results = {}
    regular_class_idx = class_names.index('regular') if 'regular' in class_names else 6
    
    for k_shot in k_shots:
        try:
            results = evaluate_anomaly_detection(
                model=model,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                k_shot=k_shot,
                device=device,
                regular_class_idx=regular_class_idx,
                seed=args.seed
            )
            
            anomaly_results[f'anomaly_{k_shot}shot'] = results
            
        except Exception as e:
            logger.error(f"Error in anomaly detection {k_shot}-shot: {e}")
            continue
    
    # Combine all results
    all_results['anomaly_detection'] = anomaly_results
    
    # Save all results
    output_file = output_dir / 'siamese_full_test_results.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(all_results, f)
    logger.info(f"\n✓ Saved all results: {output_file}")
    
    # Print summary
    logger.info(f"\n{'='*70}")
    logger.info("SUMMARY OF ALL CONFIGURATIONS")
    logger.info(f"{'='*70}")
    logger.info(f"{'Configuration':<20} {'Accuracy':<12} {'Bal. Acc':<12} {'F1 Macro':<12} {'# Test':<10}")
    logger.info("-" * 70)
    
    for config_name, results in sorted(all_results.items()):
        if config_name == 'anomaly_detection':
            continue
        logger.info(
            f"{config_name:<20} "
            f"{results['accuracy']:<12.4f} "
            f"{results['balanced_accuracy']:<12.4f} "
            f"{results['f1_macro']:<12.4f} "
            f"{results['n_test']:<10}"
        )
    
    # Anomaly detection summary
    if 'anomaly_detection' in all_results:
        logger.info(f"\n{'='*70}")
        logger.info("ANOMALY DETECTION SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"{'Configuration':<20} {'Accuracy':<12} {'Bal. Acc':<12} {'F1 Score':<12} {'Precision':<12} {'Recall':<12}")
        logger.info("-" * 90)
        
        for config_name, results in sorted(all_results['anomaly_detection'].items()):
            logger.info(
                f"{config_name:<20} "
                f"{results['accuracy']:<12.4f} "
                f"{results['balanced_accuracy']:<12.4f} "
                f"{results['f1']:<12.4f} "
                f"{results['precision']:<12.4f} "
                f"{results['recall']:<12.4f}"
            )
    
    logger.info(f"\n{'='*70}")
    logger.info("EVALUATION COMPLETE")
    logger.info(f"{'='*70}")


if __name__ == '__main__':
    main()
