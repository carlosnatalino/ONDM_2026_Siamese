#!/usr/bin/env python3
# train_siamese.py
"""
Multi-Similarity Siamese Network for DAS Event Classification

Main training and evaluation script for the multi-similarity Siamese network
designed for novelty detection and N-way K-shot classification on DAS
(Distributed Acoustic Sensing) data from Φ-OTDR systems.

This implementation is designed for scientific publication and includes:
1. Multi-Similarity Comparison: L1, L2, Cosine, Product, Learned Attention
2. Episodic Training: Class-balanced pair sampling
3. Comprehensive Evaluation: 5-way and 9-way with 1, 5, 10-shot
4. Novelty Detection: Fixed threshold and statistical approaches
5. Real-World Simulation: Incremental class deployment

Training Classes (5 total):
- "regular": Normal baseline conditions
- "walk", "car", "manipulation", "openclose": Anomaly classes

Testing: All 9 classes

Dataset Reference:
- Tomasov et al., "Enhancing Perimeter Protection using Φ-OTDR and CNN 
  for Event Classification", Optical Fiber Sensors (OFS) 2023

Architecture References:
- Bromley et al., "Signature Verification using a Siamese Time Delay NN" (1993)
- Koch et al., "Siamese Neural Networks for One-shot Image Recognition" (2015)
- Snell et al., "Prototypical Networks for Few-shot Learning" (2017)

Author: Andrei Ribeiro, Carlos Natalino
Date: January 2026
"""

import os
import sys
import argparse
import logging
import datetime
import pickle
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from siamese_multisim.models import create_siamese_network, MultiSimilaritySiameseNetwork
from siamese_multisim.training import (
    SignalAugmentation, EpisodicPairSampler, SiamesePairDataset,
    MultiSimilarityBCELoss, EpisodicTrainer
)
from siamese_multisim.evaluation import (
    run_comprehensive_nway_kshot_eval, NoveltyDetector, RealWorldSimulator,
    run_full_evaluation
)
from siamese_multisim.visualization import generate_all_plots, plot_training_curves

# =============================================================================
# Logging Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =============================================================================
# Data Loading
# =============================================================================

# Training classes: regular (normal) + 4 anomalies
TRAIN_CLASSES = ['regular', 'walk', 'car', 'manipulation', 'openclose']

# All classes for testing
ALL_CLASSES = ['regular', 'fence', 'longboard', 'manipulation', 
               'openclose', 'running', 'walk', 'car', 'construction']


def load_dataset(
    data_dir: str,
    decim_dict: Optional[Dict[str, int]] = None,
    debug: bool = False,
    train_only: bool = False
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load DAS dataset using DASDataLoader.
    
    Args:
        data_dir: Path to dataset directory
        decim_dict: Decimation factors per class
        debug: If True, use high decimation for testing
        train_only: If True, only load training classes
        
    Returns:
        x: Features (N, 2048)
        y: Labels (N,)
        class_names: List of class names
    """
    if debug and decim_dict is None:
        # High decimation for quick debugging
        decim_dict = {
            'regular': 100,
            'construction': 50,
            'car': 50,
            'fence': 30,
            'walk': 50,
            'manipulation': 50,
            'longboard': 50,
            'running': 50,
            'openclose': 30
        }
        logger.info(f"Debug mode: Using decimation {decim_dict}")
    elif decim_dict is None:
        # Default: uniform decimation factor of 10
        decim_dict = {cls: 10 for cls in ALL_CLASSES}
    
    logger.info(f"Loading dataset from {data_dir}")
    
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
    
    logger.info(f"Dataset loaded: {len(x)} samples, {len(class_names)} classes")
    logger.info(f"Class distribution: {dict(Counter(loader.str_labels))}")
    
    # Filter to training classes if requested
    if train_only:
        train_indices = [class_names.index(c) for c in TRAIN_CLASSES if c in class_names]
        mask = np.isin(y, train_indices)
        x_normalized = x_normalized[mask]
        y = y[mask]
        
        # Re-map labels to 0, 1, 2, ...
        label_map = {old: new for new, old in enumerate(sorted(np.unique(y)))}
        y = np.array([label_map[l] for l in y])
        class_names = [class_names[i] for i in sorted(train_indices)]
        
        logger.info(f"Filtered to training classes: {class_names}")
        logger.info(f"Filtered dataset: {len(x_normalized)} samples")
    
    return x_normalized, y, class_names


def create_balanced_splits(
    x: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42
) -> Tuple[np.ndarray, ...]:
    """Create class-balanced train/val/test splits."""
    np.random.seed(seed)
    
    classes = np.unique(y)
    class_indices = {c: np.where(y == c)[0] for c in classes}
    
    # Find minimum class size
    min_class_size = min(len(idx) for idx in class_indices.values())
    samples_per_class = min_class_size
    
    logger.info(f"Balanced sampling: {samples_per_class} samples per class")
    
    # Split sizes
    n_test = max(1, int(samples_per_class * test_size))
    n_val = max(1, int(samples_per_class * val_size))
    n_train = samples_per_class - n_test - n_val
    
    logger.info(f"Per class: train={n_train}, val={n_val}, test={n_test}")
    
    train_idx, val_idx, test_idx = [], [], []
    
    for c in classes:
        indices = class_indices[c].copy()
        np.random.shuffle(indices)
        indices = indices[:samples_per_class]
        
        test_idx.extend(indices[:n_test])
        val_idx.extend(indices[n_test:n_test + n_val])
        train_idx.extend(indices[n_test + n_val:])
    
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)
    
    return (
        x[train_idx], x[val_idx], x[test_idx],
        y[train_idx], y[val_idx], y[test_idx]
    )


# =============================================================================
# Training Loop
# =============================================================================

def train_model(
    trainer: EpisodicTrainer,
    train_dataset: SiamesePairDataset,
    val_dataset: SiamesePairDataset,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    early_stopping: int = 30,
    grad_clip: float = 1.0,
    output_dir: str = '.'
) -> Dict:
    """
    Full training loop with episodic sampling.
    
    Args:
        trainer: EpisodicTrainer instance
        train_dataset: Training dataset
        val_dataset: Validation dataset
        epochs: Number of epochs
        batch_size: Pairs per batch
        lr: Learning rate
        weight_decay: Weight decay
        early_stopping: Patience for early stopping
        grad_clip: Gradient clipping value
        output_dir: Output directory
        
    Returns:
        Training history
    """
    # Create sampler
    train_sampler = EpisodicPairSampler(
        labels=train_dataset.y,
        batch_size=batch_size,
        positive_ratio=0.5,
        n_batches=max(100, len(train_dataset) // batch_size)
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        trainer.model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,
        T_mult=2,
        eta_min=1e-6
    )
    
    # Loss function
    criterion = MultiSimilarityBCELoss(
        margin=0.5,
        alpha=2.0,
        beta=50.0,
        lambda_ms=0.1
    )
    
    logger.info("=" * 70)
    logger.info("Starting Multi-Similarity Siamese Training")
    logger.info("=" * 70)
    
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        # Train epoch
        train_metrics = trainer.train_epoch(
            dataset=train_dataset,
            sampler=train_sampler,
            optimizer=optimizer,
            criterion=criterion,
            grad_clip=grad_clip
        )
        
        # Validate
        val_pair_metrics = trainer.evaluate_pairs(val_dataset, n_pairs=500)
        
        # Classification evaluation
        val_class_results = trainer.evaluate_classification(
            train_dataset.x, train_dataset.y,
            val_dataset.x, val_dataset.y,
            [str(c) for c in train_dataset.classes]
        )
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        trainer.history['train_loss'].append(train_metrics['loss'])
        trainer.history['train_pair_acc'].append(train_metrics['pair_acc'])
        trainer.history['train_f1'].append(train_metrics['pair_f1'])
        trainer.history['val_loss'].append(val_pair_metrics['loss'])
        trainer.history['val_pair_acc'].append(val_pair_metrics['pair_acc'])
        trainer.history['val_f1'].append(val_pair_metrics['pair_f1'])
        trainer.history['val_class_acc'].append(val_class_results['balanced_accuracy'])
        trainer.history['val_anomaly_f1'].append(val_class_results.get('anomaly_f1', 0))
        trainer.history['lr'].append(current_lr)
        
        # Similarity metrics
        trainer.history['pos_l1_mean'].append(train_metrics['pos_l1_mean'])
        trainer.history['neg_l1_mean'].append(train_metrics['neg_l1_mean'])
        trainer.history['pos_l2_mean'].append(train_metrics['pos_l2_mean'])
        trainer.history['neg_l2_mean'].append(train_metrics['neg_l2_mean'])
        trainer.history['pos_cosine_mean'].append(train_metrics['pos_cosine_mean'])
        trainer.history['neg_cosine_mean'].append(train_metrics['neg_cosine_mean'])
        trainer.history['attention_weights'].append(train_metrics.get('attention_weights'))
        
        # Logging
        logger.info(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train F1: {train_metrics['pair_f1']:.4f} | "
            f"Val Pair Acc: {val_pair_metrics['pair_acc']:.4f} | "
            f"Val Class Acc: {val_class_results['balanced_accuracy']:.4f} | "
            f"LR: {current_lr:.2e}"
        )
        
        # Similarity statistics
        cos_gap = train_metrics['pos_cosine_mean'] - train_metrics['neg_cosine_mean']
        l2_gap = train_metrics['neg_l2_mean'] - train_metrics['pos_l2_mean']
        logger.info(
            f"  Metrics: Pos Cos: {train_metrics['pos_cosine_mean']:.3f} | "
            f"Neg Cos: {train_metrics['neg_cosine_mean']:.3f} | "
            f"Cos Gap: {cos_gap:.3f} | L2 Gap: {l2_gap:.3f}"
        )
        
        # Save best model
        val_score = val_class_results['balanced_accuracy']
        if val_score > trainer.best_val_acc:
            trainer.best_val_acc = val_score
            trainer.best_val_f1 = val_class_results.get('macro_f1', 0)
            trainer.best_model_state = {
                k: v.cpu().clone() for k, v in trainer.model.state_dict().items()
            }
            patience_counter = 0
            
            torch.save(
                trainer.best_model_state,
                os.path.join(output_dir, 'best_model.pth')
            )
            logger.info(f"  -> New best model saved!")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stopping:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    # Restore best model
    if trainer.best_model_state is not None:
        trainer.model.load_state_dict(trainer.best_model_state)
        logger.info(f"Restored best model (val_acc={trainer.best_val_acc:.4f})")
    
    return trainer.history


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(
    results: Dict,
    output_dir: str,
    args: argparse.Namespace
):
    """Generate comprehensive text report."""
    report_path = os.path.join(output_dir, 'report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Multi-Similarity Siamese Network - Experiment Report\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Configuration\n")
        f.write("-" * 40 + "\n")
        for key, value in vars(args).items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        f.write("Training Results\n")
        f.write("-" * 40 + "\n")
        if 'history' in results:
            h = results['history']
            f.write(f"  Final Train Loss: {h['train_loss'][-1]:.4f}\n")
            f.write(f"  Best Val Pair Acc: {max(h['val_pair_acc']):.4f}\n")
            f.write(f"  Best Val Class Acc: {max(h['val_class_acc']):.4f}\n")
            if h['val_anomaly_f1']:
                f.write(f"  Best Anomaly F1: {max(h['val_anomaly_f1']):.4f}\n")
        f.write("\n")
        
        f.write("Test Results\n")
        f.write("-" * 40 + "\n")
        if 'test_results' in results:
            t = results['test_results']
            f.write(f"  Accuracy: {t['accuracy']:.4f}\n")
            f.write(f"  Balanced Accuracy: {t['balanced_accuracy']:.4f}\n")
            f.write(f"  Macro F1: {t['f1_macro']:.4f}\n")
            f.write(f"  Weighted F1: {t['f1_weighted']:.4f}\n")
            f.write(f"  Anomaly F1: {t['anomaly_f1']:.4f}\n")
            f.write("\nClassification Report:\n")
            f.write(t['report'])
        f.write("\n")
        
        f.write("N-way K-shot Results\n")
        f.write("-" * 40 + "\n")
        if 'nway_results' in results:
            for config, res in results['nway_results'].items():
                f.write(f"  {config}:\n")
                f.write(f"    Accuracy: {res['accuracy']:.4f} ± {res.get('accuracy_std', 0):.4f}\n")
                f.write(f"    F1 Macro: {res['f1_macro']:.4f}\n")
        f.write("\n")
        
        f.write("Novelty Detection Results\n")
        f.write("-" * 40 + "\n")
        if 'novelty_results' in results:
            n = results['novelty_results']
            f.write(f"  Fixed Threshold:\n")
            f.write(f"    F1: {n.get('fixed_f1', 0):.4f}\n")
            f.write(f"    True Novel Rate: {n.get('fixed_true_novel_rate', 0):.4f}\n")
            f.write(f"  Statistical Threshold:\n")
            f.write(f"    F1: {n.get('statistical_f1', 0):.4f}\n")
            f.write(f"    True Novel Rate: {n.get('statistical_true_novel_rate', 0):.4f}\n")
        f.write("\n")
        
        f.write("Real-World Simulation Results\n")
        f.write("-" * 40 + "\n")
        if 'simulation_results' in results:
            for k, steps in results['simulation_results'].items():
                if steps:
                    final = steps[-1]
                    f.write(f"  {k}-shot (final step):\n")
                    f.write(f"    Anomaly F1: {final['anomaly_f1']:.4f}\n")
                    f.write(f"    Known Acc: {final['known_acc']:.4f}\n")
                    f.write(f"    Unknown Det (Fixed): {final['unknown_det_fixed']:.4f}\n")
                    f.write(f"    Unknown Det (Stat): {final['unknown_det_stat']:.4f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("End of Report\n")
    
    logger.info(f"Report saved to {report_path}")


# =============================================================================
# Main Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Multi-Similarity Siamese Network for DAS Classification'
    )
    
    # Paths
    parser.add_argument('--data_dir', type=str,
                       default='/nobackup/carda/datasets/DAS-dataset/data',
                       help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='siamese_multisim',
                       help='Output directory prefix')
    
    # Model
    parser.add_argument('--embedding_dim', type=int, default=128,
                       help='Embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--use_mlp', action='store_true',
                       help='Use MLP instead of CNN for embedding')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size (pairs)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--early_stopping', type=int, default=30,
                       help='Early stopping patience')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping')
    
    # Data
    parser.add_argument('--augment', action='store_true', default=True,
                       help='Use data augmentation')
    parser.add_argument('--train_all_classes', action='store_true',
                       help='Train on all classes (default: 5 classes)')
    
    # Debug
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode with reduced dataset')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create timestamped output directory
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.debug:
        output_dir = f"{args.output_dir}_debug"
    else:
        output_dir = f"{args.output_dir}_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup file logging
    file_handler = logging.FileHandler(os.path.join(output_dir, 'training.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Save arguments
    with open(os.path.join(output_dir, 'args.txt'), 'w') as f:
        for key, value in vars(args).items():
            f.write(f'{key}: {value}\n')
    
    logger.info("=" * 70)
    logger.info("Multi-Similarity Siamese Network")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Arguments: {vars(args)}")
    
    # =========================================================================
    # Load Data
    # =========================================================================
    
    # Load training data (5 classes)
    x_train_full, y_train_full, train_class_names = load_dataset(
        args.data_dir,
        debug=args.debug,
        train_only=not args.train_all_classes
    )
    
    # Load full dataset for testing (all 9 classes)
    x_full, y_full, all_class_names = load_dataset(
        args.data_dir,
        debug=args.debug,
        train_only=False
    )
    
    # Create balanced splits for training
    x_train, x_val, x_test_train, y_train, y_val, y_test_train = create_balanced_splits(
        x_train_full, y_train_full, seed=args.seed
    )
    
    # Create test split from full dataset (for 9-class testing)
    _, _, x_test_full, _, _, y_test_full = create_balanced_splits(
        x_full, y_full, seed=args.seed
    )
    
    logger.info(f"Training classes: {train_class_names}")
    logger.info(f"All classes: {all_class_names}")
    logger.info(f"Train: {len(x_train)}, Val: {len(x_val)}")
    logger.info(f"Test (train classes): {len(x_test_train)}")
    logger.info(f"Test (all classes): {len(x_test_full)}")
    
    # Create datasets
    augmentation = SignalAugmentation() if args.augment else None
    
    train_dataset = SiamesePairDataset(
        x_train, y_train,
        augment=args.augment,
        augmentation=augmentation
    )
    
    val_dataset = SiamesePairDataset(x_val, y_val, augment=False)
    
    # =========================================================================
    # Model Setup
    # =========================================================================
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = create_siamese_network(
        input_dim=x_train.shape[1],
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        use_mlp=args.use_mlp
    )
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")
    
    # =========================================================================
    # Training
    # =========================================================================
    
    # Find regular class index
    regular_idx = train_class_names.index('regular') if 'regular' in train_class_names else 0
    
    trainer = EpisodicTrainer(
        model=model,
        device=device,
        output_dir=output_dir,
        regular_class_idx=regular_idx
    )
    
    history = train_model(
        trainer=trainer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        early_stopping=args.early_stopping,
        grad_clip=args.grad_clip,
        output_dir=output_dir
    )
    
    # =========================================================================
    # Evaluation
    # =========================================================================
    
    logger.info("=" * 70)
    logger.info("Evaluation Phase")
    logger.info("=" * 70)
    
    results = {'history': history}
    
    # Test on training classes (5-class)
    logger.info("\n--- Test on Training Classes (5-class) ---")
    test_results_5class = trainer.evaluate_classification(
        x_train, y_train,
        x_test_train, y_test_train,
        train_class_names
    )
    logger.info(f"5-class Test Accuracy: {test_results_5class['accuracy']:.4f}")
    logger.info(f"5-class Balanced Accuracy: {test_results_5class['balanced_accuracy']:.4f}")
    logger.info(f"5-class Test Samples: {len(test_results_5class['predictions'])}")
    results['test_results_5class'] = test_results_5class
    
    # Test on ALL classes (9-class) with few-shot prototypes - for fair comparison with CNN/MLP
    logger.info("\n--- Test on All Classes (9-class with 10-shot prototypes) ---")
    
    # Split test set: use K samples per class for prototypes, rest for testing
    K_SHOT = 10
    x_prototype_support = []
    y_prototype_support = []
    x_test_query = []
    y_test_query = []
    
    for class_idx in np.unique(y_test_full):
        class_mask = y_test_full == class_idx
        class_samples = x_test_full[class_mask]
        class_labels = y_test_full[class_mask]
        
        # Use first K samples for prototypes
        x_prototype_support.append(class_samples[:K_SHOT])
        y_prototype_support.append(class_labels[:K_SHOT])
        
        # Use remaining samples for testing
        x_test_query.append(class_samples[K_SHOT:])
        y_test_query.append(class_labels[K_SHOT:])
    
    x_prototype_support = np.concatenate(x_prototype_support)
    y_prototype_support = np.concatenate(y_prototype_support)
    x_test_query = np.concatenate(x_test_query)
    y_test_query = np.concatenate(y_test_query)
    
    logger.info(f"Using {K_SHOT} samples per class for prototypes ({len(x_prototype_support)} total)")
    logger.info(f"Testing on remaining samples ({len(x_test_query)} total)")
    
    # Find the correct regular class index in the 9-class space
    regular_idx_9class = all_class_names.index('regular') if 'regular' in all_class_names else None
    logger.info(f"Regular class index in 9-class space: {regular_idx_9class}")
    
    # Evaluate with few-shot prototypes from all 9 classes
    test_results_9class = trainer.evaluate_classification(
        x_prototype_support, y_prototype_support,  # Use K-shot samples to build prototypes
        x_test_query, y_test_query,  # Test on remaining samples
        all_class_names,
        regular_class_idx=regular_idx_9class  # Pass correct index for 9-class space
    )
    
    # Add class_names to the results dictionary for visualization
    test_results_9class['class_names'] = list(all_class_names)
    logger.info(f"9-class Test Accuracy (10-shot): {test_results_9class['accuracy']:.4f}")
    logger.info(f"9-class Balanced Accuracy (10-shot): {test_results_9class['balanced_accuracy']:.4f}")
    logger.info(f"9-class F1 Macro (10-shot): {test_results_9class['f1_macro']:.4f}")
    logger.info(f"9-class F1 Weighted (10-shot): {test_results_9class['f1_weighted']:.4f}")
    logger.info(f"9-class Anomaly Detection F1 (10-shot): {test_results_9class['anomaly_f1']:.4f}")
    logger.info(f"9-class Test Samples: {len(test_results_9class['predictions'])}")
    results['test_results_9class'] = test_results_9class
    results['test_results'] = test_results_9class  # Use 9-class for main comparison
    
    # N-way K-shot on validation set (avoid test contamination)
    logger.info("\n--- N-way K-shot on Validation Set ---")
    k_shots_eval = [1, 5, 10, 15, 20]
    nway_5class = run_comprehensive_nway_kshot_eval(
        model=model,
        x=x_val,
        y=y_val,
        device=device,
        n_ways=[5],
        k_shots=k_shots_eval,
        n_episodes=100
    )
    
    # N-way K-shot on all classes (using validation subset from full data)
    logger.info("\n--- N-way K-shot on All Classes (9-class) ---")
    _, x_val_full, _, _, y_val_full, _ = create_balanced_splits(
        x_full, y_full, seed=args.seed
    )
    nway_9class = run_comprehensive_nway_kshot_eval(
        model=model,
        x=x_val_full,
        y=y_val_full,
        device=device,
        n_ways=[5, 9],
        k_shots=k_shots_eval,
        n_episodes=100
    )
    
    results['nway_results'] = {**nway_5class, **nway_9class}

    # Generate K-shot test results for confusion matrices (9-class)
    logger.info("\n--- Computing K-shot Test Results for Confusion Matrices ---")
    kshot_results = {}
    for k in k_shots_eval:
        if k >= len(x_test_full) // len(np.unique(y_test_full)):
            logger.warning(f"Skipping {k}-shot: not enough samples per class")
            continue
        
        logger.info(f"  Computing {k}-shot classification...")
        x_proto, y_proto, x_query, y_query = [], [], [], []
        for class_idx in np.unique(y_test_full):
            mask = y_test_full == class_idx
            samples = x_test_full[mask]
            labels = y_test_full[mask]
            x_proto.append(samples[:k])
            y_proto.append(labels[:k])
            x_query.append(samples[k:])
            y_query.append(labels[k:])
        
        x_proto = np.concatenate(x_proto)
        y_proto = np.concatenate(y_proto)
        x_query = np.concatenate(x_query)
        y_query = np.concatenate(y_query)
        
        # Compute prototypes and classify
        prototypes, proto_labels = trainer.compute_prototypes(x_proto, y_proto)
        preds = trainer.classify_with_prototypes(x_query, prototypes, proto_labels)
        acc = np.mean(preds == y_query)
        
        kshot_results[k] = {
            'targets': y_query,
            'predictions': preds,
            'class_names': list(all_class_names),
            'accuracy': acc
        }
        logger.info(f"    {k}-shot accuracy: {acc:.4f}")
    
    results['kshot_confusion_results'] = kshot_results
    
    # Novelty detection
    logger.info("\n--- Novelty Detection Evaluation ---")
    
    # Use training classes as "known", test on unknown
    known_class_indices = [all_class_names.index(c) for c in train_class_names if c in all_class_names]
    unknown_class_indices = [i for i in range(len(all_class_names)) if i not in known_class_indices]
    
    if unknown_class_indices:
        x_known_train = x_full[np.isin(y_full, known_class_indices)]
        y_known_train = y_full[np.isin(y_full, known_class_indices)]
        
        x_known_test = x_test_full[np.isin(y_test_full, known_class_indices)]
        x_unknown_test = x_test_full[np.isin(y_test_full, unknown_class_indices)]
        
        if len(x_known_test) > 0 and len(x_unknown_test) > 0:
            detector = NoveltyDetector(model, device)
            detector.fit(x_known_train, y_known_train)
            novelty_results = detector.evaluate_novelty_detection(x_known_test, x_unknown_test)
            results['novelty_results'] = novelty_results
            
            logger.info(f"Fixed Threshold - F1: {novelty_results['fixed_f1']:.4f}")
            logger.info(f"Statistical Threshold - F1: {novelty_results['statistical_f1']:.4f}")
    
    # Real-world simulation
    logger.info("\n--- Real-World Deployment Simulation ---")
    
    regular_idx_full = all_class_names.index('regular') if 'regular' in all_class_names else 0
    
    simulator = RealWorldSimulator(
        model=model,
        device=device,
        regular_class_idx=regular_idx_full,
        regular_class_name='regular',
        class_names=all_class_names,
        output_dir=output_dir
    )
    
    simulation_results = simulator.simulate_incremental(
        x_support=x_full,
        y_support=y_full,
        x_test=x_test_full,
        y_test=y_test_full,
        k_shots=[1, 5, 10]
    )
    results['simulation_results'] = simulation_results
    
    # =========================================================================
    # Visualization and Reporting
    # =========================================================================
    
    logger.info("\n--- Generating Visualizations ---")
    
    # Generate plots
    generate_all_plots(
        history=history,
        y_test=y_test_train,
        predictions=test_results_5class['predictions'],
        class_names=train_class_names,
        model=model,
        x_test=x_test_train,
        device=device,
        nway_results=results.get('nway_results', {}),
        simulation_results=simulation_results,
        output_dir=output_dir,
        test_results_9class=test_results_9class,
        kshot_results=results.get('kshot_confusion_results', None)
    )
    
    # Generate report
    generate_report(results, output_dir, args)
    
    # Save all results
    with open(os.path.join(output_dir, 'all_results.pkl'), 'wb') as f:
        # Remove non-picklable items
        results_to_save = {k: v for k, v in results.items() 
                          if k not in ['test_results'] or 'prototypes' not in v}
        pickle.dump(results_to_save, f)
    
    # Save results in format compatible with visualization notebook
    np.save(os.path.join(output_dir, 'history.npy'), history)
    
    # Save 9-class results (for comparison with CNN/MLP)
    test_results_for_notebook = {
        'accuracy': test_results_9class['accuracy'],
        'balanced_accuracy': test_results_9class['balanced_accuracy'],
        'f1_macro': test_results_9class['f1_macro'],
        'f1_weighted': test_results_9class['f1_weighted'],
        'confusion_matrix': test_results_9class['confusion_matrix'],
        'class_names': list(all_class_names),
        'predictions': test_results_9class['predictions'],
        'targets': test_results_9class['targets']
    }
    np.save(os.path.join(output_dir, 'test_results.npy'), test_results_for_notebook)
    
    # Also save 5-class results separately
    test_results_5class_for_notebook = {
        'accuracy': test_results_5class['accuracy'],
        'balanced_accuracy': test_results_5class['balanced_accuracy'],
        'f1_macro': test_results_5class['f1_macro'],
        'f1_weighted': test_results_5class['f1_weighted'],
        'confusion_matrix': test_results_5class['confusion_matrix'],
        'class_names': list(train_class_names),
        'predictions': test_results_5class['predictions'],
        'targets': test_results_5class['targets']
    }
    np.save(os.path.join(output_dir, 'test_results_5class.npy'), test_results_5class_for_notebook)
    
    logger.info(f"✓ Saved training history to {output_dir}/history.npy")
    logger.info(f"✓ Saved test results to {output_dir}/test_results.npy")
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)
    
    logger.info(f"Best Val Accuracy: {trainer.best_val_acc:.4f}")
    logger.info(f"Test Accuracy (5-class): {test_results_5class['accuracy']:.4f}")
    logger.info(f"Test Balanced Accuracy (5-class): {test_results_5class['balanced_accuracy']:.4f}")
    logger.info(f"Test Accuracy (9-class): {test_results_9class['accuracy']:.4f}")
    logger.info(f"Test Balanced Accuracy (9-class): {test_results_9class['balanced_accuracy']:.4f}")
    
    logger.info("\nN-way K-shot Results:")
    for config, res in results['nway_results'].items():
        logger.info(f"  {config}: Acc={res['accuracy']:.4f}, F1={res['f1_macro']:.4f}")
    
    if 'novelty_results' in results:
        logger.info("\nNovelty Detection:")
        logger.info(f"  Fixed: F1={results['novelty_results']['fixed_f1']:.4f}")
        logger.info(f"  Statistical: F1={results['novelty_results']['statistical_f1']:.4f}")
    
    logger.info("\nSimulation (Final Step):")
    for k, steps in simulation_results.items():
        if steps:
            final = steps[-1]
            logger.info(f"  {k}-shot: Anomaly F1={final['anomaly_f1']:.4f}, Known Acc={final['known_acc']:.4f}")
    
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()



