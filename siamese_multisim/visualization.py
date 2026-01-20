#!/usr/bin/env python3
"""
Multi-Similarity Siamese Network - Visualization Components

This module provides comprehensive visualization including:
- Training curves (loss, accuracy, F1)
- Similarity metric tracking plots
- Confusion matrices
- t-SNE embedding visualizations
- N-way K-shot performance plots
- Real-world simulation results
- Attention weights visualization

Author: Andrei Ribeiro, Carlos Natalino
Date: January 2026
"""

import os
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch

logger = logging.getLogger(__name__)

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150
})


def plot_training_curves(
    history: Dict[str, List],
    output_dir: str,
    filename: str = 'training_curves.png'
):
    """
    Plot comprehensive training curves.
    
    Includes:
    - Loss curves (train/val)
    - Pair accuracy curves
    - Classification accuracy
    - Anomaly detection F1
    - Learning rate schedule
    
    Args:
        history: Training history dictionary
        output_dir: Output directory
        filename: Output filename
    """
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Loss curves
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=1.5)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Pair accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, history['train_pair_acc'], 'b-', label='Train', linewidth=1.5)
    ax2.plot(epochs, history['val_pair_acc'], 'r-', label='Validation', linewidth=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Pair Accuracy')
    ax2.set_title('Pair Classification Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.4, 1.0])
    
    # 3. F1 scores
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, history['train_f1'], 'b-', label='Train Pair F1', linewidth=1.5)
    ax3.plot(epochs, history['val_f1'], 'r-', label='Val Pair F1', linewidth=1.5)
    if 'val_anomaly_f1' in history and any(history['val_anomaly_f1']):
        ax3.plot(epochs, history['val_anomaly_f1'], 'g--', label='Val Anomaly F1', linewidth=1.5)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('F1 Score')
    ax3.set_title('F1 Scores During Training')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # 4. Classification accuracy
    ax4 = fig.add_subplot(gs[1, 0])
    if 'val_class_acc' in history and any(history['val_class_acc']):
        ax4.plot(epochs, history['val_class_acc'], 'g-', linewidth=2)
        max_acc = max(history['val_class_acc'])
        ax4.axhline(y=max_acc, color='r', linestyle='--', 
                   label=f'Best: {max_acc:.4f}', alpha=0.7)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Balanced Accuracy')
    ax4.set_title('Validation Classification Accuracy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    # 5. Similarity metrics - L2 distance
    ax5 = fig.add_subplot(gs[1, 1])
    if 'pos_l2_mean' in history and any(history['pos_l2_mean']):
        ax5.plot(epochs, history['pos_l2_mean'], 'g-', label='Positive pairs', linewidth=1.5)
        ax5.plot(epochs, history['neg_l2_mean'], 'r-', label='Negative pairs', linewidth=1.5)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('L2 Distance')
    ax5.set_title('L2 Distance (Pos should decrease, Neg increase)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Similarity metrics - Cosine
    ax6 = fig.add_subplot(gs[1, 2])
    if 'pos_cosine_mean' in history and any(history['pos_cosine_mean']):
        ax6.plot(epochs, history['pos_cosine_mean'], 'g-', label='Positive pairs', linewidth=1.5)
        ax6.plot(epochs, history['neg_cosine_mean'], 'r-', label='Negative pairs', linewidth=1.5)
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Cosine Similarity')
    ax6.set_title('Cosine Similarity (Pos should increase, Neg decrease)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([-0.5, 1.0])
    
    # 7. L1 distance
    ax7 = fig.add_subplot(gs[2, 0])
    if 'pos_l1_mean' in history and any(history['pos_l1_mean']):
        ax7.plot(epochs, history['pos_l1_mean'], 'g-', label='Positive pairs', linewidth=1.5)
        ax7.plot(epochs, history['neg_l1_mean'], 'r-', label='Negative pairs', linewidth=1.5)
    ax7.set_xlabel('Epoch')
    ax7.set_ylabel('L1 Distance')
    ax7.set_title('L1 Distance (Manhattan)')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Learning rate
    ax8 = fig.add_subplot(gs[2, 1])
    if 'lr' in history and any(history['lr']):
        ax8.plot(epochs, history['lr'], 'k-', linewidth=1.5)
    ax8.set_xlabel('Epoch')
    ax8.set_ylabel('Learning Rate')
    ax8.set_title('Learning Rate Schedule')
    ax8.set_yscale('log')
    ax8.grid(True, alpha=0.3)
    
    # 9. Combined metrics summary
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    # Create summary text
    summary_lines = ["Training Summary", "=" * 30]
    if history['train_loss']:
        summary_lines.append(f"Final Train Loss: {history['train_loss'][-1]:.4f}")
    if history['val_loss']:
        summary_lines.append(f"Final Val Loss: {history['val_loss'][-1]:.4f}")
    if history['val_pair_acc']:
        summary_lines.append(f"Best Val Pair Acc: {max(history['val_pair_acc']):.4f}")
    if 'val_class_acc' in history and history['val_class_acc']:
        summary_lines.append(f"Best Val Class Acc: {max(history['val_class_acc']):.4f}")
    if 'val_anomaly_f1' in history and history['val_anomaly_f1']:
        summary_lines.append(f"Best Anomaly F1: {max(history['val_anomaly_f1']):.4f}")
    
    ax9.text(0.1, 0.5, '\n'.join(summary_lines), 
             transform=ax9.transAxes, fontsize=11,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Training curves saved to {output_dir}/{filename}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_dir: str,
    filename: str = 'confusion_matrix.png',
    normalize: bool = True
):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_dir: Output directory
        filename: Output filename
        normalize: Whether to normalize the matrix
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw counts
    cm = confusion_matrix(y_true, y_pred)
    disp1 = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp1.plot(ax=axes[0], cmap='Blues', values_format='d')
    axes[0].set_title('Confusion Matrix (Counts)')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Normalized
    cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
    disp2 = ConfusionMatrixDisplay(cm_norm, display_labels=class_names)
    disp2.plot(ax=axes[1], cmap='Blues', values_format='.2f')
    axes[1].set_title('Confusion Matrix (Normalized)')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrix saved to {output_dir}/{filename}")


@torch.no_grad()
def plot_tsne_embeddings(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    class_names: List[str],
    device: torch.device,
    output_dir: str,
    filename: str = 'embeddings_tsne.png',
    n_samples: int = 2000,
    perplexity: int = 30
):
    """
    Plot t-SNE visualization of learned embeddings.
    
    Args:
        model: Trained model
        x: Features
        y: Labels
        class_names: Class names
        device: Torch device
        output_dir: Output directory
        filename: Output filename
        n_samples: Maximum samples to plot
        perplexity: t-SNE perplexity
    """
    model.eval()
    
    # Sample if needed
    if len(x) > n_samples:
        indices = np.random.choice(len(x), n_samples, replace=False)
        x_sample = x[indices]
        y_sample = y[indices]
    else:
        x_sample = x
        y_sample = y
    
    # Compute embeddings
    embeddings = []
    batch_size = 256
    
    for i in range(0, len(x_sample), batch_size):
        batch = torch.FloatTensor(x_sample[i:i+batch_size]).to(device)
        emb = model.forward_one(batch)
        embeddings.append(emb.cpu().numpy())
    
    embeddings = np.vstack(embeddings)
    
    # t-SNE
    logger.info("Computing t-SNE visualization...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Use distinct colors
    n_classes = len(np.unique(y_sample))
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, n_classes)))
    
    for i, cls in enumerate(np.unique(y_sample)):
        mask = y_sample == cls
        label = class_names[cls] if cls < len(class_names) else f"Class {cls}"
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[i % len(colors)]],
            label=label,
            alpha=0.6,
            s=30
        )
    
    ax.legend(loc='best', markerscale=1.5)
    ax.set_title('t-SNE Visualization of Learned Embeddings', fontsize=14)
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"t-SNE visualization saved to {output_dir}/{filename}")


def plot_nway_kshot_results(
    results: Dict[str, Dict],
    output_dir: str,
    filename: str = 'nway_kshot_results.png'
):
    """
    Plot N-way K-shot evaluation results.
    
    Args:
        results: Dictionary mapping config names to results
        output_dir: Output directory
        filename: Output filename
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Parse results
    configs = list(results.keys())
    accuracies = [results[c]['accuracy'] for c in configs]
    acc_stds = [results[c].get('accuracy_std', 0) for c in configs]
    f1_scores = [results[c]['f1_macro'] for c in configs]
    
    x = np.arange(len(configs))
    width = 0.35
    
    # Accuracy plot
    bars1 = axes[0].bar(x, accuracies, width, yerr=acc_stds, 
                        label='Accuracy', color='steelblue', capsize=3)
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('N-way K-shot Classification Accuracy')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(configs, rotation=45, ha='right')
    axes[0].set_ylim([0, 1])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, acc in zip(bars1, accuracies):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
    
    # F1 plot
    bars2 = axes[1].bar(x, f1_scores, width, label='Macro F1', color='forestgreen')
    axes[1].set_ylabel('Macro F1 Score')
    axes[1].set_title('N-way K-shot F1 Score')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(configs, rotation=45, ha='right')
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for bar, f1 in zip(bars2, f1_scores):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{f1:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"N-way K-shot results saved to {output_dir}/{filename}")


def plot_simulation_results(
    results: Dict[int, List[Dict]],
    output_dir: str,
    filename: str = 'realworld_simulation.png'
):
    """
    Plot comprehensive real-world simulation results.
    
    Args:
        results: Simulation results for each k-shot
        output_dir: Output directory
        filename: Output filename
    """
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    k_shots = list(results.keys())
    colors = {1: '#e74c3c', 5: '#3498db', 10: '#2ecc71'}
    
    # 1. Anomaly Detection F1 over steps
    ax1 = fig.add_subplot(gs[0, 0])
    for k in k_shots:
        if not results[k]:
            continue
        steps = [r['step'] for r in results[k]]
        f1s = [r['anomaly_f1'] for r in results[k]]
        ax1.plot(steps, f1s, 'o-', color=colors.get(k, 'gray'),
                label=f'{k}-shot', linewidth=2, markersize=6)
    ax1.set_xlabel('Step (Classes Added)')
    ax1.set_ylabel('Anomaly Detection F1')
    ax1.set_title('Anomaly Detection Performance\n(Regular vs All Others)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # 2. Known Class Accuracy over steps
    ax2 = fig.add_subplot(gs[0, 1])
    for k in k_shots:
        if not results[k]:
            continue
        steps = [r['step'] for r in results[k]]
        accs = [r['known_acc'] for r in results[k]]
        ax2.plot(steps, accs, 's-', color=colors.get(k, 'gray'),
                label=f'{k}-shot', linewidth=2, markersize=6)
    ax2.set_xlabel('Step (Classes Added)')
    ax2.set_ylabel('Classification Accuracy')
    ax2.set_title('Known Class Classification\n(Accuracy on Classes in Pool)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # 3. Unknown Detection Rate (Fixed threshold)
    ax3 = fig.add_subplot(gs[0, 2])
    for k in k_shots:
        if not results[k]:
            continue
        steps = [r['step'] for r in results[k]]
        rates = [r['unknown_det_fixed'] for r in results[k]]
        ax3.plot(steps, rates, '^-', color=colors.get(k, 'gray'),
                label=f'{k}-shot', linewidth=2, markersize=6)
    ax3.set_xlabel('Step (Classes Added)')
    ax3.set_ylabel('Detection Rate')
    ax3.set_title('Unknown Class Detection (Fixed Threshold)\n(Flagging Novel Anomaly Types)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # 4. Unknown Detection Rate (Statistical threshold)
    ax4 = fig.add_subplot(gs[1, 0])
    for k in k_shots:
        if not results[k]:
            continue
        steps = [r['step'] for r in results[k]]
        rates = [r['unknown_det_stat'] for r in results[k]]
        ax4.plot(steps, rates, 'v-', color=colors.get(k, 'gray'),
                label=f'{k}-shot', linewidth=2, markersize=6)
    ax4.set_xlabel('Step (Classes Added)')
    ax4.set_ylabel('Detection Rate')
    ax4.set_title('Unknown Class Detection (Statistical Threshold)\n(Mean + 2*Std)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    # 5. Distance distribution
    ax5 = fig.add_subplot(gs[1, 1])
    k = 5 if 5 in k_shots else k_shots[0]
    if results.get(k):
        steps = [r['step'] for r in results[k]]
        known_dists = [r['mean_known_dist'] for r in results[k]]
        unknown_dists = [r['mean_unknown_dist'] for r in results[k]]
        
        x = np.arange(len(steps))
        width = 0.35
        ax5.bar(x - width/2, known_dists, width, label='Known', color='#3498db', alpha=0.8)
        ax5.bar(x + width/2, unknown_dists, width, label='Unknown', color='#e74c3c', alpha=0.8)
        ax5.set_xlabel('Step')
        ax5.set_ylabel('Mean Distance to Prototype')
        ax5.set_title(f'Distance Distribution ({k}-shot)\nUnknown Should Have Higher Distance')
        ax5.legend()
        ax5.set_xticks(x)
    
    # 6. Precision-Recall curve
    ax6 = fig.add_subplot(gs[1, 2])
    for k in k_shots:
        if not results[k]:
            continue
        precisions = [r['anomaly_precision'] for r in results[k]]
        recalls = [r['anomaly_recall'] for r in results[k]]
        ax6.plot(recalls, precisions, 'o-', color=colors.get(k, 'gray'),
                label=f'{k}-shot', linewidth=2, markersize=6)
    ax6.set_xlabel('Recall')
    ax6.set_ylabel('Precision')
    ax6.set_title('Anomaly Detection\nPrecision-Recall Trajectory')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim([0, 1])
    ax6.set_ylim([0, 1])
    
    # 7. Radar chart (final performance)
    ax7 = fig.add_subplot(gs[2, 0:2], projection='polar')
    categories = ['Anomaly F1', 'Known Acc', 'Unk Det (Fix)', 'Unk Det (Stat)', 'Precision', 'Recall']
    n_cats = len(categories)
    angles = [n/float(n_cats) * 2 * np.pi for n in range(n_cats)]
    angles += angles[:1]
    
    for k in k_shots:
        if not results[k]:
            continue
        final = results[k][-1]
        values = [
            final['anomaly_f1'],
            final['known_acc'],
            final['unknown_det_fixed'],
            final['unknown_det_stat'],
            final['anomaly_precision'],
            final['anomaly_recall']
        ]
        values += values[:1]
        ax7.plot(angles, values, 'o-', color=colors.get(k, 'gray'),
                label=f'{k}-shot', linewidth=2)
        ax7.fill(angles, values, color=colors.get(k, 'gray'), alpha=0.1)
    
    ax7.set_xticks(angles[:-1])
    ax7.set_xticklabels(categories)
    ax7.set_title('Final Performance Summary\n(All Classes in Pool)', fontsize=12, y=1.1)
    ax7.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    
    # 8. Summary table
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    table_data = []
    headers = ['K-shot', 'Anomaly F1', 'Known Acc', 'Unk Det']
    
    for k in k_shots:
        if not results[k]:
            continue
        final = results[k][-1]
        table_data.append([
            f'{k}-shot',
            f'{final["anomaly_f1"]:.4f}',
            f'{final["known_acc"]:.4f}',
            f'{final["unknown_det_fixed"]:.4f}'
        ])
    
    if table_data:
        table = ax8.table(
            cellText=table_data,
            colLabels=headers,
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        ax8.set_title('Final Results Summary', fontsize=12, y=0.85)
    
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Simulation results saved to {output_dir}/{filename}")


def plot_attention_weights(
    history: Dict[str, List],
    output_dir: str,
    filename: str = 'attention_weights.png'
):
    """
    Plot attention weights evolution during training.
    
    Args:
        history: Training history with attention weights
        output_dir: Output directory
        filename: Output filename
    """
    if 'attention_weights' not in history or not any(w is not None for w in history['attention_weights']):
        logger.info("No attention weights to plot")
        return
    
    # Filter valid entries
    valid_epochs = []
    weights = []
    
    for i, w in enumerate(history['attention_weights']):
        if w is not None:
            valid_epochs.append(i + 1)
            weights.append(w)
    
    if not weights:
        return
    
    weights = np.array(weights)
    metric_names = ['L1', 'L2', 'Cosine', 'Product']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, name in enumerate(metric_names):
        ax.plot(valid_epochs, weights[:, i], '-o', label=name, linewidth=1.5, markersize=4)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Attention Weight')
    ax.set_title('Learned Attention Weights for Similarity Metrics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Attention weights plot saved to {output_dir}/{filename}")


def generate_all_plots(
    history: Dict,
    y_test: np.ndarray,
    predictions: np.ndarray,
    class_names: List[str],
    model: torch.nn.Module,
    x_test: np.ndarray,
    device: torch.device,
    nway_results: Dict,
    simulation_results: Dict,
    output_dir: str
):
    """
    Generate all visualization plots.
    
    Args:
        history: Training history
        y_test: True test labels
        predictions: Predicted labels
        class_names: Class names
        model: Trained model
        x_test: Test features
        device: Torch device
        nway_results: N-way K-shot results
        simulation_results: Real-world simulation results
        output_dir: Output directory
    """
    logger.info("Generating all visualization plots...")
    
    # Training curves
    plot_training_curves(history, output_dir)
    
    # Confusion matrix
    plot_confusion_matrix(y_test, predictions, class_names, output_dir)
    
    # t-SNE embeddings
    plot_tsne_embeddings(model, x_test, y_test, class_names, device, output_dir)
    
    # N-way K-shot results
    if nway_results:
        plot_nway_kshot_results(nway_results, output_dir)
    
    # Simulation results
    if simulation_results:
        plot_simulation_results(simulation_results, output_dir)
    
    # Attention weights
    plot_attention_weights(history, output_dir)
    
    logger.info(f"All plots saved to {output_dir}")



