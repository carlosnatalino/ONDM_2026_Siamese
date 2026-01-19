#!/usr/bin/env python3
"""
FFNN (Feed-Forward Neural Network) Classifier for DAS Event Classification

This script implements a simplified fully-connected neural network as a baseline
comparison to the CNN architecture. The FFNN operates directly on the flattened
frequency-domain features without convolutional layers.

Architecture:
- Input: 2048 features (DAS Channel spectrum)
- Dense: 1024 neurons with ReLU + Dropout
- Dense: 512 neurons with ReLU + Dropout
- Dense: 256 neurons with ReLU + Dropout
- Dense: N neurons (output classes)
- Softmax activation

This serves as a simpler baseline that achieves comparable results to CNN
but with fewer inductive biases (no spatial locality assumptions).

Author: Auto-generated for comparative analysis
Date: January 2026
"""

import sys
import logging
import random
import argparse
import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    f1_score, accuracy_score, balanced_accuracy_score
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Import data loader
from data_loader import DASDataLoader, fft

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Utility Functions
# ============================================================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def get_device():
    """Detect and return the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


# ============================================================================
# Dataset Class
# ============================================================================

class DASClassificationDataset(Dataset):
    """PyTorch Dataset for DAS event classification."""
    
    def __init__(self, x, y, normalize=True, mean=None, std=None):
        if normalize:
            if mean is None or std is None:
                self.mean = np.mean(x, axis=0, keepdims=True)
                self.std = np.std(x, axis=0, keepdims=True) + 1e-8
            else:
                self.mean = mean
                self.std = std
            x = (x - self.mean) / self.std
        else:
            self.mean = None
            self.std = None
        
        self.x = torch.FloatTensor(x)
        self.y = torch.LongTensor(np.argmax(y, axis=1))
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ============================================================================
# FFNN Model Architecture
# ============================================================================

class FFNNClassifier(nn.Module):
    """
    Feed-Forward Neural Network for DAS event classification.
    
    Simplified architecture without convolutions:
    - Input: 2048 features
    - FC1: 2048 → 1024 with ReLU + Dropout
    - FC2: 1024 → 512 with ReLU + Dropout
    - FC3: 512 → 256 with ReLU + Dropout
    - FC4: 256 → num_classes
    - Softmax (applied in loss)
    
    Args:
        input_dim: Input feature dimension (default: 2048)
        num_classes: Number of output classes
        dropout: Dropout rate for regularization
    """
    
    def __init__(self, input_dim: int = 2048, num_classes: int = 9, dropout: float = 0.5):
        super(FFNNClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout
        
        # Fully connected layers
        self.fc1 = nn.Linear(input_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(dropout)
        
        self.fc4 = nn.Linear(256, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"FFNN Classifier initialized:")
        logger.info(f"  Input dim: {input_dim}")
        logger.info(f"  Hidden layers: [1024, 512, 256]")
        logger.info(f"  Output classes: {num_classes}")
        logger.info(f"  Dropout rate: {dropout}")
        logger.info(f"  Total parameters: {self.count_parameters():,}")
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x):
        """Forward pass through the network."""
        # FC1
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # FC2
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # FC3
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        # FC4 (output)
        x = self.fc4(x)
        
        return x


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(target.cpu().numpy())
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(train_loader)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy, f1, balanced_acc


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(val_loader)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy, f1, balanced_acc


def test(model, test_loader, device, class_names):
    """Test the model and generate detailed metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    # Classification report
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        digits=4
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    results = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'confusion_matrix': cm,
        'classification_report': report
    }
    
    return results


def plot_training_history(history, output_dir):
    """Plot training history curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train')
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Validation')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # F1 Score
    axes[1, 0].plot(epochs, history['train_f1'], 'b-', label='Train')
    axes[1, 0].plot(epochs, history['val_f1'], 'r-', label='Validation')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Training and Validation F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Balanced Accuracy
    axes[1, 1].plot(epochs, history['train_balanced_acc'], 'b-', label='Train')
    axes[1, 1].plot(epochs, history['val_balanced_acc'], 'r-', label='Validation')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Balanced Accuracy')
    axes[1, 1].set_title('Training and Validation Balanced Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_history.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Training history saved to {output_dir}/training_history.png")


def plot_confusion_matrix(cm, class_names, output_dir):
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - FFNN Classifier')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix saved to {output_dir}/confusion_matrix.png")


# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='FFNN Classifier for DAS Event Classification'
    )
    
    # Paths
    parser.add_argument('--data_dir', type=str,
                       default='/nobackup/carda/datasets/DAS-dataset/data',
                       help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='ffnn_results',
                       help='Output directory')
    
    # Model
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--early_stopping', type=int, default=20,
                       help='Early stopping patience')
    
    # Data
    parser.add_argument('--val_split', type=float, default=0.15,
                       help='Validation split ratio')
    parser.add_argument('--test_split', type=float, default=0.15,
                       help='Test split ratio')
    
    # Debug
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode with reduced dataset')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"{args.output_dir}_{timestamp}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup file logging
    file_handler = logging.FileHandler(f"{output_dir}/training.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("=" * 70)
    logger.info("FFNN Classifier for DAS Event Classification")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Arguments: {vars(args)}")
    
    # =========================================================================
    # Load Data
    # =========================================================================
    
    logger.info("Loading dataset...")
    
    if args.debug:
        decim_dict = {
            'regular': 100, 'walk': 100, 'car': 100,
            'manipulation': 100, 'openclose': 100, 'fence': 100,
            'longboard': 100, 'running': 100, 'construction': 100
        }
    else:
        decim_dict = None
    
    loader = DASDataLoader(
        data_dir=args.data_dir,
        sample_len=2048,
        transform=fft,
        fsize=8192,
        shift=2048,
        decimate=decim_dict,
        drop_noise=True,
    )
    
    x, y_onehot = loader.parse_dataset()
    class_names = list(loader.encoder.classes_)
    num_classes = len(class_names)
    
    logger.info(f"Dataset loaded: {len(x)} samples, {num_classes} classes")
    logger.info(f"Classes: {class_names}")
    
    # Split data
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x, y_onehot, test_size=args.test_split, random_state=args.seed, stratify=y_onehot.argmax(axis=1)
    )
    
    val_size_adjusted = args.val_split / (1 - args.test_split)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, test_size=val_size_adjusted,
        random_state=args.seed, stratify=y_train_val.argmax(axis=1)
    )
    
    logger.info(f"Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")
    
    # Create datasets
    train_dataset = DASClassificationDataset(x_train, y_train, normalize=True)
    val_dataset = DASClassificationDataset(
        x_val, y_val, normalize=True,
        mean=train_dataset.mean, std=train_dataset.std
    )
    test_dataset = DASClassificationDataset(
        x_test, y_test, normalize=True,
        mean=train_dataset.mean, std=train_dataset.std
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    # =========================================================================
    # Model Setup
    # =========================================================================
    
    device = get_device()
    
    model = FFNNClassifier(
        input_dim=x_train.shape[1],
        num_classes=num_classes,
        dropout=args.dropout
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # =========================================================================
    # Training
    # =========================================================================
    
    logger.info("=" * 70)
    logger.info("Starting Training")
    logger.info("=" * 70)
    
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [], 'train_balanced_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_balanced_acc': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc, train_f1, train_bal_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, val_f1, val_bal_acc = validate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['train_balanced_acc'].append(train_bal_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_balanced_acc'].append(val_bal_acc)
        
        # Log progress
        logger.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.4f}"
        )
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{output_dir}/best_model.pt")
            logger.info(f"✓ New best model saved (Val Acc: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.early_stopping:
            logger.info(f"Early stopping triggered after {epoch} epochs")
            break
    
    # =========================================================================
    # Evaluation
    # =========================================================================
    
    logger.info("=" * 70)
    logger.info("Final Evaluation on Test Set")
    logger.info("=" * 70)
    
    # Load best model
    model.load_state_dict(torch.load(f"{output_dir}/best_model.pt"))
    
    # Test
    test_results = test(model, test_loader, device, class_names)
    
    logger.info(f"Test Accuracy: {test_results['accuracy']:.4f}")
    logger.info(f"Test Balanced Accuracy: {test_results['balanced_accuracy']:.4f}")
    logger.info(f"Test F1 (Macro): {test_results['f1_macro']:.4f}")
    logger.info(f"Test F1 (Weighted): {test_results['f1_weighted']:.4f}")
    logger.info("\nClassification Report:")
    logger.info(f"\n{test_results['classification_report']}")
    
    # =========================================================================
    # Save Results
    # =========================================================================
    
    # Plot training history
    plot_training_history(history, output_dir)
    
    # Plot confusion matrix
    plot_confusion_matrix(test_results['confusion_matrix'], class_names, output_dir)
    
    # Save results
    np.save(f"{output_dir}/history.npy", history)
    np.save(f"{output_dir}/test_results.npy", test_results)
    
    # Save report
    with open(f"{output_dir}/report.txt", 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("FFNN Classifier - Final Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Test Accuracy: {test_results['accuracy']:.4f}\n")
        f.write(f"Test Balanced Accuracy: {test_results['balanced_accuracy']:.4f}\n")
        f.write(f"Test F1 (Macro): {test_results['f1_macro']:.4f}\n")
        f.write(f"Test F1 (Weighted): {test_results['f1_weighted']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(test_results['classification_report'])
    
    logger.info(f"\nAll results saved to {output_dir}/")
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
