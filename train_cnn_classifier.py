#!/usr/bin/env python3
"""
CNN Classifier for DAS Event Classification

This script implements a convolutional neural network (CNN) for classifying
Distributed Acoustic Sensing (DAS) events based on the architecture described
in the OFS paper. The model processes frequency-domain spectra of DAS signals
to classify events into multiple categories (the dataset contains 9 classes).

Architecture (from paper/figure):
- Input: DAS Channel spectrum (1x2048)
- Conv1D: 64 filters, kernel_size=7
- LeakyReLU activation
- MaxPool1d: kernel_size=4
- Conv1D: 256 filters, kernel_size=7
- LeakyReLU activation
- MaxPool1d: kernel_size=4
- Flatten
- Dense: 1024 neurons with Sigmoid activation
- Dense: 7 neurons (output classes)
- Softmax activation

Author: Auto-generated based on OFS paper architecture
"""

import sys
import os
import logging
import random
import argparse
from pathlib import Path
import pickle
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

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
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    logger.info(f"Random seed set to {seed}")


def get_device():
    """
    Detect and return the best available device (MPS > CUDA > CPU).
    
    Returns:
        torch.device: The device to use for computation
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon (MPS)")
    elif torch.cuda.is_available():
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
    """
    PyTorch Dataset for DAS event classification.
    
    Args:
        x: Feature array (n_samples, n_features)
        y: Label array (n_samples, n_classes) - one-hot encoded
        normalize: If True, normalize features to zero mean and unit variance
    """
    
    def __init__(self, x, y, normalize=True, mean=None, std=None):
        """
        Initialize the dataset.
        
        Args:
            x: Feature array (numpy array)
            y: Label array (numpy array, one-hot encoded)
            normalize: Whether to normalize the features
            mean: Precomputed mean for normalization (if None, compute from x)
            std: Precomputed std for normalization (if None, compute from x)
        """
        # Normalize features to help with training stability
        # The paper doesn't explicitly mention normalization, but it's a standard practice
        # and the data has mean~4.24, std~0.39, which benefits from normalization
        if normalize:
            if mean is None or std is None:
                self.mean = np.mean(x, axis=0, keepdims=True)
                self.std = np.std(x, axis=0, keepdims=True) + 1e-8  # Add small epsilon to avoid division by zero
            else:
                self.mean = mean
                self.std = std
            x = (x - self.mean) / self.std
        else:
            self.mean = None
            self.std = None
        
        # Convert to tensors
        self.x = torch.FloatTensor(x)
        # Convert one-hot to class indices for CrossEntropyLoss
        self.y = torch.LongTensor(np.argmax(y, axis=1))
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ============================================================================
# CNN Model Architecture
# ============================================================================

class DASEventClassifier(nn.Module):
    """
    CNN classifier for DAS event classification.
    
    Architecture based on the OFS paper:
    - Input: 1x2048 (DAS Channel spectrum)
    - Conv1D: 64 filters, kernel_size=7, stride=1
      Output: 64x2042 (2048 - 7 + 1 = 2042)
    - LeakyReLU activation
    - MaxPool1d: kernel_size=4
      Output: 64x510 (2042 / 4 ≈ 510, accounting for rounding)
    - Conv1D: 256 filters, kernel_size=7, stride=1
      Output: 256x504 (510 - 7 + 1 = 504)
    - LeakyReLU activation
    - MaxPool1d: kernel_size=4
      Output: 256x126 (504 / 4 = 126)
    - Flatten
      Output: 256 * 126 = 32256
    - Dense: 1024 neurons with Sigmoid activation
    - Dense: N neurons (output classes, where N is the number of classes in the dataset)
    - Softmax activation (applied in loss function)
    
    Parameters from paper:
    - Input dimension: 2048 (first 2048 elements of single-sided magnitude spectrum)
    - Number of classes: The actual dataset contains 9 classes: car, walk, construction, 
      regular, fence, openclose, longboard, running, manipulation
      (Note: The figure shows 7 classes, but the actual dataset has 9 classes)
    - Conv1D kernel size: 7 (standard choice for 1D CNNs)
    - Pooling size: 4 (reduces dimensionality by factor of 4)
    - Dense layer size: 1024 (provides sufficient capacity for classification)
    """
    
    def __init__(self, input_dim: int = 2048, num_classes: int = 9, use_sigmoid: bool = False, dropout: float = 0.5):
        """
        Initialize the CNN model.
        
        Args:
            input_dim: Input feature dimension (default: 2048 from paper)
            num_classes: Number of output classes (default: 9, actual dataset has 9 classes)
            use_sigmoid: If True, use Sigmoid activation (as in paper), else use ReLU (recommended)
            dropout: Dropout rate for regularization (default: 0.5)
        """
        super(DASEventClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.use_sigmoid = use_sigmoid
        self.dropout_rate = dropout
        
        # First convolutional block
        # Input: [batch, 1, 2048]
        # Output: [batch, 64, 2042] (2048 - 7 + 1 = 2042)
        self.network = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=64,  # From paper/figure
                kernel_size=7,    # From paper/figure
                stride=1,
                padding=0
            ),
        
            # LeakyReLU activation (negative_slope=0.01 is default)
            # From paper/figure - LeakyReLU helps with gradient flow
            nn.LeakyReLU(negative_slope=0.01),
        
            # Max pooling: reduces length by factor of 4
            # Input: [batch, 64, 2042]
            # Output: [batch, 64, 510] (2042 / 4 = 510.5, rounded down to 510)
            nn.MaxPool1d(kernel_size=4),

            nn.Dropout(p=self.dropout_rate),
        
            # Second convolutional block
            # Input: [batch, 64, 510]
            # Output: [batch, 256, 504] (510 - 7 + 1 = 504)
            nn.Conv1d(
                in_channels=64,
                out_channels=256,  # From paper/figure
                kernel_size=7,     # From paper/figure
                stride=1,
                padding=0
            ),
        
            # LeakyReLU activation
            nn.LeakyReLU(negative_slope=0.01),
        
            # Max pooling: reduces length by factor of 4
            # Input: [batch, 256, 504]
            # Output: [batch, 256, 126] (504 / 4 = 126)
            nn.MaxPool1d(kernel_size=4),

            nn.Dropout(p=self.dropout_rate),
        
            # Flatten layer
            # Input: [batch, 256, 126]
            # Output: [batch, 32256] (256 * 126 = 32256)
            nn.Flatten(),

            # nn.AdaptiveAvgPool1d(256 * 126),
        
            # First fully connected layer
            # Input: [batch, 32256]
            # Output: [batch, 1024]
            nn.Linear(
                in_features=256 * 126,  # 32256 from paper/figure
                out_features=1024       # From paper/figure
            ),
        
            # Dropout for regularization to prevent overfitting
            # The model has 33M parameters and is prone to overfitting
            # Dropout randomly sets some neurons to zero during training
            nn.Dropout(p=self.dropout_rate),
            nn.Sigmoid(),
            nn.Linear(
                in_features=1024,
                out_features=num_classes
            ),
        )
        
        # Activation function
        # NOTE: The paper mentions Sigmoid, but this causes vanishing gradients.
        # Using ReLU instead for better gradient flow. If you want to match the
        # paper exactly, set use_sigmoid=True, but expect slower convergence.
        # The paper achieved 91% accuracy, likely with careful initialization
        # and normalization that we're also adding.
        # if self.use_sigmoid:
        #     self.activation = nn.Sigmoid()
        # else:
        #     self.activation = nn.ReLU()
        
        # Initialize weights properly for better training
        # self._initialize_weights()
        
        # Output layer
        # Input: [batch, 1024]
        # Output: [batch, num_classes]
        # self.fc2 = nn.Linear(
        #     in_features=1024,
        #     out_features=num_classes
        # )
        
        # Softmax is applied in the loss function (CrossEntropyLoss includes it)
        # but we can also apply it explicitly if needed
    
    def _initialize_weights(self):
        """
        Initialize weights using He initialization for ReLU or Xavier for Sigmoid.
        This helps with training stability and convergence.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                if self.use_sigmoid:
                    nn.init.xavier_uniform_(m.weight)
                else:
                    nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if self.use_sigmoid:
                    nn.init.xavier_uniform_(m.weight)
                else:
                    nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch, input_dim]
            
        Returns:
            Output tensor of shape [batch, num_classes]
        """
        # Reshape input from [batch, 2048] to [batch, 1, 2048] for Conv1d
        # Conv1d expects [batch, channels, length]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        
        # # First convolutional block
        # x = self.conv1(x)        # [batch, 64, 2042]
        # x = self.leaky_relu1(x)   # [batch, 64, 2042]
        # x = self.pool1(x)         # [batch, 64, 510]
        # x = self.dropout1(x)      # [batch, 64, 510]
        # # Second convolutional block
        # x = self.conv2(x)         # [batch, 256, 504]
        # x = self.leaky_relu2(x)  # [batch, 256, 504]
        # x = self.pool2(x)         # [batch, 256, 126]
        # x = self.dropout2(x)      # [batch, 256, 126]
        # # Flatten
        # x = self.flatten(x)       # [batch, 32256]
        # x = self.global_pool(x)   # [batch, 32256]
        # # Fully connected layers
        # x = self.fc1(x)           # [batch, 1024]
        # x = self.activation(x)     # [batch, 1024] - ReLU or Sigmoid
        # x = self.dropout_fc1(x)   # [batch, 1024] - Dropout for regularization (only active during training)
        # x = self.fc2(x)           # [batch, num_classes]
        
        return self.network(x)


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device, class_weights=None):
    """
    Train the model for one epoch.
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on
        class_weights: Optional class weights tensor for handling class imbalance
        
    Returns:
        Average training loss and accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        # This helps stabilize training, especially with the large model
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * data.size(0)
        probs = torch.softmax(output, dim=1)
        _, predicted = torch.max(probs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    avg_loss = running_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy


def evaluate(model, data_loader, criterion, device):
    """
    Evaluate the model on a dataset.
    
    Args:
        model: The neural network model
        data_loader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to run on
        
    Returns:
        Average loss, accuracy, and predictions
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item() * data.size(0)
            probs = torch.softmax(output, dim=1)
            _, predicted = torch.max(probs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    avg_loss = running_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy, all_predictions, all_targets


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs,
    learning_rate,
    weight_decay,
    device,
    class_weights=None,
    early_stopping_patience=10,
    log_interval=50,
    save_dir=None,
):
    """
    Train the model for multiple epochs.
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay (L2 regularization)
        device: Device to run on
        class_weights: Optional class weights tensor for handling class imbalance
        early_stopping_patience: Number of epochs to wait before early stopping
        log_interval: Interval for logging batch progress
        
    Returns:
        Training history dictionary and best model state
    """
    # Loss function: CrossEntropyLoss includes Softmax
    # Use class weights if provided to handle class imbalance
    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        logger.info(f"Using class weights: {class_weights.cpu().numpy()}")
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer: Adam with weight decay
    # Learning rate and weight decay are hyperparameters not specified in paper
    # Using common defaults: lr=1e-3, weight_decay=1e-4
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    logger.info(f"Starting training for {num_epochs} epochs...")
    logger.info(f"Learning rate: {learning_rate}, Weight decay: {weight_decay}")
    logger.info(f"Early stopping patience: {early_stopping_patience} epochs")
    
    best_val_acc = 0.0
    best_model_state = None
    epochs_without_improvement = 0
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, class_weights
        )
        
        # Validate
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Early stopping: save best model and check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
            improvement_msg = " * BEST *"
        else:
            epochs_without_improvement += 1
            improvement_msg = ""
        
        # Log progress
        logger.info(
            f"Epoch [{epoch}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}{improvement_msg}"
        )
        
        # Early stopping
        if epochs_without_improvement >= early_stopping_patience:
            logger.info(
                f"Early stopping triggered after {epoch} epochs. "
                f"No improvement for {early_stopping_patience} epochs."
            )
            break
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        torch.save(best_model_state, f"{save_dir}/cnn_best_classifier_model.pth")
        logger.info(f"Loaded best model state (validation accuracy: {best_val_acc:.4f})")
    
    return history


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main training script."""
    parser = argparse.ArgumentParser(
        description='Train CNN classifier for DAS event classification'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/nobackup/carda/datasets/DAS-dataset/data',
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs (default: 10)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size (default: 64)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate (default: 1e-4, reduced from 1e-3 for better stability)'
    )
    parser.add_argument(
        '--use_sigmoid',
        action='store_true',
        help='Use Sigmoid activation (as in paper) instead of ReLU (not recommended)'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4,
        help='Weight decay (default: 1e-4)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--val_size',
        type=float,
        default=0.1,
        help='Validation set size (default: 0.1)'
    )
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.1,
        help='Test set size (default: 0.1)'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.5,
        help='Dropout rate for regularization (default: 0.5)'
    )
    parser.add_argument(
        '--early_stopping_patience',
        type=int,
        default=10,
        help='Early stopping patience (epochs without improvement, default: 10)'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='checkpoints',
        help='Directory to save checkpoints (default: checkpoints)'
    )
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    # Set random seed
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    
    # ========================================================================
    # Data Loading
    # ========================================================================
    logger.info("=" * 70)
    logger.info("Loading and preprocessing dataset...")
    logger.info("=" * 70)
    
    # Decimation dictionary (from README.md example)
    # Reduces dataset size by sampling every Nth sample for 'regular' class
    decim_dict = {
        # 'regular': 90,  # Decimate regular class by factor of 50
        # 'fence': 90,
        # 'longboard': 90,
        # 'manipulation': 90,
        # 'openclose': 90,
        # 'running': 90,
        # 'walk': 90,
        # 'car': 90,
        # 'construction': 90,
    }
    
    # Initialize data loader with parameters from README.md
    # Parameters from paper/README:
    # - sample_len: 2048 (first 2048 elements of spectrum)
    # - transform: fft (FFT transformation to frequency domain)
    # - fsize: 8192 (window size for sliding window)
    # - shift: 2048 (overlap of 75% with fsize=8192)
    parser_loader = DASDataLoader(
        data_dir=args.data_dir,
        sample_len=2048,      # From paper: first 2048 elements of spectrum
        transform=fft,         # FFT transformation (from README.md)
        fsize=8192,           # Window size (from README.md)
        shift=2048,           # Step size (from README.md)
        decimate=decim_dict,  # Decimation (from README.md)
    )
    
    # Parse dataset
    x, y = parser_loader.parse_dataset()
    
    logger.info(f"Dataset loaded: {len(x)} samples")
    logger.info(f"Feature shape: {x.shape}")
    logger.info(f"Label shape: {y.shape}")
    logger.info(f"Number of classes: {y.shape[1]}")
    logger.info(f"Class names: {parser_loader.encoder.classes_}")
    
    # Get number of classes from data
    num_classes = y.shape[1]
    
    # Get class weights from data loader (computed during encoding)
    # These weights help handle class imbalance
    # class_weights = torch.FloatTensor(parser_loader.class_weights)
    # logger.info(f"Class weights for handling imbalance: {class_weights.numpy()}")
    
    # ========================================================================
    # Data Splitting: Train / Validation / Test
    # ========================================================================
    logger.info("=" * 70)
    logger.info("Splitting dataset into train, validation, and test sets...")
    logger.info("=" * 70)
    
    # Convert one-hot to class indices for stratified split
    y_classes = np.argmax(y, axis=1)
    
    # First split: separate test set
    # Paper uses 80:10:10 split (train:val:test)
    test_size_actual = args.test_size
    val_size_actual = args.val_size / (1 - test_size_actual)  # Adjust for two-stage split
    
    X_temp, X_test, Y_temp, Y_test = train_test_split(
        x, y,
        test_size=test_size_actual,
        random_state=args.seed,
        stratify=y_classes
    )
    
    # Second split: separate train and validation from remaining data
    y_temp_classes = np.argmax(Y_temp, axis=1)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_temp, Y_temp,
        test_size=val_size_actual,
        random_state=args.seed,
        stratify=y_temp_classes
    )
    
    logger.info(f"Training samples: {len(X_train)} ({100*len(X_train)/len(x):.1f}%)")
    logger.info(f"Validation samples: {len(X_val)} ({100*len(X_val)/len(x):.1f}%)")
    logger.info(f"Test samples: {len(X_test)} ({100*len(X_test)/len(x):.1f}%)")
    
    # get class weights from training set
    # count the number of samples per class in the training set
    class_counts = Counter(np.argmax(Y_train, axis=1))
    logger.info(f"Class counts: {class_counts}")
    # get the class weights
    class_weights = torch.FloatTensor([x / len(X_train) for x in class_counts.values()])
    logger.info(f"Class weights: {class_weights.numpy()}")
    
    # ========================================================================
    # Create DataLoaders
    # ========================================================================
    # Normalize training data and use same normalization for validation and test
    complete_dataset = DASClassificationDataset(x, y, normalize=True)
    train_dataset = DASClassificationDataset(X_train, Y_train, normalize=True, mean=complete_dataset.mean, std=complete_dataset.std)
    val_dataset = DASClassificationDataset(X_val, Y_val, normalize=True, mean=complete_dataset.mean, std=complete_dataset.std)
    test_dataset = DASClassificationDataset(X_test, Y_test, normalize=True, mean=complete_dataset.mean, std=complete_dataset.std)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # ========================================================================
    # Model Initialization
    # ========================================================================
    logger.info("=" * 70)
    logger.info("Initializing model...")
    logger.info("=" * 70)
    
    model = DASEventClassifier(
        input_dim=2048,      # From paper
        num_classes=num_classes,
        use_sigmoid=args.use_sigmoid,
        dropout=args.dropout
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model architecture:")
    logger.info(model)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # ========================================================================
    # Training
    # ========================================================================
    logger.info("=" * 70)
    logger.info("Starting training...")
    logger.info("=" * 70)
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        class_weights=class_weights,
        early_stopping_patience=args.early_stopping_patience,
        save_dir=args.save_dir
    )

    # Final evaluation on training set
    logger.info("=" * 70)
    logger.info("Final evaluation on training set...")
    logger.info("=" * 70)
    
    criterion = nn.CrossEntropyLoss()
    train_loss, train_acc, train_predictions, train_targets = evaluate(
        model, train_loader, criterion, device
    )
    
    logger.info(f"Final Training Loss: {train_loss:.4f}")
    logger.info(f"Final Training Accuracy: {train_acc:.4f}")

    # Classification report on training set
    logger.info("\nTraining Set Classification Report:")
    logger.info("\n" + classification_report(
        train_targets,
        train_predictions,
        target_names=parser_loader.encoder.classes_
    ))
    
    # Confusion matrix for training set
    train_cm = confusion_matrix(train_targets, train_predictions)
    ConfusionMatrixDisplay(train_cm, display_labels=parser_loader.encoder.classes_).plot()
    plt.xticks(ha='right', rotation=45)
    plt.tight_layout()
    plt.savefig(f'{args.save_dir}/cnn_confusion_matrix_training.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{args.save_dir}/cnn_confusion_matrix_training.pdf', bbox_inches='tight')
    plt.close()
    logger.info("\nTraining Set Confusion Matrix:")
    logger.info(f"\n{train_cm}")
    
    # ========================================================================
    # Final Evaluation on Validation Set
    # ========================================================================
    logger.info("=" * 70)
    logger.info("Final evaluation on validation set...")
    logger.info("=" * 70)
    
    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc, val_predictions, val_targets = evaluate(
        model, val_loader, criterion, device
    )
    
    logger.info(f"Final Validation Loss: {val_loss:.4f}")
    logger.info(f"Final Validation Accuracy: {val_acc:.4f}")

    # Also show validation set results for comparison
    logger.info("\n" + "=" * 70)
    logger.info("Validation Set Classification Report (for comparison):")
    logger.info("=" * 70)
    logger.info("\n" + classification_report(
        val_targets,
        val_predictions,
        target_names=parser_loader.encoder.classes_
    ))
    
    # Confusion matrix for validation set
    val_cm = confusion_matrix(val_targets, val_predictions)
    ConfusionMatrixDisplay(val_cm, display_labels=parser_loader.encoder.classes_).plot()
    plt.xticks(ha='right', rotation=45)
    plt.tight_layout()
    plt.savefig(f'{args.save_dir}/cnn_confusion_matrix_validation.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{args.save_dir}/cnn_confusion_matrix_validation.pdf', bbox_inches='tight')
    plt.close()
    logger.info("\nValidation Set Confusion Matrix:")
    logger.info(f"\n{val_cm}")
    # ========================================================================
    # Test Set Evaluation
    # ========================================================================
    logger.info("=" * 70)
    logger.info("Evaluating on held-out test set...")
    logger.info("=" * 70)
    
    test_loss, test_acc, test_predictions, test_targets = evaluate(
        model, test_loader, criterion, device
    )
    
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    
    # Classification report on test set
    logger.info("\nTest Set Classification Report:")
    logger.info("\n" + classification_report(
        test_targets,
        test_predictions,
        target_names=parser_loader.encoder.classes_
    ))
    
    # Confusion matrix for test set
    test_cm = confusion_matrix(test_targets, test_predictions)
    ConfusionMatrixDisplay(test_cm, display_labels=parser_loader.encoder.classes_).plot()
    plt.xticks(ha='right', rotation=45)
    plt.tight_layout()
    plt.savefig(f'{args.save_dir}/cnn_confusion_matrix_test.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{args.save_dir}/cnn_confusion_matrix_test.pdf', bbox_inches='tight')
    plt.close()
    logger.info("\nTest Set Confusion Matrix:")
    logger.info(f"\n{test_cm}")
    
    # ========================================================================
    # Plot Training History
    # ========================================================================
    logger.info("=" * 70)
    logger.info("Plotting training history...")
    logger.info("=" * 70)
    
    plt.figure()
    plt.plot(history['train_loss'], 'b-', label='Training', linewidth=2, marker='o', markersize=4)
    plt.plot(history['val_loss'], 'g-', label='Validation', linewidth=2, marker='s', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3, ls=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{args.save_dir}/training_history_cnn_classifier_loss.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{args.save_dir}/training_history_cnn_classifier_loss.pdf', bbox_inches='tight')
    plt.close()
    
    # Plot accuracy
    plt.figure()
    plt.plot(history['train_acc'], 'b-', label='Training', linewidth=2, marker='o', markersize=4)
    plt.plot(history['val_acc'], 'g-', label='Validation', linewidth=2, marker='s', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3, ls=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{args.save_dir}/training_history_cnn_classifier_acc.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{args.save_dir}/training_history_cnn_classifier_acc.pdf', bbox_inches='tight')
    plt.close()
    logger.info("Training history saved to 'training_history_cnn_classifier_acc.png'")
    
    # Print summary
    logger.info("=" * 70)
    logger.info("Training Summary")
    logger.info("=" * 70)
    logger.info(f"Total Epochs: {len(history['train_loss'])}")
    logger.info(f"Final Training Loss: {history['train_loss'][-1]:.4f}")
    logger.info(f"Final Training Accuracy: {history['train_acc'][-1]:.4f}")
    logger.info(f"Final Validation Loss: {history['val_loss'][-1]:.4f}")
    logger.info(f"Final Validation Accuracy: {history['val_acc'][-1]:.4f}")
    logger.info(f"Best Validation Accuracy: {max(history['val_acc']):.4f} "
                f"(Epoch {history['val_acc'].index(max(history['val_acc'])) + 1})")
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info("=" * 70)
    
    logger.info("Training completed successfully!")

    with open(f'{args.save_dir}/training_info.pkl', 'wb') as f:
        pickle.dump({
            'training_loss': history['train_loss'],
            'training_acc': history['train_acc'],
            'validation_loss': history['val_loss'],
            'validation_acc': history['val_acc'],
            'training_predictions': train_predictions,
            'training_targets': train_targets,
            'validation_predictions': val_predictions,
            'validation_targets': val_targets,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_predictions': test_predictions,
            'test_targets': test_targets,
        }, f)


if __name__ == '__main__':
    main()

