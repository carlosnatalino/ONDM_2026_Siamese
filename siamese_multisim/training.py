#!/usr/bin/env python3
"""
Multi-Similarity Siamese Network - Training Components

This module implements the training infrastructure including:
- Class-balanced episodic sampling for few-shot learning
- Binary Cross-Entropy loss with multi-similarity regularization
- Signal augmentation for DAS data
- Training and evaluation loops

References:
- Vinyals et al., "Matching Networks for One Shot Learning" (2016)
- Snell et al., "Prototypical Networks for Few-shot Learning" (2017)
- Wang et al., "Multi-Similarity Loss with General Pair Weighting" (2019)

Author: Andrei Ribeiro, Carlos Natalino
Date: January 2026
"""

import random
import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler
from sklearn.metrics import f1_score, accuracy_score

logger = logging.getLogger(__name__)


# =============================================================================
# Data Augmentation
# =============================================================================

class SignalAugmentation:
    """
    Augmentation strategies for 1D frequency spectra (FFT-transformed DAS signals).
    
    These augmentations help the model become robust to:
    - Sensor noise (Gaussian noise injection)
    - Signal strength variations (random scaling)
    - Missing frequency information (frequency masking - SpecAugment style)
    - Temporal variations (time warping via interpolation)
    
    Reference: Park et al., "SpecAugment: Simple Data Augmentation for ASR" (2019)
    
    Args:
        noise_std: Standard deviation for Gaussian noise
        scale_range: Range for random scaling
        freq_mask_param: Maximum frequency mask width
        n_freq_masks: Number of frequency masks to apply
        dropout_rate: Probability of zeroing individual elements
    """
    
    def __init__(
        self,
        noise_std: float = 0.05,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        freq_mask_param: int = 50,
        n_freq_masks: int = 1,
        dropout_rate: float = 0.05
    ):
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.freq_mask_param = freq_mask_param
        self.n_freq_masks = n_freq_masks
        self.dropout_rate = dropout_rate
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply random augmentations to input signal."""
        x = x.copy()
        
        # Gaussian noise injection
        if random.random() > 0.5:
            noise = np.random.normal(0, self.noise_std, x.shape)
            x = x + noise
        
        # Random scaling (amplitude variation)
        if random.random() > 0.5:
            scale = random.uniform(*self.scale_range)
            x = x * scale
        
        # Frequency masking (SpecAugment-style)
        if random.random() > 0.5:
            for _ in range(self.n_freq_masks):
                f = random.randint(0, self.freq_mask_param)
                f0 = random.randint(0, max(0, len(x) - f))
                x[f0:f0 + f] = 0
        
        # Random element dropout
        if random.random() > 0.5:
            mask = np.random.random(x.shape) > self.dropout_rate
            x = x * mask
        
        return x.astype(np.float32)


# =============================================================================
# Episodic Pair Sampler
# =============================================================================

class EpisodicPairSampler(Sampler):
    """
    Episodic sampler for Siamese pair training.
    
    Implements class-balanced sampling strategy following the episodic
    training paradigm used in few-shot learning (Vinyals et al., 2016).
    
    Strategy:
    1. Each episode samples pairs in a class-balanced manner
    2. Positive pairs: same class, different instances
    3. Negative pairs: different classes
    4. Equal class representation regardless of dataset imbalance
    
    This ensures minority classes receive adequate training signal,
    which is crucial for the imbalanced DAS dataset.
    
    Args:
        labels: Array of class labels
        batch_size: Number of pairs per batch
        positive_ratio: Fraction of positive (same-class) pairs
        n_batches: Number of batches per epoch
    """
    
    def __init__(
        self,
        labels: np.ndarray,
        batch_size: int = 64,
        positive_ratio: float = 0.5,
        n_batches: int = 100
    ):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.positive_ratio = positive_ratio
        self.n_batches = n_batches
        
        # Build class-to-indices mapping
        self.class_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)
        
        self.classes = list(self.class_indices.keys())
        self.n_classes = len(self.classes)
        
        # Log sampling statistics
        logger.info(f"EpisodicPairSampler initialized:")
        logger.info(f"  Classes: {self.n_classes}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Positive ratio: {positive_ratio}")
        logger.info(f"  Batches per epoch: {n_batches}")
        for cls in self.classes:
            logger.info(f"    Class {cls}: {len(self.class_indices[cls])} samples")
    
    def __iter__(self):
        """Generate batches of pair indices for episodic training."""
        for _ in range(self.n_batches):
            batch_pairs = []
            batch_labels = []
            
            # Calculate positive and negative pair counts
            n_positive = int(self.batch_size * self.positive_ratio)
            n_negative = self.batch_size - n_positive
            
            # Sample positive pairs (same class)
            for _ in range(n_positive):
                # Uniformly sample a class (class-balanced)
                cls = random.choice(self.classes)
                indices = self.class_indices[cls]
                
                if len(indices) >= 2:
                    idx1, idx2 = random.sample(indices, 2)
                else:
                    # Single sample class: use same sample (augmentation will differ)
                    idx1 = idx2 = indices[0]
                
                batch_pairs.append((idx1, idx2))
                batch_labels.append(1)  # Positive pair
            
            # Sample negative pairs (different classes)
            for _ in range(n_negative):
                # Sample two different classes
                cls1, cls2 = random.sample(self.classes, 2)
                
                idx1 = random.choice(self.class_indices[cls1])
                idx2 = random.choice(self.class_indices[cls2])
                
                batch_pairs.append((idx1, idx2))
                batch_labels.append(0)  # Negative pair
            
            # Shuffle pairs within batch
            combined = list(zip(batch_pairs, batch_labels))
            random.shuffle(combined)
            batch_pairs, batch_labels = zip(*combined)
            
            yield list(batch_pairs), list(batch_labels)
    
    def __len__(self):
        return self.n_batches


# =============================================================================
# Pair Dataset
# =============================================================================

class SiamesePairDataset(Dataset):
    """
    Dataset for Siamese pair training.
    
    Provides access to individual samples and pair construction
    with optional data augmentation.
    
    Args:
        x: Feature array (N, feature_dim)
        y: Label array (N,)
        augment: Whether to apply augmentation
        augmentation: Augmentation function
    """
    
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        augment: bool = False,
        augmentation: Optional[SignalAugmentation] = None
    ):
        self.x = x
        self.y = y
        self.augment = augment
        self.augmentation = augmentation or SignalAugmentation()
        
        # Build class indices for analysis
        self.class_indices = {}
        for idx, label in enumerate(y):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)
        
        self.classes = list(self.class_indices.keys())
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        """Get a single sample."""
        x = self.x[idx]
        if self.augment:
            x = self.augmentation(x)
        return torch.FloatTensor(x), self.y[idx]
    
    def get_pair(self, idx1: int, idx2: int, label: int):
        """Get a pair of samples with similarity label."""
        x1 = self.x[idx1]
        x2 = self.x[idx2]
        
        if self.augment:
            x1 = self.augmentation(x1)
            x2 = self.augmentation(x2)
        
        return (
            torch.FloatTensor(x1),
            torch.FloatTensor(x2),
            torch.FloatTensor([label])
        )


# =============================================================================
# Multi-Similarity BCE Loss
# =============================================================================

class MultiSimilarityBCELoss(nn.Module):
    """
    Binary Cross-Entropy Loss with Multi-Similarity regularization.
    
    Combines:
    1. BCE Loss: Standard binary cross-entropy for pair classification
    2. Multi-Similarity Regularization: Encourages proper embedding structure
    
    The regularization term considers:
    - Positive pairs should have small embedding distances
    - Negative pairs should have large embedding distances
    - Hard examples (near the margin) receive more gradient
    
    Reference: Wang et al., "Multi-Similarity Loss with General Pair Weighting" (2019)
    
    Args:
        margin: Distance margin for contrastive component
        alpha: Scaling factor for positive pair loss
        beta: Scaling factor for negative pair loss
        lambda_ms: Weight for multi-similarity regularization
    """
    
    def __init__(
        self,
        margin: float = 0.5,
        alpha: float = 2.0,
        beta: float = 50.0,
        lambda_ms: float = 0.1
    ):
        super().__init__()
        
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.lambda_ms = lambda_ms
        
        self.bce_loss = nn.BCELoss()
        
        logger.info(f"MultiSimilarityBCELoss initialized:")
        logger.info(f"  Margin: {margin}, Alpha: {alpha}, Beta: {beta}")
        logger.info(f"  Lambda MS: {lambda_ms}")
    
    def forward(
        self,
        similarity: torch.Tensor,
        labels: torch.Tensor,
        emb1: torch.Tensor,
        emb2: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined BCE + Multi-Similarity loss.
        
        Args:
            similarity: Predicted similarity scores (batch, 1)
            labels: Ground truth labels (batch, 1)
            emb1: First embeddings (batch, embedding_dim)
            emb2: Second embeddings (batch, embedding_dim)
            
        Returns:
            loss: Combined loss value
            metrics: Dictionary of loss components
        """
        # Primary BCE Loss
        bce = self.bce_loss(similarity, labels)
        
        # Multi-Similarity regularization
        # Compute pairwise L2 distances
        distances = torch.sqrt(((emb1 - emb2) ** 2).sum(dim=1) + 1e-8)
        
        pos_mask = labels.squeeze() == 1
        neg_mask = labels.squeeze() == 0
        
        ms_loss = torch.tensor(0.0, device=similarity.device)
        
        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            pos_distances = distances[pos_mask]
            neg_distances = distances[neg_mask]
            
            # Positive loss: penalize large distances for positive pairs
            # log(1 + exp(alpha * (d - margin)))
            pos_loss = torch.log1p(
                torch.exp(self.alpha * (pos_distances - self.margin))
            ).mean()
            
            # Negative loss: penalize small distances for negative pairs
            # log(1 + exp(-beta * (d - margin)))
            neg_loss = torch.log1p(
                torch.exp(-self.beta * (neg_distances - self.margin))
            ).mean()
            
            ms_loss = pos_loss + neg_loss
        
        # Combined loss
        total_loss = bce + self.lambda_ms * ms_loss
        
        metrics = {
            'bce': bce.item(),
            'ms': ms_loss.item() if isinstance(ms_loss, torch.Tensor) else ms_loss,
            'total': total_loss.item()
        }
        
        return total_loss, metrics


# =============================================================================
# Episodic Trainer
# =============================================================================

class EpisodicTrainer:
    """
    Trainer for episodic Siamese network training.
    
    Implements the training loop with:
    - Episodic batch sampling
    - Multi-similarity loss
    - Comprehensive metric tracking (including F1)
    - Early stopping
    - Checkpoint saving
    
    Args:
        model: The Siamese network
        device: Torch device (cuda/cpu)
        output_dir: Directory for saving results
        regular_class_idx: Index of "regular" class for anomaly detection
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        output_dir: str,
        regular_class_idx: int = 0
    ):
        self.model = model.to(device)
        self.device = device
        self.output_dir = output_dir
        self.regular_class_idx = regular_class_idx
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_pair_acc': [],
            'train_f1': [],
            'val_loss': [],
            'val_pair_acc': [],
            'val_class_acc': [],
            'val_f1': [],
            'val_anomaly_f1': [],
            'lr': [],
            # Similarity metric tracking
            'pos_l1_mean': [],
            'neg_l1_mean': [],
            'pos_l2_mean': [],
            'neg_l2_mean': [],
            'pos_cosine_mean': [],
            'neg_cosine_mean': [],
            'attention_weights': []
        }
        
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.best_model_state = None
    
    def train_epoch(
        self,
        dataset: SiamesePairDataset,
        sampler: EpisodicPairSampler,
        optimizer: torch.optim.Optimizer,
        criterion: MultiSimilarityBCELoss,
        grad_clip: float = 1.0
    ) -> Dict[str, float]:
        """
        Train for one epoch using episodic sampling.
        
        Returns:
            Dictionary of epoch metrics
        """
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        # Similarity tracking per positive/negative
        pos_metrics = {'l1': [], 'l2': [], 'cosine': [], 'product': []}
        neg_metrics = {'l1': [], 'l2': [], 'cosine': [], 'product': []}
        attention_weights_all = []
        
        for batch_idx, (pairs, labels) in enumerate(sampler):
            # Construct batch
            x1_batch, x2_batch, label_batch = [], [], []
            
            for (idx1, idx2), label in zip(pairs, labels):
                x1, x2, lbl = dataset.get_pair(idx1, idx2, label)
                x1_batch.append(x1)
                x2_batch.append(x2)
                label_batch.append(lbl)
            
            x1 = torch.stack(x1_batch).to(self.device)
            x2 = torch.stack(x2_batch).to(self.device)
            labels_tensor = torch.stack(label_batch).to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            similarity, emb1, emb2, sim_metrics = self.model(x1, x2)
            
            # Compute loss
            loss, loss_metrics = criterion(similarity, labels_tensor, emb1, emb2)
            
            # Backward pass
            loss.backward()
            
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            preds = (similarity > 0.5).float().squeeze().cpu().numpy()
            labs = labels_tensor.squeeze().cpu().numpy()
            all_preds.extend(preds.tolist() if hasattr(preds, 'tolist') else [preds])
            all_labels.extend(labs.tolist() if hasattr(labs, 'tolist') else [labs])
            
            # Track similarity metrics per positive/negative pair
            with torch.no_grad():
                l1_dist = torch.abs(emb1 - emb2).sum(dim=1)
                l2_dist = torch.sqrt(((emb1 - emb2) ** 2).sum(dim=1) + 1e-8)
                cosine_sim = F.cosine_similarity(emb1, emb2, dim=1)
                product = (emb1 * emb2).sum(dim=1)
                
                pos_mask = labels_tensor.squeeze() == 1
                neg_mask = labels_tensor.squeeze() == 0
                
                if pos_mask.sum() > 0:
                    pos_metrics['l1'].extend(l1_dist[pos_mask].cpu().numpy().tolist())
                    pos_metrics['l2'].extend(l2_dist[pos_mask].cpu().numpy().tolist())
                    pos_metrics['cosine'].extend(cosine_sim[pos_mask].cpu().numpy().tolist())
                    pos_metrics['product'].extend(product[pos_mask].cpu().numpy().tolist())
                
                if neg_mask.sum() > 0:
                    neg_metrics['l1'].extend(l1_dist[neg_mask].cpu().numpy().tolist())
                    neg_metrics['l2'].extend(l2_dist[neg_mask].cpu().numpy().tolist())
                    neg_metrics['cosine'].extend(cosine_sim[neg_mask].cpu().numpy().tolist())
                    neg_metrics['product'].extend(product[neg_mask].cpu().numpy().tolist())
                
                if 'attention_weights' in sim_metrics:
                    attention_weights_all.append(sim_metrics['attention_weights'])
        
        # Compute epoch metrics
        avg_loss = total_loss / len(sampler)
        pair_acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds])
        pair_f1 = f1_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds], average='binary')
        
        return {
            'loss': avg_loss,
            'pair_acc': pair_acc,
            'pair_f1': pair_f1,
            'pos_l1_mean': np.mean(pos_metrics['l1']) if pos_metrics['l1'] else 0,
            'neg_l1_mean': np.mean(neg_metrics['l1']) if neg_metrics['l1'] else 0,
            'pos_l2_mean': np.mean(pos_metrics['l2']) if pos_metrics['l2'] else 0,
            'neg_l2_mean': np.mean(neg_metrics['l2']) if neg_metrics['l2'] else 0,
            'pos_cosine_mean': np.mean(pos_metrics['cosine']) if pos_metrics['cosine'] else 0,
            'neg_cosine_mean': np.mean(neg_metrics['cosine']) if neg_metrics['cosine'] else 0,
            'attention_weights': np.mean(attention_weights_all, axis=0) if attention_weights_all else None
        }
    
    @torch.no_grad()
    def evaluate_pairs(
        self,
        dataset: SiamesePairDataset,
        n_pairs: int = 500
    ) -> Dict[str, float]:
        """Evaluate on random pairs."""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        total_loss = 0.0
        
        criterion = nn.BCELoss()
        
        for _ in range(n_pairs):
            # Sample random pair
            if random.random() > 0.5:
                # Positive pair
                cls = random.choice(dataset.classes)
                indices = dataset.class_indices[cls]
                if len(indices) >= 2:
                    idx1, idx2 = random.sample(indices, 2)
                else:
                    idx1 = idx2 = indices[0]
                label = 1
            else:
                # Negative pair
                cls1, cls2 = random.sample(dataset.classes, 2)
                idx1 = random.choice(dataset.class_indices[cls1])
                idx2 = random.choice(dataset.class_indices[cls2])
                label = 0
            
            x1 = torch.FloatTensor(dataset.x[idx1]).unsqueeze(0).to(self.device)
            x2 = torch.FloatTensor(dataset.x[idx2]).unsqueeze(0).to(self.device)
            label_tensor = torch.FloatTensor([[label]]).to(self.device)
            
            similarity, _, _, _ = self.model(x1, x2)
            
            loss = criterion(similarity, label_tensor)
            total_loss += loss.item()
            
            pred = (similarity > 0.5).float().item()
            all_preds.append(pred)
            all_labels.append(label)
        
        pair_acc = accuracy_score(all_labels, all_preds)
        pair_f1 = f1_score(all_labels, all_preds, average='binary')
        
        return {
            'loss': total_loss / n_pairs,
            'pair_acc': pair_acc,
            'pair_f1': pair_f1
        }
    
    @torch.no_grad()
    def compute_prototypes(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> Tuple[torch.Tensor, List[int]]:
        """Compute class prototypes (mean embeddings)."""
        self.model.eval()
        
        classes = np.unique(y)
        prototypes = []
        prototype_labels = []
        
        for cls in classes:
            mask = y == cls
            x_cls = torch.FloatTensor(x[mask]).to(self.device)
            
            # Compute embeddings in batches
            embeddings = []
            batch_size = 256
            
            for i in range(0, len(x_cls), batch_size):
                batch = x_cls[i:i+batch_size]
                emb = self.model.forward_one(batch)
                embeddings.append(emb.cpu())
            
            embeddings = torch.cat(embeddings, dim=0)
            prototype = embeddings.mean(dim=0)
            
            prototypes.append(prototype)
            prototype_labels.append(cls)
        
        return torch.stack(prototypes).to(self.device), prototype_labels
    
    @torch.no_grad()
    def classify_with_prototypes(
        self,
        x: np.ndarray,
        prototypes: torch.Tensor,
        prototype_labels: List[int]
    ) -> np.ndarray:
        """Classify samples using nearest prototype."""
        self.model.eval()
        
        predictions = []
        batch_size = 256
        
        for i in range(0, len(x), batch_size):
            batch = torch.FloatTensor(x[i:i+batch_size]).to(self.device)
            embeddings = self.model.forward_one(batch)
            
            # L2 distances to prototypes
            distances = torch.cdist(embeddings, prototypes)
            
            # Nearest prototype
            nearest = distances.argmin(dim=1)
            batch_preds = [prototype_labels[n] for n in nearest.cpu().numpy()]
            predictions.extend(batch_preds)
        
        return np.array(predictions)
    
    @torch.no_grad()
    def evaluate_classification(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        class_names: List[str],
        regular_class_idx: Optional[int] = None
    ) -> Dict:
        """Evaluate classification using prototype-based classification.
        
        Args:
            x_train: Training data for building prototypes
            y_train: Training labels
            x_test: Test data
            y_test: Test labels
            class_names: List of class names
            regular_class_idx: Optional override for regular class index.
                             If None, uses self.regular_class_idx.
                             If provided, uses this value instead (important for multi-class evaluation).
        """
        from sklearn.metrics import classification_report, balanced_accuracy_score, confusion_matrix
        
        # Compute prototypes from training data
        prototypes, proto_labels = self.compute_prototypes(x_train, y_train)
        
        # Classify test samples
        predictions = self.classify_with_prototypes(x_test, prototypes, proto_labels)
        
        # Ensure predictions are numpy arrays (not tensors)
        if hasattr(predictions, 'cpu'):
            predictions = predictions.cpu().numpy()
        predictions = np.asarray(predictions)
        y_test = np.asarray(y_test)
        
        # Compute metrics
        accuracy = accuracy_score(y_test, predictions)
        balanced_acc = balanced_accuracy_score(y_test, predictions)
        macro_f1 = f1_score(y_test, predictions, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        
        # Binary anomaly detection F1 (regular vs others)
        # Use provided regular_class_idx if given, otherwise use stored value
        regular_idx = regular_class_idx if regular_class_idx is not None else self.regular_class_idx
        
        if regular_idx is not None:
            y_binary_true = (y_test != regular_idx).astype(int)
            y_binary_pred = (predictions != regular_idx).astype(int)
            anomaly_f1 = f1_score(y_binary_true, y_binary_pred, average='binary', zero_division=0)
        else:
            anomaly_f1 = 0.0
        
        report = classification_report(
            y_test, predictions,
            target_names=[str(c) for c in class_names],
            digits=4,
            zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'f1_macro': macro_f1,
            'f1_weighted': weighted_f1,
            'anomaly_f1': anomaly_f1,
            'confusion_matrix': cm,
            'predictions': predictions,
            'targets': y_test,
            'report': report,
            'prototypes': prototypes.cpu(),
            'prototype_labels': proto_labels
        }



