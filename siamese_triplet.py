# %%
"""
Siamese Network with Triplet Loss and Prototypical Networks approach.

This implementation addresses the fundamental task mismatch between binary similarity
training and N-way K-shot classification by:
1. Using Triplet Loss with margin to enforce separation between classes
2. Implementing Prototypical Networks approach (class prototypes)
3. Using softmax over distances during training for proper ranking
4. Including comprehensive N-way K-shot and Open-Set Recognition evaluation
"""

import argparse
import time
from collections import Counter
import datetime
import logging
import random
import os
import pickle
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler

# Importing necessary modules for data loading and transformation
from data_loader import DASDataLoader, fft

logging.basicConfig(level=logging.INFO)


# %%
def get_dataset(data_dir):
    decim_dict = {
        # The 'regular' label will be decimated by a factor of 50
        # 'regular': 50,
        # 'fence': 50,
        # 'longboard': 50,
        # 'manipulation': 50,
        # 'openclose': 50,
        # 'running': 50,
        # 'walk': 50,
        # 'car': 50,
        # 'construction': 50,
    }

    # Initializing the DASDataLoader with dataset parameters
    parser = DASDataLoader(
        data_dir,  # Path to the dataset directory
        2048,  # Sample length
        transform=fft,  # Applying FFT as a preprocessing step
        fsize=8192,  # Window size for sliding window segmentation
        # Step size for the sliding window (overlap of 75% with fsize=8192)
        shift=2048,
        # Dictionary specifying the decimation factor for each label
        decimate=decim_dict,
    )


    # %%
    # Parsing the dataset into features (x) and labels (y)
    x, y = parser.parse_dataset()

    # Output parsed dataset details
    full_mean = np.mean(x, axis=0, keepdims=True)
    full_std = np.std(x, axis=0, keepdims=True) + 1e-8
    x_normalized = (x - full_mean) / full_std
    # x_normalized.shape
    Y = y.argmax(axis=1)
    return x_normalized, Y, parser, full_mean, full_std


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device():
    """Detect and return best available device (MPS > CUDA > CPU)"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


# =============================================================================
# Dataset for Triplet Loss Training
# =============================================================================

class TripletDataset(Dataset):
    """
    Dataset that generates triplets (anchor, positive, negative) for triplet loss training.
    """
    def __init__(self, x, y, triplets_per_sample=2, seed=42, augment=False):
        self.x = torch.FloatTensor(x)
        self.y = torch.LongTensor(y)
        self.triplets_per_sample = triplets_per_sample
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.augment = augment
        self.class_indices = {c.item(): np.where(self.y.numpy() == c.item())[0] 
                              for c in torch.unique(self.y)}
        self.classes = list(self.class_indices.keys())
        
    def __len__(self):
        return len(self.x) * self.triplets_per_sample
    
    def _augment(self, sample):
        """Simple augmentation: noise injection and frequency masking"""
        if not self.augment:
            return sample
        
        sample = sample.clone()
        # 50% chance to apply each augmentation
        if self.rng.rand() < 0.5:
            # Add Gaussian noise
            noise = torch.randn_like(sample) * 0.1
            sample = sample + noise
        
        if self.rng.rand() < 0.3:
            # Frequency masking
            mask_width = self.rng.randint(10, 100)
            mask_start = self.rng.randint(0, len(sample) - mask_width)
            sample[mask_start:mask_start + mask_width] = 0
            
        return sample
    
    def __getitem__(self, idx):
        anchor_idx = idx % len(self.x)
        anchor_sample = self.x[anchor_idx]
        anchor_label = self.y[anchor_idx].item()

        # Select positive sample (same class, different sample)
        positive_candidates = [i for i in self.class_indices[anchor_label] if i != anchor_idx]
        if len(positive_candidates) == 0:
            positive_idx = anchor_idx  # Fallback to same sample
        else:
            positive_idx = self.rng.choice(positive_candidates)
        positive_sample = self.x[positive_idx]
        
        # Select negative sample (different class)
        negative_classes = [c for c in self.classes if c != anchor_label]
        negative_class = self.rng.choice(negative_classes)
        negative_idx = self.rng.choice(self.class_indices[negative_class])
        negative_sample = self.x[negative_idx]
        
        if self.augment:
            anchor_sample = self._augment(anchor_sample)
            positive_sample = self._augment(positive_sample)
            negative_sample = self._augment(negative_sample)
    
        return anchor_sample, positive_sample, negative_sample, anchor_label


# =============================================================================
# Episodic Batch Sampler for Prototypical Networks
# =============================================================================

class EpisodicBatchSampler(Sampler):
    """
    Samples batches in an episodic manner for prototypical networks training.
    Each batch contains n_way classes with n_support + n_query samples per class.
    """
    def __init__(self, labels, n_way, n_support, n_query, n_episodes):
        self.labels = np.array(labels)
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.n_episodes = n_episodes
        
        self.classes = np.unique(self.labels)
        self.class_indices = {c: np.where(self.labels == c)[0] for c in self.classes}
        
        # Filter classes with enough samples
        self.valid_classes = [c for c in self.classes 
                              if len(self.class_indices[c]) >= n_support + n_query]
        
        if len(self.valid_classes) < n_way:
            print(f"Warning: Only {len(self.valid_classes)} classes have enough samples. Adjusting n_way.")
            self.n_way = len(self.valid_classes)
    
    def __iter__(self):
        for _ in range(self.n_episodes):
            # Sample n_way classes
            episode_classes = np.random.choice(self.valid_classes, self.n_way, replace=False)
            
            batch_indices = []
            for c in episode_classes:
                # Sample n_support + n_query samples from this class
                class_samples = np.random.choice(
                    self.class_indices[c], 
                    self.n_support + self.n_query, 
                    replace=False
                )
                batch_indices.extend(class_samples)
            
            yield batch_indices
    
    def __len__(self):
        return self.n_episodes


class PrototypicalDataset(Dataset):
    """Simple dataset wrapper for prototypical networks."""
    def __init__(self, x, y, augment=False, seed=42):
        self.x = torch.FloatTensor(x)
        self.y = torch.LongTensor(y)
        self.augment = augment
        self.rng = np.random.RandomState(seed)
        
    def __len__(self):
        return len(self.x)
    
    def _augment(self, sample):
        if not self.augment:
            return sample
        sample = sample.clone()
        if self.rng.rand() < 0.5:
            noise = torch.randn_like(sample) * 0.1
            sample = sample + noise
        if self.rng.rand() < 0.3:
            mask_width = self.rng.randint(10, 100)
            mask_start = self.rng.randint(0, len(sample) - mask_width)
            sample[mask_start:mask_start + mask_width] = 0
        return sample
    
    def __getitem__(self, idx):
        sample = self.x[idx]
        if self.augment:
            sample = self._augment(sample)
        return sample, self.y[idx]


# =============================================================================
# Embedding Networks (unchanged from siamese.py)
# =============================================================================

class EmbeddingNetwork(nn.Module):
    def __init__(self, input_dim=2048, embedding_dim=128):
        super(EmbeddingNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, embedding_dim),
        )
    
    def forward(self, x):
        return self.network(x)
    

class EmbeddingConvNetwork(nn.Module):
    def __init__(self, input_dim=2048, embedding_dim=128, dropout=0.5):
        super(EmbeddingConvNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.network = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=4),

            nn.Conv1d(64, 256, kernel_size=7, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=4),
            
            nn.Flatten(),
            nn.Linear(256 * 126, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),  # ADD: Dropout
            nn.Linear(1024, embedding_dim),
        )
    
    def forward(self, x):
        # Reshape input from [batch, features] to [batch, channels, length]
        # Conv1d expects [batch, channels, length], so we add a length dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        # Case 2: Input is (Batch, Length, 1) -> Permute to (Batch, 1, Length)
        # Some datasets put channels last; PyTorch needs channels second.
        elif x.dim() == 3 and x.shape[2] == 1:
            x = x.permute(0, 2, 1)
        return self.network(x)


# =============================================================================
# Loss Functions
# =============================================================================

class TripletLoss(nn.Module):
    """
    Triplet Loss with margin.
    L = max(0, d(a, p) - d(a, n) + margin)
    """
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


class PrototypicalLoss(nn.Module):
    """
    Prototypical Networks Loss.
    Uses softmax over negative squared distances to prototypes.
    This naturally provides ranking between classes.
    """
    def __init__(self, temperature=1.0):
        super(PrototypicalLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, support_embeddings, support_labels, query_embeddings, query_labels, n_way, n_support):
        """
        Args:
            support_embeddings: [n_way * n_support, embedding_dim]
            support_labels: [n_way * n_support]
            query_embeddings: [n_way * n_query, embedding_dim]
            query_labels: [n_way * n_query]
            n_way: number of classes
            n_support: number of support samples per class
        """
        # Compute prototypes (class centroids)
        unique_labels = torch.unique(support_labels)
        prototypes = []
        label_to_idx = {}
        
        for idx, label in enumerate(unique_labels):
            mask = support_labels == label
            prototype = support_embeddings[mask].mean(dim=0)
            prototypes.append(prototype)
            label_to_idx[label.item()] = idx
        
        prototypes = torch.stack(prototypes)  # [n_way, embedding_dim]
        
        # Compute distances from queries to prototypes
        # query_embeddings: [n_query, embedding_dim]
        # prototypes: [n_way, embedding_dim]
        dists = torch.cdist(query_embeddings, prototypes, p=2)  # [n_query, n_way]
        
        # Convert distances to log-probabilities using softmax over negative distances
        log_probs = F.log_softmax(-dists / self.temperature, dim=1)  # [n_query, n_way]
        
        # Map query labels to prototype indices
        target_indices = torch.tensor([label_to_idx[l.item()] for l in query_labels], 
                                       device=query_labels.device)
        
        # Cross-entropy loss
        loss = F.nll_loss(log_probs, target_indices)
        
        # Compute accuracy
        preds = log_probs.argmax(dim=1)
        acc = (preds == target_indices).float().mean()
        
        return loss, acc, prototypes


# =============================================================================
# Main Classifier with Triplet/Prototypical Training
# =============================================================================

class PrototypicalClassifier:
    """
    Classifier using Prototypical Networks approach with optional Triplet Loss pre-training.
    """
    def __init__(self, input_dim=2048, embedding_dim=128, network='cnn', 
                 early_stopping_patience=25, margin=1.0, temperature=1.0):
        self.device = get_device()
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.temperature = temperature
        
        if network == 'cnn':
            self.embedding_network = EmbeddingConvNetwork(input_dim, embedding_dim).to(self.device)
        elif network == 'mlp':
            self.embedding_network = EmbeddingNetwork(input_dim, embedding_dim).to(self.device)
        else:
            raise ValueError(f"Invalid network: {network}")
            
        self.early_stopping_patience = early_stopping_patience
        self.triplet_loss = TripletLoss(margin=margin)
        self.proto_loss = PrototypicalLoss(temperature=temperature)
        
        self.training_history = {
            'epochs': [],
            'train_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_loss': [],
            'triplet_loss': [],
            'proto_loss': [],
        }
    
    def compute_prototypes(self, x, y):
        """Compute class prototypes from samples."""
        self.embedding_network.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).to(self.device)
            embeddings = self.embedding_network(x_tensor)
            
            unique_labels = np.unique(y)
            prototypes = {}
            for label in unique_labels:
                mask = y == label
                class_embeddings = embeddings[mask]
                prototypes[label] = class_embeddings.mean(dim=0)
        
        return prototypes
    
    def forward_distance(self, x, prototypes_tensor):
        """Compute distances from samples to prototypes."""
        embeddings = self.embedding_network(x)
        # Compute L2 distances
        dists = torch.cdist(embeddings, prototypes_tensor, p=2)
        return dists
    
    def fit_prototypical(
        self, 
        train_dataset,
        val_dataset,
        n_way=5,
        n_support=5,
        n_query=15,
        n_episodes_train=100,
        n_episodes_val=50,
        epochs=100, 
        lr=1e-3, 
        weight_decay=1e-4,
        args=None,
    ):
        """
        Train using Prototypical Networks approach with episodic training.
        """
        self.embedding_network.train()
        
        optimizer = optim.Adam(
            self.embedding_network.parameters(),
            lr=lr, 
            weight_decay=weight_decay
        )
        
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        best_val_acc = 0.0
        best_state = None
        epochs_without_improvement = 0
        total_time = 0
        
        width = len(str(epochs))
        
        for epoch in range(1, epochs + 1):
            self.embedding_network.train()
            train_loss = 0.0
            train_acc = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            
            # Create episodic sampler for this epoch
            train_sampler = EpisodicBatchSampler(
                train_dataset.y.numpy(), 
                n_way=n_way, 
                n_support=n_support, 
                n_query=n_query,
                n_episodes=n_episodes_train
            )
            
            for batch_indices in train_sampler:
                # Get batch data
                batch_x = train_dataset.x[batch_indices].to(self.device)
                batch_y = train_dataset.y[batch_indices].to(self.device)
                
                # Split into support and query
                samples_per_class = n_support + n_query
                support_x = []
                support_y = []
                query_x = []
                query_y = []
                
                for i in range(n_way):
                    start_idx = i * samples_per_class
                    support_x.append(batch_x[start_idx:start_idx + n_support])
                    support_y.append(batch_y[start_idx:start_idx + n_support])
                    query_x.append(batch_x[start_idx + n_support:start_idx + samples_per_class])
                    query_y.append(batch_y[start_idx + n_support:start_idx + samples_per_class])
                
                support_x = torch.cat(support_x, dim=0)
                support_y = torch.cat(support_y, dim=0)
                query_x = torch.cat(query_x, dim=0)
                query_y = torch.cat(query_y, dim=0)
                
                # Forward pass
                optimizer.zero_grad()
                support_embeddings = self.embedding_network(support_x)
                query_embeddings = self.embedding_network(query_x)
                
                # Compute prototypical loss
                loss, acc, _ = self.proto_loss(
                    support_embeddings, support_y,
                    query_embeddings, query_y,
                    n_way, n_support
                )
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_acc += acc.item()
                n_batches += 1
            
            scheduler.step()
            
            avg_loss = train_loss / n_batches if n_batches > 0 else 0
            avg_acc = train_acc / n_batches if n_batches > 0 else 0
            epoch_time = time.time() - epoch_start_time
            total_time += epoch_time
            avg_epoch_time = total_time / epoch
            estimated_remaining_time = avg_epoch_time * (epochs - epoch)
            estimated_time_str = datetime.datetime.fromtimestamp(estimated_remaining_time).strftime("%H:%M:%S")
            
            self.training_history['epochs'].append(epoch)
            self.training_history['train_loss'].append(avg_loss)
            self.training_history['train_acc'].append(avg_acc)
            self.training_history['proto_loss'].append(avg_loss)
            
            # Validation
            val_acc = None
            val_loss = None
            best = ""
            if val_dataset is not None:
                val_acc, val_loss = self._validate_prototypical(
                    val_dataset, n_way, n_support, n_query, n_episodes_val
                )
                self.training_history['val_acc'].append(val_acc)
                self.training_history['val_loss'].append(val_loss)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = self.embedding_network.state_dict().copy()
                    epochs_without_improvement = 0
                    best = " * BEST *"
                else:
                    epochs_without_improvement += 1
            
            print(f"Epoch {epoch:{width}d}: Train Loss={avg_loss:.4f} | Train Acc={avg_acc:.4f}" +
                  (f" | Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f}" if val_acc is not None else '') + 
                  f" | Time={epoch_time:.2f}s | ETA={estimated_time_str}" + 
                  (f" {best}" if len(best) > 0 else ''))
            
            if epochs_without_improvement >= self.early_stopping_patience:
                print(f"Early stopping after {epoch} epochs. No improvement for {self.early_stopping_patience} epochs.")
                break
        
        if best_state is not None:
            checkpoint = {
                'embedding_network_state_dict': best_state,
                'embedding_dim': self.embedding_dim,
                'margin': self.margin,
                'temperature': self.temperature,
            }
            save_path = f"{args.save_dir}/best_prototypical_model_{args.embedding_dim}_{args.n_way}_{args.n_support}_{args.network}.pth"
            torch.save(checkpoint, save_path)
            print(f"Best model saved to '{save_path}'")
    
    def _validate_prototypical(self, val_dataset, n_way, n_support, n_query, n_episodes):
        """Validate using episodic evaluation."""
        self.embedding_network.eval()
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0
        
        val_sampler = EpisodicBatchSampler(
            val_dataset.y.numpy(),
            n_way=n_way,
            n_support=n_support,
            n_query=n_query,
            n_episodes=n_episodes
        )
        
        with torch.no_grad():
            for batch_indices in val_sampler:
                batch_x = val_dataset.x[batch_indices].to(self.device)
                batch_y = val_dataset.y[batch_indices].to(self.device)
                
                samples_per_class = n_support + n_query
                support_x = []
                support_y = []
                query_x = []
                query_y = []
                
                for i in range(n_way):
                    start_idx = i * samples_per_class
                    support_x.append(batch_x[start_idx:start_idx + n_support])
                    support_y.append(batch_y[start_idx:start_idx + n_support])
                    query_x.append(batch_x[start_idx + n_support:start_idx + samples_per_class])
                    query_y.append(batch_y[start_idx + n_support:start_idx + samples_per_class])
                
                support_x = torch.cat(support_x, dim=0)
                support_y = torch.cat(support_y, dim=0)
                query_x = torch.cat(query_x, dim=0)
                query_y = torch.cat(query_y, dim=0)
                
                support_embeddings = self.embedding_network(support_x)
                query_embeddings = self.embedding_network(query_x)
                
                loss, acc, _ = self.proto_loss(
                    support_embeddings, support_y,
                    query_embeddings, query_y,
                    n_way, n_support
                )
                
                total_loss += loss.item()
                total_acc += acc.item()
                n_batches += 1
        
        self.embedding_network.train()
        return total_acc / n_batches, total_loss / n_batches
    
    def fit_triplet(
        self, 
        train_loader, 
        val_loader=None, 
        epochs=50, 
        lr=1e-3, 
        weight_decay=1e-4,
        args=None,
    ):
        """
        Pre-train using Triplet Loss to learn good embeddings.
        """
        self.embedding_network.train()
        
        optimizer = optim.Adam(
            self.embedding_network.parameters(),
            lr=lr, 
            weight_decay=weight_decay
        )
        
        best_val_loss = float('inf')
        best_state = None
        epochs_without_improvement = 0
        total_time = 0
        
        width = len(str(epochs))
        
        for epoch in range(1, epochs + 1):
            train_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            
            for batch_idx, (anchor, positive, negative, labels) in enumerate(train_loader):
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)
                
                optimizer.zero_grad()
                
                anchor_emb = self.embedding_network(anchor)
                positive_emb = self.embedding_network(positive)
                negative_emb = self.embedding_network(negative)
                
                loss = self.triplet_loss(anchor_emb, positive_emb, negative_emb)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                n_batches += 1
            
            avg_loss = train_loss / n_batches if n_batches > 0 else 0
            epoch_time = time.time() - epoch_start_time
            total_time += epoch_time
            avg_epoch_time = total_time / epoch
            estimated_remaining_time = avg_epoch_time * (epochs - epoch)
            estimated_time_str = datetime.datetime.fromtimestamp(estimated_remaining_time).strftime("%H:%M:%S")
            
            self.training_history['triplet_loss'].append(avg_loss)
            
            # Validation
            val_loss = None
            best = ""
            if val_loader is not None:
                val_loss = self._validate_triplet(val_loader)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = self.embedding_network.state_dict().copy()
                    epochs_without_improvement = 0
                    best = " * BEST *"
                else:
                    epochs_without_improvement += 1
            
            print(f"[Triplet] Epoch {epoch:{width}d}: Train Loss={avg_loss:.4f}" +
                  (f" | Val Loss={val_loss:.4f}" if val_loss is not None else '') + 
                  f" | Time={epoch_time:.2f}s | ETA={estimated_time_str}" + 
                  (f" {best}" if len(best) > 0 else ''))
            
            if epochs_without_improvement >= self.early_stopping_patience // 2:
                print(f"Triplet pre-training stopped after {epoch} epochs.")
                break
        
        if best_state is not None:
            self.embedding_network.load_state_dict(best_state)
            print("Loaded best triplet pre-trained weights.")
    
    def _validate_triplet(self, val_loader):
        """Validate triplet loss."""
        self.embedding_network.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for anchor, positive, negative, _ in val_loader:
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)
                
                anchor_emb = self.embedding_network(anchor)
                positive_emb = self.embedding_network(positive)
                negative_emb = self.embedding_network(negative)
                
                loss = self.triplet_loss(anchor_emb, positive_emb, negative_emb)
                total_loss += loss.item()
                n_batches += 1
        
        self.embedding_network.train()
        return total_loss / n_batches
    
    def load_best_model(self, args):
        save_path = f"{args.save_dir}/best_prototypical_model_{args.embedding_dim}_{args.n_way}_{args.n_support}_{args.network}.pth"
        checkpoint = torch.load(save_path, map_location=self.device)
        self.embedding_network.load_state_dict(checkpoint['embedding_network_state_dict'])
        self.embedding_network.to(self.device)
        print(f"Best model loaded from {save_path}")


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_n_way_k_shot(model, x, y, n_way=5, k_shot=1, n_trials=1000, seed=42):
    """
    Evaluates N-way K-shot classification accuracy using prototypical approach.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    unique_classes = np.unique(y)
    class_indices = {c: np.where(y == c)[0] for c in unique_classes}
    
    # Filter classes with enough samples
    valid_classes = [c for c in unique_classes if len(class_indices[c]) >= k_shot + 1]
    
    if n_way > len(valid_classes):
        print(f"Warning: n_way ({n_way}) > valid classes ({len(valid_classes)}). Adjusting.")
        n_way = len(valid_classes)
    
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    all_confidences = []
    
    model.embedding_network.eval()
    
    with torch.no_grad():
        for trial in range(n_trials):
            # Sample n_way classes
            episode_classes = np.random.choice(valid_classes, n_way, replace=False)
            
            # Build support set
            support_samples = []
            support_labels = []
            
            for cls_idx, cls in enumerate(episode_classes):
                available = class_indices[cls].copy()
                np.random.shuffle(available)
                for idx in available[:k_shot]:
                    support_samples.append(x[idx])
                    support_labels.append(cls_idx)
            
            support_x = torch.tensor(np.array(support_samples), dtype=torch.float32).to(model.device)
            support_y = torch.tensor(support_labels, dtype=torch.long).to(model.device)
            
            # Compute prototypes
            support_embeddings = model.embedding_network(support_x)
            prototypes = []
            for cls_idx in range(n_way):
                mask = support_y == cls_idx
                prototype = support_embeddings[mask].mean(dim=0)
                prototypes.append(prototype)
            prototypes = torch.stack(prototypes)  # [n_way, embedding_dim]
            
            # Select query sample
            query_cls_idx = np.random.randint(0, n_way)
            query_cls = episode_classes[query_cls_idx]
            available_query = [i for i in class_indices[query_cls] 
                               if i not in class_indices[query_cls][:k_shot]]
            if len(available_query) == 0:
                continue
            
            query_idx = np.random.choice(available_query)
            query_x = torch.tensor(x[query_idx:query_idx+1], dtype=torch.float32).to(model.device)
            query_embedding = model.embedding_network(query_x)
            
            # Compute distances and predict
            dists = torch.cdist(query_embedding, prototypes, p=2)  # [1, n_way]
            probs = F.softmax(-dists / model.temperature, dim=1)
            pred = dists.argmin(dim=1).item()
            confidence = probs.max().item()
            
            if pred == query_cls_idx:
                correct += 1
            total += 1
            
            all_predictions.append(pred)
            all_targets.append(query_cls_idx)
            all_confidences.append(confidence)
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, all_predictions, all_targets, all_confidences


def evaluate_open_set_recognition(model, x, y, k_shot=5, n_trials=500, seed=42):
    """
    Evaluates Open Set Recognition (OSR) capability.
    Tests the model's ability to detect novel classes not in the support set.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    unique_classes = np.unique(y)
    class_indices = {c: np.where(y == c)[0] for c in unique_classes}
    
    known_distances = []
    unknown_distances = []
    per_class_results = {c: {'distances': [], 'correct': 0, 'total': 0} for c in unique_classes}
    
    model.embedding_network.eval()
    
    with torch.no_grad():
        for trial in range(n_trials):
            # Hold out one class as unknown
            unknown_class = np.random.choice(unique_classes)
            known_classes = [c for c in unique_classes if c != unknown_class]
            
            # Build support set from known classes
            support_samples = []
            support_labels = []
            
            for cls_idx, cls in enumerate(known_classes):
                available = class_indices[cls].copy()
                np.random.shuffle(available)
                if len(available) < k_shot + 1:
                    continue
                for idx in available[:k_shot]:
                    support_samples.append(x[idx])
                    support_labels.append(cls_idx)
            
            if len(support_samples) == 0:
                continue
            
            support_x = torch.tensor(np.array(support_samples), dtype=torch.float32).to(model.device)
            support_y = torch.tensor(support_labels, dtype=torch.long).to(model.device)
            
            # Compute prototypes for known classes
            support_embeddings = model.embedding_network(support_x)
            prototypes = []
            for cls_idx in range(len(known_classes)):
                mask = support_y == cls_idx
                if mask.sum() == 0:
                    continue
                prototype = support_embeddings[mask].mean(dim=0)
                prototypes.append(prototype)
            
            if len(prototypes) == 0:
                continue
            prototypes = torch.stack(prototypes)
            
            # Test with known class sample
            known_cls = np.random.choice(known_classes)
            available_known = [i for i in class_indices[known_cls] 
                               if i not in class_indices[known_cls][:k_shot]]
            if len(available_known) > 0:
                query_idx = np.random.choice(available_known)
                query_x = torch.tensor(x[query_idx:query_idx+1], dtype=torch.float32).to(model.device)
                query_embedding = model.embedding_network(query_x)
                
                dists = torch.cdist(query_embedding, prototypes, p=2)
                min_dist = dists.min().item()
                known_distances.append(min_dist)
            
            # Test with unknown class sample
            available_unknown = list(class_indices[unknown_class])
            if len(available_unknown) > 0:
                query_idx = np.random.choice(available_unknown)
                query_x = torch.tensor(x[query_idx:query_idx+1], dtype=torch.float32).to(model.device)
                query_embedding = model.embedding_network(query_x)
                
                dists = torch.cdist(query_embedding, prototypes, p=2)
                min_dist = dists.min().item()
                unknown_distances.append(min_dist)
                per_class_results[unknown_class]['distances'].append(min_dist)
                per_class_results[unknown_class]['total'] += 1
    
    # Compute metrics
    # Use distance threshold - unknown samples should have larger distances
    if len(known_distances) > 0 and len(unknown_distances) > 0:
        # Find optimal threshold
        all_dists = known_distances + unknown_distances
        labels = [0] * len(known_distances) + [1] * len(unknown_distances)  # 0=known, 1=unknown
        
        # AUROC: higher distance -> more likely unknown
        auroc = roc_auc_score(labels, all_dists)
        
        # Find threshold at various percentiles
        thresholds = np.percentile(known_distances, [50, 75, 90, 95])
        results_per_threshold = {}
        
        for thresh in thresholds:
            known_correct = sum(1 for d in known_distances if d <= thresh)
            unknown_correct = sum(1 for d in unknown_distances if d > thresh)
            known_acc = known_correct / len(known_distances)
            unknown_acc = unknown_correct / len(unknown_distances)
            balanced_acc = (known_acc + unknown_acc) / 2
            results_per_threshold[thresh] = {
                'known_acc': known_acc,
                'unknown_acc': unknown_acc,
                'balanced_acc': balanced_acc
            }
    else:
        auroc = 0.5
        results_per_threshold = {}
    
    results = {
        'known_distances': known_distances,
        'unknown_distances': unknown_distances,
        'auroc': auroc,
        'per_threshold': results_per_threshold,
        'per_class': per_class_results,
        'mean_known_dist': np.mean(known_distances) if known_distances else 0,
        'mean_unknown_dist': np.mean(unknown_distances) if unknown_distances else 0,
        'std_known_dist': np.std(known_distances) if known_distances else 0,
        'std_unknown_dist': np.std(unknown_distances) if unknown_distances else 0,
    }
    
    return results


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_training_history(model, args):
    """Plot training history and save to files."""
    epochs = model.training_history['epochs']
    train_loss = model.training_history['train_loss']
    train_acc = model.training_history['train_acc']
    val_acc = model.training_history['val_acc']
    val_loss = model.training_history['val_loss']
    
    prefix = f"{args.save_dir}/prototypical_{args.embedding_dim}_{args.n_way}_{args.n_support}_{args.network}"
    
    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss', marker='o', markersize=3)
    if val_loss:
        plt.plot(epochs, val_loss, 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=3)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Prototypical Networks Training Loss', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, ls=":")
    plt.tight_layout()
    plt.savefig(f'{prefix}_loss.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{prefix}_loss.pdf', bbox_inches='tight')
    plt.close()
    
    # Plot Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc, 'b-', linewidth=2, label='Training Accuracy', marker='o', markersize=3)
    if val_acc:
        plt.plot(epochs, val_acc, 'r-', linewidth=2, label='Validation Accuracy', marker='s', markersize=3)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Prototypical Networks Training Accuracy', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, ls=":")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f'{prefix}_accuracy.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{prefix}_accuracy.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Training plots saved to {prefix}_*.png/pdf")


def plot_n_way_k_shot_results(results, class_names, args):
    """Plot N-way K-shot evaluation results."""
    prefix = f"{args.save_dir}/prototypical_{args.embedding_dim}_{args.n_way}_{args.n_support}_{args.network}"
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart of accuracies
    configs = list(results.keys())
    accuracies = [results[c]['accuracy'] * 100 for c in configs]
    config_labels = [f"{c[0]}-way\n{c[1]}-shot" for c in configs]
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(configs)))
    bars = axes[0].bar(config_labels, accuracies, color=colors)
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_xlabel('Configuration', fontsize=12)
    axes[0].set_title('N-way K-shot Classification Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 100)
    
    # Add chance levels
    for i, (n_way, k_shot) in enumerate(configs):
        chance = 100.0 / n_way
        axes[0].axhline(y=chance, color=colors[i], linestyle='--', alpha=0.3)
    
    for bar, acc in zip(bars, accuracies):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                     f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    axes[0].grid(axis='y', alpha=0.3)
    
    # Confidence distribution
    all_confidences = []
    all_correct = []
    for config, data in results.items():
        preds = data['predictions']
        targets = data['targets']
        confs = data['confidences']
        for p, t, c in zip(preds, targets, confs):
            all_confidences.append(c)
            all_correct.append(1 if p == t else 0)
    
    correct_confs = [c for c, corr in zip(all_confidences, all_correct) if corr]
    incorrect_confs = [c for c, corr in zip(all_confidences, all_correct) if not corr]
    
    axes[1].hist(correct_confs, bins=30, alpha=0.7, label='Correct', color='green', density=True)
    axes[1].hist(incorrect_confs, bins=30, alpha=0.7, label='Incorrect', color='red', density=True)
    axes[1].set_xlabel('Confidence Score', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    axes[1].set_title('Confidence Distribution', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{prefix}_nway_kshot.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{prefix}_nway_kshot.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"N-way K-shot plots saved to {prefix}_nway_kshot.png/pdf")


def plot_osr_results(osr_results, class_names, args):
    """Plot Open Set Recognition results."""
    prefix = f"{args.save_dir}/prototypical_{args.embedding_dim}_{args.n_way}_{args.n_support}_{args.network}"
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Distance distribution
    ax1 = axes[0, 0]
    ax1.hist(osr_results['known_distances'], bins=30, alpha=0.7, label='Known Classes', 
             color='#2ecc71', density=True)
    ax1.hist(osr_results['unknown_distances'], bins=30, alpha=0.7, label='Unknown Classes', 
             color='#e74c3c', density=True)
    ax1.set_xlabel('Min Distance to Prototypes', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Distance Distribution: Known vs Unknown', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ROC Curve
    ax2 = axes[0, 1]
    labels = [0] * len(osr_results['known_distances']) + [1] * len(osr_results['unknown_distances'])
    scores = osr_results['known_distances'] + osr_results['unknown_distances']
    
    if len(set(labels)) > 1:
        fpr, tpr, _ = roc_curve(labels, scores)
        ax2.plot(fpr, tpr, color='#3498db', linewidth=2, 
                 label=f'ROC (AUROC = {osr_results["auroc"]:.3f})')
        ax2.fill_between(fpr, tpr, alpha=0.2, color='#3498db')
    ax2.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1, label='Random')
    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.set_title('ROC Curve for Open Set Recognition', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    
    # Threshold analysis
    ax3 = axes[1, 0]
    if osr_results['per_threshold']:
        thresholds = sorted(osr_results['per_threshold'].keys())
        known_accs = [osr_results['per_threshold'][t]['known_acc'] * 100 for t in thresholds]
        unknown_accs = [osr_results['per_threshold'][t]['unknown_acc'] * 100 for t in thresholds]
        balanced_accs = [osr_results['per_threshold'][t]['balanced_acc'] * 100 for t in thresholds]
        
        ax3.plot(thresholds, known_accs, 'o-', color='#2ecc71', linewidth=2, label='Known Acc', markersize=6)
        ax3.plot(thresholds, unknown_accs, 's-', color='#e74c3c', linewidth=2, label='Unknown Acc', markersize=6)
        ax3.plot(thresholds, balanced_accs, '^-', color='#3498db', linewidth=2, label='Balanced Acc', markersize=6)
    ax3.set_xlabel('Distance Threshold', fontsize=12)
    ax3.set_ylabel('Accuracy (%)', fontsize=12)
    ax3.set_title('Threshold Analysis', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 100)
    
    # Summary stats
    ax4 = axes[1, 1]
    ax4.axis('off')
    summary_text = f"""
    Open Set Recognition Summary
    ============================
    
    AUROC: {osr_results['auroc']:.4f}
    
    Known Class Distances:
      Mean: {osr_results['mean_known_dist']:.4f}
      Std:  {osr_results['std_known_dist']:.4f}
    
    Unknown Class Distances:
      Mean: {osr_results['mean_unknown_dist']:.4f}
      Std:  {osr_results['std_unknown_dist']:.4f}
    
    Separation: {osr_results['mean_unknown_dist'] - osr_results['mean_known_dist']:.4f}
    """
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{prefix}_osr.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{prefix}_osr.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"OSR plots saved to {prefix}_osr.png/pdf")


def plot_confusion_matrix(cm, class_names, title='Confusion Matrix', cmap=plt.cm.Blues, 
                          normalize=False, save_path=None):
    """Plots a confusion matrix using matplotlib."""
    if normalize:
        cm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=15, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right', fontsize=11)
    plt.yticks(tick_marks, class_names, fontsize=11)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=11, fontweight='bold')
    plt.ylabel('True Label', fontsize=13)
    plt.xlabel('Predicted Label', fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}.pdf", bbox_inches='tight', dpi=150)
        plt.savefig(f"{save_path}.png", bbox_inches='tight', dpi=150)
        print(f"Confusion matrix plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


# =============================================================================
# Main Function
# =============================================================================

def main(args):
    # Load and preprocess dataset
    x_normalized, Y, parser, full_mean, full_std = get_dataset(args.data_dir)
    set_seed(args.seed)
    
    # Save normalization statistics for later use
    np.save(f"{args.save_dir}/train_mean.npy", full_mean)
    np.save(f"{args.save_dir}/train_std.npy", full_std)
    print(f"Normalization statistics saved to {args.save_dir}/train_mean.npy and train_std.npy")
    
    test_size_actual = args.test_size
    val_size_actual = args.val_size / (1 - test_size_actual)
    
    # Split dataset
    X_temp, X_test, Y_temp, Y_test = train_test_split(
        x_normalized, Y,
        test_size=test_size_actual,
        random_state=args.seed,
        stratify=Y
    )
    
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_temp, Y_temp,
        test_size=val_size_actual,
        random_state=args.seed,
        stratify=Y_temp
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Print class distribution
    print("\nClass distribution:")
    for split_name, split_y in [("Train", Y_train), ("Val", Y_val), ("Test", Y_test)]:
        print(f"{split_name}:", end="\t")
        counter = Counter(split_y)
        for key in range(len(parser.encoder.classes_)):
            value = counter.get(key, 0)
            print(f"{key}: {value:<5}", end="\t")
        print()
    
    # Create datasets
    train_dataset = PrototypicalDataset(X_train, Y_train, augment=args.augment, seed=args.seed)
    val_dataset = PrototypicalDataset(X_val, Y_val, augment=False, seed=args.seed)
    test_dataset = PrototypicalDataset(X_test, Y_test, augment=False, seed=args.seed)
    
    # Get input dimension
    input_dim = x_normalized.shape[1]
    print(f"\nInput dimension: {input_dim}")
    
    # Initialize model
    model = PrototypicalClassifier(
        input_dim=input_dim, 
        embedding_dim=args.embedding_dim, 
        network=args.network, 
        early_stopping_patience=args.early_stopping_patience,
        margin=args.margin,
        temperature=args.temperature
    )
    
    # Optional: Triplet loss pre-training
    if args.triplet_pretrain_epochs > 0:
        print("\n" + "="*60)
        print("Phase 1: Triplet Loss Pre-training")
        print("="*60)
        
        triplet_train = TripletDataset(X_train, Y_train, triplets_per_sample=2, 
                                        seed=args.seed, augment=args.augment)
        triplet_val = TripletDataset(X_val, Y_val, triplets_per_sample=1, seed=args.seed)
        
        triplet_train_loader = DataLoader(
            triplet_train, batch_size=args.batch_size, shuffle=True,
            num_workers=4, pin_memory=torch.cuda.is_available()
        )
        triplet_val_loader = DataLoader(
            triplet_val, batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=torch.cuda.is_available()
        )
        
        model.fit_triplet(
            triplet_train_loader,
            triplet_val_loader,
            epochs=args.triplet_pretrain_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            args=args
        )
    
    # Prototypical training
    print("\n" + "="*60)
    print("Phase 2: Prototypical Networks Training")
    print("="*60)
    
    model.fit_prototypical(
        train_dataset,
        val_dataset,
        n_way=args.n_way,
        n_support=args.n_support,
        n_query=args.n_query,
        n_episodes_train=args.n_episodes_train,
        n_episodes_val=args.n_episodes_val,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        args=args
    )
    
    print("\nTraining completed!")
    
    # Plot training history
    plot_training_history(model, args)
    
    # Print summary
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    print(f"Total Epochs: {len(model.training_history['epochs'])}")
    print(f"Final Training Loss: {model.training_history['train_loss'][-1]:.4f}")
    print(f"Final Training Accuracy: {model.training_history['train_acc'][-1]:.4f}")
    if model.training_history['val_acc']:
        print(f"Final Validation Accuracy: {model.training_history['val_acc'][-1]:.4f}")
        best_idx = np.argmax(model.training_history['val_acc'])
        print(f"Best Validation Accuracy: {model.training_history['val_acc'][best_idx]:.4f} (Epoch {best_idx + 1})")
    print("="*60)
    
    # Load best model for evaluation
    model.load_best_model(args)
    
    # ==========================================================================
    # N-way K-shot Evaluation
    # ==========================================================================
    print("\n" + "="*60)
    print("N-way K-shot Evaluation")
    print("="*60)
    
    n_way_k_shot_configs = [
        (5, 1),   # 5-way 1-shot
        (5, 5),   # 5-way 5-shot
        (9, 1),   # 9-way 1-shot (all classes)
        (9, 5),   # 9-way 5-shot (all classes)
    ]
    
    nway_results = {}
    class_names = list(parser.encoder.classes_)
    
    for n_way, k_shot in n_way_k_shot_configs:
        print(f"\nEvaluating {n_way}-way {k_shot}-shot...")
        acc, preds, targets, confs = evaluate_n_way_k_shot(
            model, X_test, Y_test, 
            n_way=n_way, k_shot=k_shot, 
            n_trials=args.n_eval_trials, 
            seed=args.seed
        )
        
        nway_results[(n_way, k_shot)] = {
            'accuracy': acc,
            'predictions': preds,
            'targets': targets,
            'confidences': confs
        }
        
        chance = 100.0 / n_way
        print(f"  {n_way}-way {k_shot}-shot Accuracy: {acc*100:.2f}% (chance: {chance:.1f}%)")
    
    # Plot N-way K-shot results
    plot_n_way_k_shot_results(nway_results, class_names, args)
    
    # ==========================================================================
    # Open Set Recognition Evaluation
    # ==========================================================================
    print("\n" + "="*60)
    print("Open Set Recognition Evaluation")
    print("="*60)
    
    osr_results = evaluate_open_set_recognition(
        model, X_test, Y_test,
        k_shot=args.n_support,
        n_trials=args.n_osr_trials,
        seed=args.seed
    )
    
    print(f"\nAUROC: {osr_results['auroc']:.4f}")
    print(f"Known distances: {osr_results['mean_known_dist']:.4f} ± {osr_results['std_known_dist']:.4f}")
    print(f"Unknown distances: {osr_results['mean_unknown_dist']:.4f} ± {osr_results['std_unknown_dist']:.4f}")
    print(f"Separation: {osr_results['mean_unknown_dist'] - osr_results['mean_known_dist']:.4f}")
    
    if osr_results['per_threshold']:
        print("\nThreshold Analysis:")
        for thresh, metrics in osr_results['per_threshold'].items():
            print(f"  Threshold {thresh:.3f}: Known={metrics['known_acc']*100:.1f}%, "
                  f"Unknown={metrics['unknown_acc']*100:.1f}%, Balanced={metrics['balanced_acc']*100:.1f}%")
    
    # Plot OSR results
    plot_osr_results(osr_results, class_names, args)
    
    # ==========================================================================
    # Save all results
    # ==========================================================================
    prefix = f"{args.save_dir}/prototypical_{args.embedding_dim}_{args.n_way}_{args.n_support}_{args.network}"
    
    all_results = {
        'training_history': model.training_history,
        'nway_kshot_results': nway_results,
        'osr_results': osr_results,
        'args': vars(args),
        'class_names': class_names,
    }
    
    with open(f"{prefix}_results.pkl", "wb") as f:
        pickle.dump(all_results, f)
    
    print(f"\nAll results saved to {prefix}_results.pkl")
    
    # Final summary table
    print("\n" + "="*60)
    print("Final Results Summary")
    print("="*60)
    print(f"{'Configuration':<20} {'Accuracy':>12} {'vs Chance':>12}")
    print("-"*44)
    for (n_way, k_shot), data in nway_results.items():
        chance = 100.0 / n_way
        acc = data['accuracy'] * 100
        improvement = acc - chance
        print(f"{n_way}-way {k_shot}-shot{'':<8} {acc:>10.2f}% {improvement:>+10.2f}%")
    print("-"*44)
    print(f"{'OSR AUROC':<20} {osr_results['auroc']:>12.4f}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prototypical Networks with Triplet Loss')
    
    # Model architecture
    parser.add_argument('--embedding_dim', type=int, default=512,
                        help='Embedding dimension')
    parser.add_argument('--network', type=str, default='cnn', choices=['cnn', 'mlp'],
                        help='Network architecture')
    
    # Prototypical Networks parameters
    parser.add_argument('--n_way', type=int, default=5,
                        help='Number of classes per episode')
    parser.add_argument('--n_support', type=int, default=5,
                        help='Number of support samples per class')
    parser.add_argument('--n_query', type=int, default=15,
                        help='Number of query samples per class')
    parser.add_argument('--n_episodes_train', type=int, default=100,
                        help='Number of training episodes per epoch')
    parser.add_argument('--n_episodes_val', type=int, default=50,
                        help='Number of validation episodes per epoch')
    
    # Loss parameters
    parser.add_argument('--margin', type=float, default=1.0,
                        help='Margin for triplet loss')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for softmax in prototypical loss')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for triplet pre-training')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs for prototypical training')
    parser.add_argument('--triplet_pretrain_epochs', type=int, default=0,
                        help='Number of epochs for triplet pre-training (0 to skip)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--early_stopping_patience', type=int, default=25,
                        help='Early stopping patience')
    parser.add_argument('--augment', action='store_true', default=True,
                        help='Use data augmentation')
    
    # Evaluation parameters
    parser.add_argument('--n_eval_trials', type=int, default=1000,
                        help='Number of trials for N-way K-shot evaluation')
    parser.add_argument('--n_osr_trials', type=int, default=500,
                        help='Number of trials for OSR evaluation')
    
    # Data parameters
    parser.add_argument('--data-dir', type=str, default='/nobackup/carda/datasets/DAS-dataset/data',
                        help='Path to dataset')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set size ratio')
    parser.add_argument('--val_size', type=float, default=0.2,
                        help='Validation set size ratio')
    
    # Other
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save models and results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    print("Arguments:", args)
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)

