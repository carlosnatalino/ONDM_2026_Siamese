# %%
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
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Importing necessary modules for data loading and transformation
from data_loader import DASDataLoader, fft

logging.basicConfig(level=logging.INFO)

# %%
def get_dataset(data_dir):
    decim_dict = {
        # The 'regular' label will be decimated by a factor of 50
        'regular': 50,
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
    return x_normalized, Y, parser


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

class SimilarityDataset(Dataset):
    def __init__(self, x, y, pairs_per_sample=2, seed=42, augment=False):
        self.x = torch.FloatTensor(x)
        self.y = torch.LongTensor(y)
        self.pairs_per_sample = pairs_per_sample
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.augment = augment
        self.class_indices = {c: np.where(self.y == c)[0] for c in np.unique(self.y)}
        
    def __len__(self):
        if self.augment:
            return len(self.x) * self.pairs_per_sample * 2
        else:
            return len(self.x) * self.pairs_per_sample
    
    def _augment(self, sample):
        """Simple augmentation: noise injection and frequency masking"""
        if not self.augment:
            return sample
        
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

        if self.rng.rand() < 0.5:
            positive_idx = self.rng.choice(self.class_indices[anchor_label])
            pair = self.x[positive_idx]
            if self.augment:
                pair = self._augment(pair)
            similarity_label = 1.0
        else:
            # Randomly select a different class
            negative_classes = [c for c in self.class_indices.keys()
                               if c != anchor_label]
            negative_class = self.rng.choice(negative_classes)
            positive_idx = self.rng.choice(self.class_indices[negative_class])
            pair = self.x[positive_idx]
            if self.augment:
                pair = self._augment(pair)
            similarity_label = 0.0
    
        return anchor_sample, pair, torch.FloatTensor([similarity_label])


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
        # if x.dim() == 2:
        #     x = x.unsqueeze(-1)  # [batch, 2048] -> [batch, 2048, 1]
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        # Case 2: Input is (Batch, Length, 1) -> Permute to (Batch, 1, Length)
        # Some datasets put channels last; PyTorch needs channels second.
        elif x.dim() == 3 and x.shape[2] == 1:
            x = x.permute(0, 2, 1)
        return self.network(x)


class SimilarityHead(nn.Module):
    def __init__(self, embedding_dim):
        super(SimilarityHead, self).__init__()

        # Learnable weights for different similarity measures
        self.similarity_weights = nn.Parameter(torch.ones(embedding_dim + 3) / 4)  # Equal initial weights

        # Learnable transformation for each similarity measure
        self.l1_transform = nn.Linear(1, 1)
        self.l2_transform = nn.Linear(1, 1)
        self.element_wise_transform = nn.Linear(embedding_dim, 1)
        self.cosine_transform = nn.Linear(1, 1)

        # Final combination layer
        self.combination_layer = nn.Linear(embedding_dim + 3, 1)

    def forward(self, embedding_1, embedding_2):
        l1_distance = torch.sum(torch.abs(embedding_1 - embedding_2), dim=1, keepdim=True)

        l2_distance = F.pairwise_distance(embedding_1, embedding_2, keepdim=True)

        element_wise_product = embedding_1 * embedding_2

        cosine_similarity = F.cosine_similarity(embedding_1, embedding_2, dim=1).unsqueeze(1)

        combined_features = torch.cat([l1_distance, l2_distance, element_wise_product, cosine_similarity], dim=1)
        
        weighted_features = combined_features * F.softmax(self.similarity_weights, dim=0)
        
        # Final similarity score
        similarity = torch.sigmoid(self.combination_layer(weighted_features))
        
        return similarity

class SiameseClassifier:
    def __init__(self, input_dim=2048, embedding_dim=128, network='cnn', early_stopping_patience=25):
        self.device = get_device()
        if network == 'cnn':
            self.embedding_network = EmbeddingConvNetwork(input_dim, embedding_dim).to(self.device)
        elif network == 'mlp':
            self.embedding_network = EmbeddingNetwork(input_dim, embedding_dim).to(self.device)
        else:
            raise ValueError(f"Invalid network: {network}")
        self.similarity_head = SimilarityHead(embedding_dim).to(self.device)
        self.early_stopping_patience = early_stopping_patience
        self.criterion = nn.BCELoss()
        self.training_history = {
            'epochs': [],
            'train_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_loss': []
        }
    
    def forward(self, x1, x2):
        embedding_1 = self.embedding_network(x1)
        embedding_2 = self.embedding_network(x2)
        return self.similarity_head(embedding_1, embedding_2)

    def fit(
        self, 
        train_loader, 
        val_loader=None, 
        epochs=20, 
        lr=1e-3, 
        weight_decay=1e-4,
        args=None,
    ):

        self.embedding_network.train()
        self.similarity_head.train()

        optimizer = optim.Adam(
            list(self.embedding_network.parameters()) + list(self.similarity_head.parameters()),
            lr=lr, 
            weight_decay=weight_decay
        )

        best_val_acc = 0.0
        best_embedding_network_state = None
        best_similarity_head_state = None
        epochs_without_improvement = 0
        total_time = 0

        width = len(str(epochs))

        for epoch in range(1, epochs + 1):
            train_loss = 0.0
            correct = 0
            total = 0
            epoch_start_time = time.time()

            for batch_idx, (x1, x2, labels) in enumerate(train_loader):
                x1 = x1.to(self.device)
                x2 = x2.to(self.device)
                labels = labels.float().to(self.device).view(-1, 1)

                optimizer.zero_grad()
                outputs = self.forward(x1, x2)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * x1.size(0)
                preds = (outputs > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                # if log_interval and (batch_idx + 1) % log_interval == 0:
                #     print(f"Epoch [{epoch}/{epochs}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

            avg_loss = train_loss / total if total > 0 else 0
            train_acc = correct / total if total > 0 else 0
            epoch_time = time.time() - epoch_start_time
            total_time += epoch_time
            avg_epoch_time = total_time / (epoch + 1)
            estimated_remaining_time = avg_epoch_time * (epochs - epoch)
            estimated_time_str = datetime.datetime.fromtimestamp(estimated_remaining_time).strftime("%H:%M:%S")

            self.training_history['epochs'].append(epoch)
            self.training_history['train_loss'].append(avg_loss)
            self.training_history['train_acc'].append(train_acc)

            val_acc = None
            best = ""
            if val_loader is not None:
                val_acc, val_loss, _, _ = self.evaluate(val_loader)
                self.training_history['val_acc'].append(val_acc)
                self.training_history['val_loss'].append(val_loss)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_embedding_network_state = self.embedding_network.state_dict().copy()
                    best_similarity_head_state = self.similarity_head.state_dict().copy()
                    epochs_without_improvement = 0
                    best = " * BEST *"
                else:
                    epochs_without_improvement += 1

            print(f"Epoch {epoch:{width}d}: Train Loss={avg_loss:.4f} | Train Acc={train_acc:.4f}" +
                  (f" | Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f}" if val_acc is not None else '') + f" | Time={epoch_time:.2f} s | Estimated Time={estimated_time_str}", f" | {best}" if len(best) > 0 else '')
            
            if epochs_without_improvement >= self.early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs. No improvement for {self.early_stopping_patience} epochs.")
                break

        if best_embedding_network_state is not None and best_similarity_head_state is not None:
            # Save the best model checkpoint to file
            checkpoint = {
                'embedding_network_state_dict': best_embedding_network_state,
                'similarity_head_state_dict': best_similarity_head_state,
            }
            torch.save(checkpoint, f"{args.save_dir}/best_siamese_model_{args.embedding_dim}_{args.batch_size}_{args.augment}_{args.network}.pth")
            print(f"Best model state saved to '{args.save_dir}/best_siamese_model_{args.embedding_dim}_{args.batch_size}_{args.augment}_{args.network}.pth'")

    def evaluate(self, data_loader):
        self.embedding_network.eval()
        self.similarity_head.eval()
        correct = 0
        total = 0
        loss = 0.0
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            for x1, x2, labels in data_loader:
                x1 = x1.to(self.device)
                x2 = x2.to(self.device)
                labels = labels.float().to(self.device).view(-1, 1)
                outputs = self.forward(x1, x2)
                preds = (outputs > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                loss += self.criterion(outputs, labels).item() * labels.size(0)
                all_predictions.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
        acc = correct / total if total > 0 else 0
        loss = loss / total if total > 0 else 0
        self.embedding_network.train()
        self.similarity_head.train()
        return acc, loss, all_predictions, all_targets
    
    def load_best_model(self, args):
        checkpoint = torch.load(f"{args.save_dir}/best_siamese_model_{args.embedding_dim}_{args.batch_size}_{args.augment}_{args.network}.pth")
        self.embedding_network.load_state_dict(checkpoint['embedding_network_state_dict'])
        self.similarity_head.load_state_dict(checkpoint['similarity_head_state_dict'])
        self.embedding_network.to(self.device)
        self.similarity_head.to(self.device)
        print(f"Best model loaded")

    def classify(self, x, samples):
        x = x.to(self.device)
        samples = samples.to(self.device)
        _class = -1
        for i, sample in enumerate(samples):
            output = self.forward(x, sample)
            if output > 0.5:
                _class = i
                break
        return _class

def evaluate_one_shot(self, x, y, n_trials=100, n_classes=None, seed=42):
    """
    Evaluates one-shot classification accuracy.

    Args:
        x (np.ndarray): Feature matrix of shape (N, feature_dim)
        y (np.ndarray): Label vector of shape (N,)
        n_trials (int): Number of one-shot trials
        n_classes (int): How many unique classes to sample from (if None, uses all present in y)
        seed (int): Random state

    Returns:
        accuracy (float)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_classes is None:
        unique_classes = np.unique(y)
    else:
        unique_classes = np.random.choice(np.unique(y), n_classes, replace=False)

    correct = 0
    total = 0
    all_predictions = []
    all_targets = []

    for _ in range(n_trials):
        # Randomly choose n_classes for this trial
        trial_classes = np.random.choice(unique_classes, len(unique_classes), replace=False) if n_classes is None else \
            np.random.choice(unique_classes, n_classes, replace=False)

        # For each class, select one random reference sample
        reference_indices = []
        for c in trial_classes:
            idxs = np.where(y == c)[0]
            reference_idx = np.random.choice(idxs)
            reference_indices.append(reference_idx)
        references = x[reference_indices]
        references_torch = torch.tensor(references, dtype=torch.float32).to(self.device)
        
        # Now choose a query: pick a random class, and then a sample from that class (excluding the reference)
        cls_idx = np.random.choice(len(trial_classes))
        cls = trial_classes[cls_idx]
        candidate_idxs = np.where(y == cls)[0]
        candidate_idxs = [i for i in candidate_idxs if i != reference_indices[cls_idx]]
        if not candidate_idxs:  # rare case: only 1 example in class
            continue
        query_idx = np.random.choice(candidate_idxs)
        query = x[query_idx:query_idx+1]
        query_torch = torch.tensor(query, dtype=torch.float32).to(self.device)

        # Run classification
        self.embedding_network.eval()
        self.similarity_head.eval()
        with torch.no_grad():
            # Expand query to match shape for comparison with all references
            query_batch = query_torch.expand(references_torch.shape[0], -1)
            outputs = self.forward(query_batch, references_torch)  # shape: [n_classes, 1]
            pred = torch.argmax(outputs.view(-1)).item()
            if pred == cls_idx:
                correct += 1
            total += 1
            all_predictions.append(pred)
            all_targets.append(cls_idx)

    accuracy = correct / total if total > 0 else 0.0
    print(f"One-shot accuracy over {total} trials: {accuracy:.4f}")
    return accuracy, all_predictions, all_targets



def plot_training_history(model, args):
    epochs = model.training_history['epochs']
    train_loss = model.training_history['train_loss']
    train_acc = model.training_history['train_acc']
    val_acc = model.training_history['val_acc']
    val_loss = model.training_history['val_loss']

    # Create figure with subplots
    plt.figure()

    # Plot 1: Training Loss
    plt.plot(epochs, train_loss, 'g-', linewidth=2, label='Training Loss', marker='o', markersize=4)
    if val_loss:
        plt.plot(epochs, val_loss, 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{args.save_dir}/training_history_siamese_loss_{args.embedding_dim}_{args.batch_size}_{args.augment}_{args.network}.png')
    plt.close()

    plt.figure()
    # Plot 2: Training and Validation Accuracy
    plt.plot(epochs, train_acc, 'g-', linewidth=2, label='Training Accuracy', marker='o', markersize=4)
    if val_acc:
        plt.plot(epochs, val_acc, 'r-', linewidth=2, label='Validation Accuracy', marker='s', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{args.save_dir}/training_history_siamese_acc_{args.embedding_dim}_{args.batch_size}_{args.augment}_{args.network}.png')
    plt.close()


def main(args):
    x_normalized, Y, parser = get_dataset(args.data_dir)
    set_seed(args.seed)

    # [markdown]
    # 0: car             - 0.6290234501333418
    # 1: construction    - 0.4519333691399926
    # 2: fence           - 3.66179012345679
    # 3: longboard       - 1.1070242227447467
    # 4: manipulation    - 0.7653976917171482
    # 5: openclose       - 2.218146465496289
    # 6: regular         - 3.137930122457616
    # 7: running         - 1.0197605356574955
    # 8: walk            - 1.0368447730410921

    test_size_actual = args.test_size
    val_size_actual = args.val_size / (1 - test_size_actual)  # Adjust for two-stage split

    # Split dataset into train, validation, and test sets

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

    # Create SimilarityDataset instances for training and validation
    indices_selected = ( (Y_train == 6) | (Y_train == 8) | (Y_train == 0) | (Y_train == 2) | (Y_train == 4)).nonzero()[0]
    X_train_selected = X_train[indices_selected]
    Y_train_selected = Y_train[indices_selected]
    
    print("Training samples:", end="\t")
    counter = Counter(Y_train_selected)
    for key in range(len(parser.encoder.classes_)):
        if key in counter:
            value = counter[key]
        else:
            value = 0
        print(f"{key}: {value:<5}", end="\t")
    print()

    print("Validation samples:", end="\t")
    counter = Counter(Y_val)
    for key in range(len(parser.encoder.classes_)):
        if key in counter:
            value = counter[key]
        else:
            value = 0
        print(f"{key}: {value:<5}", end="\t")
    print()

    print("Test samples:      ", end="\t")
    counter = Counter(Y_test)
    for key in range(len(parser.encoder.classes_)):
        if key in counter:
            value = counter[key]
        else:
            value = 0
        print(f"{key}: {value:<5}", end="\t")
    print()

    train_dataset = SimilarityDataset(X_train_selected, Y_train_selected, pairs_per_sample=2, seed=42, augment=args.augment)
    val_dataset = SimilarityDataset(X_val, Y_val, pairs_per_sample=2, seed=42)
    test_dataset = SimilarityDataset(X_test, Y_test, pairs_per_sample=2, seed=42)

    # Create DataLoaders
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

    # Get input dimension from the data
    input_dim = x_normalized.shape[1]
    print(f"Input dimension: {input_dim}")

    # Initialize the Siamese Classifier
    model = SiameseClassifier(input_dim=input_dim, embedding_dim=args.embedding_dim, network=args.network, early_stopping_patience=args.early_stopping_patience)

    # Train the model
    model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=1e-3,
        weight_decay=1e-4,
        args=args,
    )

    print("\nTraining completed!")

    # Extract training history
    plot_training_history(model, args)

    labels = [name for name in parser.encoder.classes_]

    # Print summary statistics
    print("\n" + "="*50)
    print("Training Summary")
    print("="*50)
    print(f"Total Epochs: {len(model.training_history['epochs'])}")
    print(f"Final Training Loss: {model.training_history['train_loss'][-1]:.4f}")
    print(f"Final Training Accuracy: {model.training_history['train_acc'][-1]:.4f}")
    if model.training_history['val_acc']:
        print(f"Final Validation Accuracy: {model.training_history['val_acc'][-1]:.4f}")
        print(f"Best Validation Accuracy: {max(model.training_history['val_acc']):.4f} (Epoch {model.training_history['val_acc'].index(max(model.training_history['val_acc']))})")
    print("="*50)

    # Evaluate on test set
    model.load_best_model(args)
    test_acc, test_loss, test_predictions, test_targets = model.evaluate(test_loader)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Confusion Matrix: {confusion_matrix(test_targets, test_predictions)}")
    print(f"Test Classification Report: {classification_report(test_targets, test_predictions, labels=labels, target_names=parser.encoder.classes_)}")

    with open(f"{args.save_dir}/siamese_test_classification_report_{args.embedding_dim}_{args.batch_size}_{args.augment}_{args.network}.pkl", "wb") as f:
        pickle.dump(
            {
                'training_loss': model.training_history['train_loss'],
                'training_acc': model.training_history['train_acc'],
                'validation_acc': model.training_history['val_acc'],
                'classification_report': classification_report(test_targets, test_predictions, labels=labels, target_names=parser.encoder.classes_),
                'confusion_matrix': confusion_matrix(test_targets, test_predictions),
                'test_acc': test_acc,
                'test_loss': test_loss,
                'test_predictions': test_predictions,
                'test_targets': test_targets,
            },
            f
        )

    # Plot the confusion matrix for the training set
    train_acc, train_loss, train_predictions, train_targets = model.evaluate(train_loader)
    plot_confusion_matrix(
        confusion_matrix(train_targets, train_predictions),
        class_names=parser.encoder.classes_,
        title="Training Set Confusion Matrix",
        normalize=True,
        save_path=f"{args.save_dir}/siamese_train_confusion_matrix_{args.embedding_dim}_{args.batch_size}_{args.augment}_{args.network}"
    )

    # Plot the confusion matrix for the validation set
    val_acc, val_loss, val_predictions, val_targets = model.evaluate(val_loader)
    plot_confusion_matrix(
        confusion_matrix(val_targets, val_predictions),
        class_names=parser.encoder.classes_,
        title="Validation Set Confusion Matrix",
        normalize=True,
        save_path=f"{args.save_dir}/siamese_val_confusion_matrix_{args.embedding_dim}_{args.batch_size}_{args.augment}_{args.network}"
    )
    # Plot the confusion matrix for the test set
    plot_confusion_matrix(
        confusion_matrix(test_targets, test_predictions),
        class_names=parser.encoder.classes_,
        title="Test Set Confusion Matrix",
        normalize=True,
        save_path=f"{args.save_dir}/siamese_test_confusion_matrix_{args.embedding_dim}_{args.batch_size}_{args.augment}_{args.network}"
    )

    # Evaluate one-shot classification accuracy
    one_shot_acc, one_shot_predictions, one_shot_targets = evaluate_one_shot(model, X_test, Y_test, n_trials=100, n_classes=None, seed=42)
    print(f"One-shot accuracy: {one_shot_acc:.4f}")
    print(f"One-shot confusion matrix: {confusion_matrix(one_shot_targets, one_shot_predictions)}")
    print(f"One-shot classification report: {classification_report(one_shot_targets, one_shot_predictions, labels=labels, target_names=parser.encoder.classes_)}")
    plot_confusion_matrix(
        confusion_matrix(one_shot_targets, one_shot_predictions),
        class_names=parser.encoder.classes_,
        title="One-shot Set Confusion Matrix",
        normalize=True,
        save_path=f"{args.save_dir}/siamese_one_shot_confusion_matrix_{args.embedding_dim}_{args.batch_size}_{args.augment}_{args.network}"
    )


def plot_confusion_matrix(cm, class_names, title='Confusion Matrix', cmap=plt.cm.Blues, normalize=False, save_path=None):
    """
    Plots a confusion matrix using matplotlib.
    
    Args:
        cm (np.ndarray): Confusion matrix.
        class_names (list): List of class names.
        title (str): Title for the plot.
        cmap: Colormap.
        normalize (bool): Whether to normalize the matrix.
        save_path (str or None): Path to save the figure. If None, plt.show() is used.
    """
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--augment', type=bool, default=True)
    parser.add_argument('--network', type=str, default='cnn', choices=['cnn', 'mlp'])
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument(
        '--early_stopping_patience',
        type=int,
        default=25,
        help='Early stopping patience (epochs without improvement, default: 10)'
    )
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--val_size', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--data-dir', type=str, default='/nobackup/carda/datasets/DAS-dataset/data')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print("Arguments:", args)
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)
