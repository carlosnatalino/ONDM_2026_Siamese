#!/usr/bin/env python3
"""
Multi-Similarity Siamese Network - Evaluation Components

This module implements comprehensive evaluation including:
- N-way K-shot episodic classification
- Novelty detection with threshold and statistical approaches
- Real-world deployment simulation
- Comprehensive metrics tracking

References:
- Snell et al., "Prototypical Networks for Few-shot Learning" (2017)
- Vinyals et al., "Matching Networks for One Shot Learning" (2016)
- Bendale & Boult, "Towards Open Set Deep Networks" (2016)

Author: Andrei Ribeiro, Carlos Natalino
Date: January 2026
"""

import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    balanced_accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, average_precision_score
)

logger = logging.getLogger(__name__)


# =============================================================================
# N-way K-shot Evaluation
# =============================================================================

@torch.no_grad()
def evaluate_nway_kshot(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    n_way: int = 5,
    k_shot: int = 5,
    n_query: int = 10,
    n_episodes: int = 100,
    device: torch.device = torch.device('cpu')
) -> Dict[str, float]:
    """
    Evaluate N-way K-shot classification accuracy using episodic evaluation.
    
    Protocol (following Vinyals et al., 2016; Snell et al., 2017):
    1. Sample n_way classes for the episode
    2. Sample k_shot support samples per class (to build prototypes)
    3. Sample n_query query samples per class (to evaluate)
    4. Classify queries by nearest prototype in embedding space
    
    Args:
        model: Trained Siamese network
        x: Test features (N, feature_dim)
        y: Test labels (N,)
        n_way: Number of classes per episode
        k_shot: Number of support samples per class
        n_query: Number of query samples per class
        n_episodes: Number of evaluation episodes
        device: Torch device
        
    Returns:
        Dictionary with accuracy and F1 scores
    """
    model.eval()
    
    classes = np.unique(y)
    class_indices = {c: np.where(y == c)[0] for c in classes}
    
    # Filter classes with sufficient samples
    min_samples = k_shot + n_query
    valid_classes = [c for c in classes if len(class_indices[c]) >= min_samples]
    
    if len(valid_classes) < n_way:
        logger.warning(f"Only {len(valid_classes)} classes have enough samples, using all of them")
        n_way = len(valid_classes)
    
    if n_way < 2:
        logger.warning("Insufficient classes for evaluation")
        return {'accuracy': 0.0, 'f1_macro': 0.0, 'f1_weighted': 0.0}
    
    all_accuracies = []
    all_predictions = []
    all_true_labels = []
    
    for episode in range(n_episodes):
        # Sample classes for this episode
        episode_classes = np.random.choice(valid_classes, n_way, replace=False)
        
        support_embeddings = []
        query_embeddings = []
        query_labels = []
        
        for cls_idx, cls in enumerate(episode_classes):
            # Sample support and query indices
            indices = np.random.choice(
                class_indices[cls],
                k_shot + n_query,
                replace=False
            )
            
            support_idx = indices[:k_shot]
            query_idx = indices[k_shot:]
            
            # Compute support embeddings and prototype
            support_x = torch.FloatTensor(x[support_idx]).to(device)
            support_emb = model.forward_one(support_x)
            prototype = support_emb.mean(dim=0)  # Class prototype
            support_embeddings.append(prototype)
            
            # Compute query embeddings
            query_x = torch.FloatTensor(x[query_idx]).to(device)
            query_emb = model.forward_one(query_x)
            query_embeddings.append(query_emb)
            query_labels.extend([cls_idx] * n_query)
        
        # Stack prototypes and queries
        prototypes = torch.stack(support_embeddings)  # (n_way, emb_dim)
        queries = torch.cat(query_embeddings, dim=0)  # (n_way * n_query, emb_dim)
        query_labels = torch.LongTensor(query_labels).to(device)
        
        # Classify by nearest prototype (L2 distance)
        distances = torch.cdist(queries, prototypes)
        predictions = distances.argmin(dim=1)
        
        # Compute episode accuracy
        acc = (predictions == query_labels).float().mean().item()
        all_accuracies.append(acc)
        
        # Collect for F1 computation
        all_predictions.extend(predictions.cpu().numpy().tolist())
        all_true_labels.extend(query_labels.cpu().numpy().tolist())
    
    # Compute aggregate metrics
    mean_acc = np.mean(all_accuracies)
    std_acc = np.std(all_accuracies)
    f1_macro = f1_score(all_true_labels, all_predictions, average='macro')
    f1_weighted = f1_score(all_true_labels, all_predictions, average='weighted')
    
    return {
        'accuracy': mean_acc,
        'accuracy_std': std_acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'n_way': n_way,
        'k_shot': k_shot,
        'n_episodes': n_episodes
    }


def run_comprehensive_nway_kshot_eval(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    n_ways: List[int] = [5, 9],
    k_shots: List[int] = [1, 5, 10],
    n_episodes: int = 100
) -> Dict[str, Dict]:
    """
    Run comprehensive N-way K-shot evaluation across multiple configurations.
    
    Args:
        model: Trained model
        x: Test features
        y: Test labels
        device: Torch device
        n_ways: List of N values
        k_shots: List of K values
        n_episodes: Episodes per configuration
        
    Returns:
        Dictionary mapping configuration names to results
    """
    results = {}
    
    n_classes = len(np.unique(y))
    logger.info(f"Running N-way K-shot evaluation (total classes: {n_classes})")
    
    for n_way in n_ways:
        if n_way > n_classes:
            logger.warning(f"Skipping {n_way}-way: only {n_classes} classes available")
            continue
            
        for k_shot in k_shots:
            config_name = f"{n_way}way_{k_shot}shot"
            
            logger.info(f"  Evaluating {n_way}-way {k_shot}-shot...")
            
            result = evaluate_nway_kshot(
                model=model,
                x=x,
                y=y,
                n_way=n_way,
                k_shot=k_shot,
                n_query=min(10, len(x) // (n_way * 2)),
                n_episodes=n_episodes,
                device=device
            )
            
            results[config_name] = result
            
            logger.info(
                f"    Accuracy: {result['accuracy']:.4f} ± {result['accuracy_std']:.4f} | "
                f"F1 Macro: {result['f1_macro']:.4f}"
            )
    
    return results


# =============================================================================
# Novelty Detection
# =============================================================================

class NoveltyDetector:
    """
    Novelty detection using embedding distances.
    
    Implements two approaches:
    1. Fixed Threshold: Flag as novel if distance > threshold
    2. Statistical: Flag as novel if distance > mean + k*std of known distances
    
    Reference: Bendale & Boult, "Towards Open Set Deep Networks" (2016)
    
    Args:
        model: Trained embedding model
        device: Torch device
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device
    ):
        self.model = model
        self.device = device
        
        self.prototypes = None
        self.prototype_labels = None
        self.known_distances = None
        self.threshold_fixed = None
        self.threshold_statistical = None
    
    @torch.no_grad()
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        statistical_k: float = 2.0
    ):
        """
        Fit the detector on known class data.
        
        Computes:
        - Class prototypes
        - Distribution of distances for known samples
        - Threshold values
        
        Args:
            x: Training features
            y: Training labels
            statistical_k: Number of standard deviations for statistical threshold
        """
        self.model.eval()
        
        # Compute class prototypes
        classes = np.unique(y)
        prototypes = []
        prototype_labels = []
        
        for cls in classes:
            mask = y == cls
            x_cls = torch.FloatTensor(x[mask]).to(self.device)
            
            embeddings = []
            batch_size = 256
            
            for i in range(0, len(x_cls), batch_size):
                batch = x_cls[i:i+batch_size]
                emb = self.model.forward_one(batch)
                embeddings.append(emb)
            
            embeddings = torch.cat(embeddings, dim=0)
            prototype = embeddings.mean(dim=0)
            
            prototypes.append(prototype)
            prototype_labels.append(cls)
        
        self.prototypes = torch.stack(prototypes).to(self.device)
        self.prototype_labels = prototype_labels
        
        # Compute distances of all known samples to their nearest prototype
        all_embeddings = []
        batch_size = 256
        
        for i in range(0, len(x), batch_size):
            batch = torch.FloatTensor(x[i:i+batch_size]).to(self.device)
            emb = self.model.forward_one(batch)
            all_embeddings.append(emb)
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        # Distance to nearest prototype
        distances = torch.cdist(all_embeddings, self.prototypes)
        min_distances = distances.min(dim=1)[0].cpu().numpy()
        
        self.known_distances = min_distances
        
        # Compute thresholds
        mean_dist = np.mean(min_distances)
        std_dist = np.std(min_distances)
        
        # Statistical threshold: mean + k * std
        self.threshold_statistical = mean_dist + statistical_k * std_dist
        
        # Fixed threshold: 95th percentile of known distances
        self.threshold_fixed = np.percentile(min_distances, 95)
        
        logger.info(f"NoveltyDetector fitted:")
        logger.info(f"  Known classes: {len(classes)}")
        logger.info(f"  Mean distance: {mean_dist:.4f}")
        logger.info(f"  Std distance: {std_dist:.4f}")
        logger.info(f"  Fixed threshold (95%ile): {self.threshold_fixed:.4f}")
        logger.info(f"  Statistical threshold (mean + {statistical_k}*std): {self.threshold_statistical:.4f}")
    
    @torch.no_grad()
    def detect(
        self,
        x: np.ndarray,
        method: str = 'both'
    ) -> Dict[str, np.ndarray]:
        """
        Detect novel samples.
        
        Args:
            x: Test features
            method: 'fixed', 'statistical', or 'both'
            
        Returns:
            Dictionary with:
            - predictions: Nearest prototype class predictions
            - distances: Distance to nearest prototype
            - novel_fixed: Boolean mask for fixed threshold
            - novel_statistical: Boolean mask for statistical threshold
        """
        self.model.eval()
        
        # Compute embeddings
        all_embeddings = []
        batch_size = 256
        
        for i in range(0, len(x), batch_size):
            batch = torch.FloatTensor(x[i:i+batch_size]).to(self.device)
            emb = self.model.forward_one(batch)
            all_embeddings.append(emb)
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        # Compute distances to prototypes
        distances = torch.cdist(all_embeddings, self.prototypes)
        min_distances, min_indices = distances.min(dim=1)
        
        min_distances = min_distances.cpu().numpy()
        min_indices = min_indices.cpu().numpy()
        
        # Predictions
        predictions = np.array([self.prototype_labels[i] for i in min_indices])
        
        # Novelty detection
        novel_fixed = min_distances > self.threshold_fixed
        novel_statistical = min_distances > self.threshold_statistical
        
        return {
            'predictions': predictions,
            'distances': min_distances,
            'novel_fixed': novel_fixed,
            'novel_statistical': novel_statistical
        }
    
    def evaluate_novelty_detection(
        self,
        x_known: np.ndarray,
        x_novel: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate novelty detection performance.
        
        Args:
            x_known: Samples from known classes (should NOT be detected as novel)
            x_novel: Samples from unknown classes (should be detected as novel)
            
        Returns:
            Dictionary of metrics for both threshold methods
        """
        results_known = self.detect(x_known)
        results_novel = self.detect(x_novel)
        
        metrics = {}
        
        for method in ['fixed', 'statistical']:
            # True labels: 0 = known, 1 = novel
            y_true = np.concatenate([
                np.zeros(len(x_known)),
                np.ones(len(x_novel))
            ])
            
            # Predictions
            y_pred = np.concatenate([
                results_known[f'novel_{method}'].astype(int),
                results_novel[f'novel_{method}'].astype(int)
            ])
            
            # Compute metrics
            metrics[f'{method}_precision'] = precision_score(y_true, y_pred, zero_division=0)
            metrics[f'{method}_recall'] = recall_score(y_true, y_pred, zero_division=0)
            metrics[f'{method}_f1'] = f1_score(y_true, y_pred, average='binary', zero_division=0)
            metrics[f'{method}_accuracy'] = accuracy_score(y_true, y_pred)
            
            # Detection rates
            metrics[f'{method}_true_novel_rate'] = results_novel[f'novel_{method}'].mean()
            metrics[f'{method}_false_novel_rate'] = results_known[f'novel_{method}'].mean()
        
        # Distance statistics
        metrics['known_distance_mean'] = results_known['distances'].mean()
        metrics['novel_distance_mean'] = results_novel['distances'].mean()
        metrics['distance_separation'] = results_novel['distances'].mean() - results_known['distances'].mean()
        
        return metrics


# =============================================================================
# Real-World Deployment Simulation
# =============================================================================

class RealWorldSimulator:
    """
    Simulates real-world incremental deployment scenario.
    
    Scenario (as described in Tomasov et al., 2023):
    1. System starts with only "regular" (baseline) class
    2. Any deviation is initially flagged as "anomaly"
    3. As operators identify new event types, add them to known pool
    4. System should:
       - Detect anomalies (regular vs everything else)
       - Flag unknown class types (novel event types)
       - Correctly classify known event types
    
    This simulates real DAS perimeter protection deployment.
    
    Args:
        model: Trained embedding model
        device: Torch device
        regular_class_idx: Index of "regular" class
        regular_class_name: Name of "regular" class
        class_names: List of all class names
        output_dir: Directory for saving results
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        regular_class_idx: int,
        regular_class_name: str,
        class_names: List[str],
        output_dir: str
    ):
        self.model = model
        self.device = device
        self.regular_class_idx = regular_class_idx
        self.regular_class_name = regular_class_name
        self.class_names = list(class_names)
        self.output_dir = output_dir
        
        # Anomaly classes (all non-regular)
        self.anomaly_classes = [
            i for i, name in enumerate(self.class_names) 
            if i != self.regular_class_idx
        ]
        
        logger.info(f"RealWorldSimulator initialized:")
        logger.info(f"  Regular class: {regular_class_name} (idx={regular_class_idx})")
        logger.info(f"  Anomaly classes: {[self.class_names[i] for i in self.anomaly_classes]}")
    
    @torch.no_grad()
    def compute_prototype(self, x: np.ndarray) -> torch.Tensor:
        """Compute prototype from samples."""
        self.model.eval()
        
        embeddings = []
        batch_size = 256
        
        for i in range(0, len(x), batch_size):
            batch = torch.FloatTensor(x[i:i+batch_size]).to(self.device)
            emb = self.model.forward_one(batch)
            embeddings.append(emb.cpu())
        
        return torch.cat(embeddings, dim=0).mean(dim=0)
    
    @torch.no_grad()
    def classify_with_pool(
        self,
        x: np.ndarray,
        pool: Dict[int, torch.Tensor]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Classify using current prototype pool."""
        self.model.eval()
        
        pool_classes = list(pool.keys())
        prototypes = torch.stack([pool[c] for c in pool_classes]).to(self.device)
        
        all_embeddings = []
        batch_size = 256
        
        for i in range(0, len(x), batch_size):
            batch = torch.FloatTensor(x[i:i+batch_size]).to(self.device)
            emb = self.model.forward_one(batch)
            all_embeddings.append(emb)
        
        queries = torch.cat(all_embeddings, dim=0)
        
        distances = torch.cdist(queries, prototypes)
        min_distances, min_indices = distances.min(dim=1)
        
        predictions = np.array([pool_classes[i] for i in min_indices.cpu().numpy()])
        
        return predictions, min_distances.cpu().numpy()
    
    def simulate_incremental(
        self,
        x_support: np.ndarray,
        y_support: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        k_shots: List[int] = [1, 5, 10],
        class_order: List[str] = None
    ) -> Dict[int, List[Dict]]:
        """
        Run incremental deployment simulation.
        
        Args:
            x_support: Support samples for building prototypes
            y_support: Support labels
            x_test: Test samples
            y_test: Test labels
            k_shots: List of k-shot values to test
            class_order: Order to introduce classes (None = regular first, then random)
            
        Returns:
            Results for each k-shot configuration
        """
        logger.info("=" * 60)
        logger.info("Real-World Incremental Deployment Simulation")
        logger.info("=" * 60)
        
        # Define class introduction order
        if class_order is None:
            class_order = [self.regular_class_name]
            anomaly_names = [self.class_names[i] for i in self.anomaly_classes]
            np.random.shuffle(anomaly_names)
            class_order.extend(anomaly_names)
        
        # Filter to classes in dataset
        class_order = [c for c in class_order if c in self.class_names]
        logger.info(f"Class introduction order: {class_order}")
        
        results = {k: [] for k in k_shots}
        
        for k in k_shots:
            logger.info(f"\n--- {k}-shot simulation ---")
            
            known_pool = {}  # class_idx -> prototype
            
            for step, class_name in enumerate(class_order):
                class_idx = self.class_names.index(class_name)
                
                # Get k samples for prototype
                mask = y_support == class_idx
                class_samples = x_support[mask]
                
                if len(class_samples) < k:
                    logger.warning(f"  Skipping {class_name}: only {len(class_samples)} samples")
                    continue
                
                # Sample k shots and compute prototype
                shot_idx = np.random.choice(len(class_samples), min(k, len(class_samples)), replace=False)
                k_shots_samples = class_samples[shot_idx]
                
                prototype = self.compute_prototype(k_shots_samples)
                known_pool[class_idx] = prototype
                
                known_names = [self.class_names[i] for i in known_pool.keys()]
                logger.info(f"  Step {step+1}: Added '{class_name}'. Pool: {known_names}")
                
                # Evaluate current state
                metrics = self._evaluate_step(
                    x_test, y_test, known_pool, step, class_name
                )
                
                results[k].append(metrics)
                
                logger.info(
                    f"    Anomaly F1: {metrics['anomaly_f1']:.4f} | "
                    f"Known Acc: {metrics['known_acc']:.4f} | "
                    f"Unknown Det (fixed): {metrics['unknown_det_fixed']:.4f} | "
                    f"Unknown Det (stat): {metrics['unknown_det_stat']:.4f}"
                )
        
        return results
    
    def _evaluate_step(
        self,
        x_test: np.ndarray,
        y_test: np.ndarray,
        pool: Dict[int, torch.Tensor],
        step: int,
        new_class: str
    ) -> Dict:
        """Evaluate at one step of incremental deployment."""
        known_classes = set(pool.keys())
        unknown_classes = set(range(len(self.class_names))) - known_classes
        
        # Separate known and unknown samples
        known_mask = np.isin(y_test, list(known_classes))
        unknown_mask = np.isin(y_test, list(unknown_classes))
        
        x_known = x_test[known_mask]
        y_known = y_test[known_mask]
        x_unknown = x_test[unknown_mask]
        
        # Classify known samples
        if len(x_known) > 0:
            pred_known, dist_known = self.classify_with_pool(x_known, pool)
            known_acc = accuracy_score(y_known, pred_known)
        else:
            pred_known, dist_known = np.array([]), np.array([])
            known_acc = 0.0
        
        # Anomaly detection: regular vs others
        all_preds, all_dists = self.classify_with_pool(x_test, pool)
        
        y_binary_true = (y_test != self.regular_class_idx).astype(int)
        y_binary_pred = (all_preds != self.regular_class_idx).astype(int)
        
        anomaly_f1 = f1_score(y_binary_true, y_binary_pred, average='binary', zero_division=0)
        anomaly_prec = precision_score(y_binary_true, y_binary_pred, zero_division=0)
        anomaly_rec = recall_score(y_binary_true, y_binary_pred, zero_division=0)
        
        # Unknown detection using distance thresholds
        unknown_det_fixed = 0.0
        unknown_det_stat = 0.0
        mean_known_dist = 0.0
        mean_unknown_dist = 0.0
        
        if len(x_unknown) > 0 and len(dist_known) > 0:
            _, dist_unknown = self.classify_with_pool(x_unknown, pool)
            
            mean_known_dist = np.mean(dist_known)
            mean_unknown_dist = np.mean(dist_unknown)
            std_known_dist = np.std(dist_known)
            
            # Fixed threshold: 95th percentile
            threshold_fixed = np.percentile(dist_known, 95)
            unknown_det_fixed = (dist_unknown > threshold_fixed).mean()
            
            # Statistical threshold: mean + 2*std
            threshold_stat = mean_known_dist + 2 * std_known_dist
            unknown_det_stat = (dist_unknown > threshold_stat).mean()
        
        return {
            'step': step,
            'new_class': new_class,
            'n_known_classes': len(known_classes),
            'known_classes': list(known_classes),
            'anomaly_f1': anomaly_f1,
            'anomaly_precision': anomaly_prec,
            'anomaly_recall': anomaly_rec,
            'known_acc': known_acc,
            'unknown_det_fixed': unknown_det_fixed,
            'unknown_det_stat': unknown_det_stat,
            'mean_known_dist': mean_known_dist,
            'mean_unknown_dist': mean_unknown_dist
        }


# =============================================================================
# Comprehensive Evaluation Runner
# =============================================================================

def run_full_evaluation(
    model: torch.nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    device: torch.device,
    regular_class_name: str = 'regular',
    output_dir: str = '.'
) -> Dict:
    """
    Run comprehensive evaluation suite.
    
    Includes:
    - Standard classification metrics
    - N-way K-shot evaluation (5-way and 9-way with 1, 5, 10 shots)
    - Novelty detection evaluation
    - Real-world simulation
    
    Args:
        model: Trained model
        x_train, y_train: Training data (for prototypes)
        x_test, y_test: Test data
        class_names: List of class names
        device: Torch device
        regular_class_name: Name of normal class
        output_dir: Output directory
        
    Returns:
        Comprehensive results dictionary
    """
    logger.info("=" * 60)
    logger.info("Running Comprehensive Evaluation Suite")
    logger.info("=" * 60)
    
    results = {}
    
    regular_idx = class_names.index(regular_class_name) if regular_class_name in class_names else 0
    
    # 1. N-way K-shot evaluation
    logger.info("\n1. N-way K-shot Evaluation")
    logger.info("-" * 40)
    
    nway_results = run_comprehensive_nway_kshot_eval(
        model=model,
        x=x_test,
        y=y_test,
        device=device,
        n_ways=[5, 9],
        k_shots=[1, 5, 10],
        n_episodes=100
    )
    results['nway_kshot'] = nway_results
    
    # 2. Novelty detection
    logger.info("\n2. Novelty Detection Evaluation")
    logger.info("-" * 40)
    
    # Use training classes for known, leave one out for novel
    unique_classes = np.unique(y_test)
    if len(unique_classes) > 3:
        # Leave out some classes for novelty testing
        known_classes = unique_classes[:-2]
        novel_classes = unique_classes[-2:]
        
        x_train_known = x_train[np.isin(y_train, known_classes)]
        y_train_known = y_train[np.isin(y_train, known_classes)]
        x_test_known = x_test[np.isin(y_test, known_classes)]
        x_test_novel = x_test[np.isin(y_test, novel_classes)]
        
        if len(x_test_known) > 0 and len(x_test_novel) > 0:
            detector = NoveltyDetector(model, device)
            detector.fit(x_train_known, y_train_known)
            
            novelty_results = detector.evaluate_novelty_detection(x_test_known, x_test_novel)
            results['novelty_detection'] = novelty_results
            
            logger.info(f"  Fixed threshold - F1: {novelty_results['fixed_f1']:.4f}, "
                       f"True Novel Rate: {novelty_results['fixed_true_novel_rate']:.4f}")
            logger.info(f"  Statistical threshold - F1: {novelty_results['statistical_f1']:.4f}, "
                       f"True Novel Rate: {novelty_results['statistical_true_novel_rate']:.4f}")
    
    # 3. Real-world simulation
    logger.info("\n3. Real-World Deployment Simulation")
    logger.info("-" * 40)
    
    simulator = RealWorldSimulator(
        model=model,
        device=device,
        regular_class_idx=regular_idx,
        regular_class_name=regular_class_name,
        class_names=class_names,
        output_dir=output_dir
    )
    
    simulation_results = simulator.simulate_incremental(
        x_support=x_train,
        y_support=y_train,
        x_test=x_test,
        y_test=y_test,
        k_shots=[1, 5, 10]
    )
    results['simulation'] = simulation_results
    
    return results



