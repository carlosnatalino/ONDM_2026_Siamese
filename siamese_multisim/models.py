#!/usr/bin/env python3
"""
Multi-Similarity Siamese Network - Model Architectures

This module implements the embedding network and multi-similarity comparison head
for the Siamese network. It includes 5 different similarity metrics:
1. L1 Distance (Manhattan) - Chopra et al., 2005
2. L2 Distance (Euclidean) - Bromley et al., 1993
3. Cosine Similarity - invariant to magnitude
4. Element-wise Product - captures interaction patterns
5. Learned/Attention-weighted combination - learnable metric fusion

References:
- Bromley et al., "Signature Verification using a Siamese Time Delay Neural Network" (1993)
- Chopra et al., "Learning a Similarity Metric Discriminatively" (2005)
- Koch et al., "Siamese Neural Networks for One-shot Image Recognition" (2015)
- Snell et al., "Prototypical Networks for Few-shot Learning" (2017)

Author: Andrei Ribeiro, Carlos Natalino
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class EmbeddingNetwork(nn.Module):
    """
    1D CNN encoder for FFT-transformed DAS signals.
    
    Architecture based on Tomasov et al. (2023) "Enhancing Perimeter Protection 
    using Φ-OTDR and CNN for Event Classification":
    - Convolutional layers with 64 and 256 filters
    - LeakyReLU activation (Mastromichalakis, 2020)
    - MaxPooling for dimensionality reduction
    - Dense layers for final embedding
    
    The network learns to map 2048-dim FFT features to a compact
    embedding space where similar signals are close together.
    
    Args:
        input_dim: Input feature dimension (default: 2048 for FFT features)
        embedding_dim: Output embedding dimension
        dropout: Dropout rate for regularization
    """
    
    def __init__(
        self,
        input_dim: int = 2048,
        embedding_dim: int = 128,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # Convolutional encoder following Tomasov et al. (2023) architecture
        # Modified for 1D signals with additional regularization
        self.conv_layers = nn.Sequential(
            # Block 1: Initial feature extraction (64 filters as in reference)
            nn.Conv1d(1, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),  # LeakyReLU as per reference
            nn.MaxPool1d(4),    # Pool size of 4 as in reference
            nn.Dropout(dropout * 0.5),
            
            # Block 2: Intermediate features
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(4),
            nn.Dropout(dropout * 0.5),
            
            # Block 3: High-level features (256 filters as in reference)
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),
        )
        
        # Calculate flattened size after convolutions
        # Input: 2048 -> /4 -> 512 -> /4 -> 128 -> /4 -> 32
        self.flat_size = 256 * (input_dim // 64)  # 256 * 32 = 8192
        
        # Fully connected layers for embedding projection
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flat_size, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            
            nn.Linear(512, embedding_dim),
        )
        
        # Whether to L2-normalize embeddings
        self.normalize = True
        
        logger.info(f"EmbeddingNetwork initialized:")
        logger.info(f"  Input dim: {input_dim}")
        logger.info(f"  Embedding dim: {embedding_dim}")
        logger.info(f"  Flat size after conv: {self.flat_size}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute embeddings.
        
        Args:
            x: Input tensor (batch, features) or (batch, 1, features)
            
        Returns:
            embeddings: L2-normalized embeddings (batch, embedding_dim)
        """
        # Add channel dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, features)
        
        # Convolutional encoding
        x = self.conv_layers(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected projection
        embeddings = self.fc_layers(x)
        
        # L2 normalize for cosine similarity compatibility
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings


class AttentionSimilarityFusion(nn.Module):
    """
    Learnable attention-weighted fusion of multiple similarity metrics.
    
    This module learns to weight different similarity metrics based on
    the input embeddings, allowing the model to adaptively combine
    L1, L2, cosine, and product similarities.
    
    The attention mechanism is inspired by:
    - Vaswani et al., "Attention is All You Need" (2017)
    - Wang et al., "Multi-Similarity Loss with General Pair Weighting" (2019)
    
    Args:
        embedding_dim: Dimension of input embeddings
        n_metrics: Number of similarity metrics to fuse (default: 4)
    """
    
    def __init__(self, embedding_dim: int = 128, n_metrics: int = 4):
        super().__init__()
        
        self.n_metrics = n_metrics
        
        # Attention network: learns to weight each metric based on embeddings
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, n_metrics),
            nn.Softmax(dim=1)  # Attention weights sum to 1
        )
        
        logger.info(f"AttentionSimilarityFusion initialized with {n_metrics} metrics")
    
    def forward(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        similarities: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attention-weighted fusion of similarities.
        
        Args:
            emb1: First embeddings (batch, embedding_dim)
            emb2: Second embeddings (batch, embedding_dim)
            similarities: Stacked similarities (batch, n_metrics)
            
        Returns:
            fused: Attention-weighted similarity (batch, 1)
        """
        # Concatenate embeddings for attention computation
        combined = torch.cat([emb1, emb2], dim=1)
        
        # Compute attention weights
        attention_weights = self.attention(combined)  # (batch, n_metrics)
        
        # Weighted sum of similarities
        fused = (attention_weights * similarities).sum(dim=1, keepdim=True)
        
        return fused, attention_weights


class MultiSimilaritySiameseNetwork(nn.Module):
    """
    Multi-Similarity Siamese Network with 5 different similarity metrics.
    
    This network computes pairwise similarity using multiple complementary
    metrics, each capturing different aspects of embedding similarity:
    
    1. L1 Distance (Manhattan): Robust to outliers, sparse differences
       Reference: Chopra et al. (2005)
       
    2. L2 Distance (Euclidean): Standard metric, penalizes large differences
       Reference: Bromley et al. (1993)
       
    3. Cosine Similarity: Direction-based, magnitude-invariant
       Commonly used in NLP and metric learning
       
    4. Element-wise Product: Captures element-wise correlations
       Reference: Mou et al., "Natural Language Inference" (2016)
       
    5. Learned Attention-weighted Combination: Adaptive metric fusion
       Reference: Inspired by Vaswani et al. (2017)
    
    Args:
        input_dim: Input feature dimension
        embedding_dim: Embedding space dimension
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        input_dim: int = 2048,
        embedding_dim: int = 128,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Shared embedding network (Siamese weight sharing)
        self.embedding_net = EmbeddingNetwork(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            dropout=dropout
        )
        
        # Attention-based similarity fusion
        self.attention_fusion = AttentionSimilarityFusion(
            embedding_dim=embedding_dim,
            n_metrics=4  # L1, L2, cosine, product
        )
        
        # Multi-metric comparison head
        # Input: concat(e1, e2) + L1 + L2 + product + cosine + attention_fused
        # Dimensions: 2*emb + emb + emb + emb + 1 + 1 = 5*emb + 2
        comparison_dim = 5 * embedding_dim + 2
        
        self.comparison_head = nn.Sequential(
            nn.Linear(comparison_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout * 0.5),
            
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        logger.info("=" * 60)
        logger.info("MultiSimilaritySiameseNetwork initialized")
        logger.info("=" * 60)
        logger.info(f"  Embedding dim: {embedding_dim}")
        logger.info(f"  Comparison input dim: {comparison_dim}")
        logger.info("  Similarity metrics:")
        logger.info("    1. L1 Distance (Manhattan)")
        logger.info("    2. L2 Distance (Euclidean)")
        logger.info("    3. Cosine Similarity")
        logger.info("    4. Element-wise Product")
        logger.info("    5. Learned Attention-weighted Fusion")
    
    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """Get embedding for a single input."""
        return self.embedding_net(x)
    
    def compute_individual_similarities(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all individual similarity metrics between embedding pairs.
        
        Args:
            emb1: First embeddings (batch, embedding_dim)
            emb2: Second embeddings (batch, embedding_dim)
            
        Returns:
            Dictionary containing all similarity metrics
        """
        # 1. L1 Distance (Manhattan): per-dimension absolute difference
        # Lower values = more similar
        l1_dist = torch.abs(emb1 - emb2)  # (batch, emb_dim)
        l1_scalar = l1_dist.sum(dim=1, keepdim=True)  # (batch, 1)
        
        # 2. L2 Distance (Euclidean): per-dimension squared difference
        # Lower values = more similar
        l2_dist = (emb1 - emb2) ** 2  # (batch, emb_dim)
        l2_scalar = torch.sqrt(l2_dist.sum(dim=1, keepdim=True) + 1e-8)  # (batch, 1)
        
        # 3. Cosine Similarity: directional similarity [-1, 1]
        # Higher values = more similar
        cosine_sim = F.cosine_similarity(emb1, emb2, dim=1, eps=1e-8)
        cosine_sim = cosine_sim.unsqueeze(1)  # (batch, 1)
        
        # 4. Element-wise Product: captures correlation patterns
        product = emb1 * emb2  # (batch, emb_dim)
        product_scalar = product.sum(dim=1, keepdim=True)  # (batch, 1)
        
        return {
            'l1_vector': l1_dist,
            'l1_scalar': l1_scalar,
            'l2_vector': l2_dist,
            'l2_scalar': l2_scalar,
            'cosine': cosine_sim,
            'product_vector': product,
            'product_scalar': product_scalar
        }
    
    def compute_similarity_features(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute comprehensive similarity features for the comparison head.
        
        Args:
            emb1: First embeddings (batch, embedding_dim)
            emb2: Second embeddings (batch, embedding_dim)
            
        Returns:
            features: Combined feature vector for comparison head
            metrics: Dictionary of individual metrics for logging
        """
        # Compute individual similarities
        sims = self.compute_individual_similarities(emb1, emb2)
        
        # 1. Concatenation: raw embedding information
        concat = torch.cat([emb1, emb2], dim=1)  # (batch, 2*emb_dim)
        
        # 5. Attention-weighted fusion of scalar similarities
        # Stack scalar similarities: L1, L2 (inverted), cosine, product
        # Normalize to similar scales
        scalar_sims = torch.cat([
            1.0 / (1.0 + sims['l1_scalar']),  # Convert distance to similarity
            1.0 / (1.0 + sims['l2_scalar']),  # Convert distance to similarity
            (sims['cosine'] + 1) / 2,          # Scale to [0, 1]
            torch.sigmoid(sims['product_scalar'])  # Scale to [0, 1]
        ], dim=1)  # (batch, 4)
        
        attention_fused, attention_weights = self.attention_fusion(
            emb1, emb2, scalar_sims
        )
        
        # Combine all features
        combined = torch.cat([
            concat,                    # 2 * embedding_dim
            sims['l1_vector'],         # embedding_dim
            sims['l2_vector'],         # embedding_dim
            sims['product_vector'],    # embedding_dim
            sims['cosine'],            # 1
            attention_fused            # 1
        ], dim=1)
        
        # Return features and metrics for analysis
        metrics = {
            'l1_mean': sims['l1_scalar'].mean().item(),
            'l2_mean': sims['l2_scalar'].mean().item(),
            'cosine_mean': sims['cosine'].mean().item(),
            'product_mean': sims['product_scalar'].mean().item(),
            'attention_weights': attention_weights.mean(dim=0).cpu().detach().numpy()
        }
        
        return combined, metrics
    
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass for a pair of inputs.
        
        Args:
            x1: First input (batch, features)
            x2: Second input (batch, features)
            
        Returns:
            similarity: Predicted similarity score (batch, 1)
            emb1: Embedding of first input
            emb2: Embedding of second input
            metrics: Dictionary of similarity metrics
        """
        # Get embeddings using shared network
        emb1 = self.embedding_net(x1)
        emb2 = self.embedding_net(x2)
        
        # Compute multi-metric similarity features
        similarity_features, metrics = self.compute_similarity_features(emb1, emb2)
        
        # Final similarity prediction
        similarity = self.comparison_head(similarity_features)
        
        return similarity, emb1, emb2, metrics


class MLPEmbeddingNetwork(nn.Module):
    """
    Fully-connected (MLP) embedding network as an alternative to CNN.
    
    Use this if Conv1D does not yield good results. MLPs can be effective
    for 1D frequency spectra where spatial locality is less important.
    
    Args:
        input_dim: Input feature dimension
        embedding_dim: Output embedding dimension
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        input_dim: int = 2048,
        embedding_dim: int = 128,
        hidden_dims: Tuple[int, ...] = (1024, 512, 256),
        dropout: float = 0.3
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, embedding_dim))
        
        self.network = nn.Sequential(*layers)
        self.normalize = True
        
        logger.info(f"MLPEmbeddingNetwork: {input_dim} -> {hidden_dims} -> {embedding_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute embedding from input features."""
        if x.dim() == 3:
            x = x.squeeze(1)  # Remove channel dimension if present
        
        embeddings = self.network(x)
        
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings


def create_siamese_network(
    input_dim: int = 2048,
    embedding_dim: int = 128,
    dropout: float = 0.3,
    use_mlp: bool = False,
    mlp_hidden_dims: Tuple[int, ...] = (1024, 512, 256)
) -> MultiSimilaritySiameseNetwork:
    """
    Factory function to create the Siamese network.
    
    Args:
        input_dim: Input feature dimension
        embedding_dim: Embedding dimension
        dropout: Dropout rate
        use_mlp: If True, use MLP instead of CNN for embedding
        mlp_hidden_dims: Hidden dimensions for MLP (if use_mlp=True)
        
    Returns:
        Configured MultiSimilaritySiameseNetwork
    """
    network = MultiSimilaritySiameseNetwork(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        dropout=dropout
    )
    
    if use_mlp:
        logger.info("Replacing CNN embedding network with MLP")
        network.embedding_net = MLPEmbeddingNetwork(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            hidden_dims=mlp_hidden_dims,
            dropout=dropout
        )
    
    return network



