"""
DAS Foundation Model for Event Classification
==============================================

This module implements a foundation model approach for Distributed Acoustic Sensing (DAS)
event classification, specifically designed for the Tomasov dataset. It incorporates
state-of-the-art techniques from recent literature (2023-2025):

1. Vision Transformer (ViT) backbone - Following MAEPD architecture principles
2. Masked Autoencoder (MAE) for self-supervised pre-training
3. Visual Prompt Tuning (VPT) for parameter-efficient fine-tuning
4. Support for transfer learning and domain adaptation

References:
- MAEPD: A Foundation Model for DAS Signal Recognition (arXiv 2508.04316)
- Cross-modal pre-training framework for DAS (arXiv 2511.09342)
- Tomasov et al., Scientific Data 2025

Author: Generated for research purposes
License: MIT
"""

import os
import logging
import math
from typing import Optional, Callable, List, Tuple, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import h5py
from glob import glob
from collections import Counter
from multiprocessing import Pool

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

# Scikit-learn for metrics and preprocessing
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    f1_score, 
    accuracy_score
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Classes
# =============================================================================

class FineTuningMode(Enum):
    """Fine-tuning strategies for the foundation model."""
    FULL = "full"                    # Fine-tune all parameters
    LINEAR_PROBE = "linear_probe"    # Only train classifier head
    VPT_SHALLOW = "vpt_shallow"      # Visual Prompt Tuning - shallow
    VPT_DEEP = "vpt_deep"            # Visual Prompt Tuning - deep
    LORA = "lora"                    # Low-Rank Adaptation


@dataclass
class DASFoundationConfig:
    """Configuration for the DAS Foundation Model.
    
    This configuration follows the MAEPD architecture principles while being
    optimized for the Tomasov dataset characteristics:
    - 20 kHz pulse rate → 8192-sample windows
    - 2048-element magnitude spectrum after preprocessing
    - 9 event classes for perimeter security
    """
    # Input configuration (matching Tomasov preprocessing)
    input_dim: int = 2048           # Magnitude spectrum length
    window_size: int = 8192         # Raw signal window size
    
    # Patch embedding configuration
    patch_size: int = 64            # Divide spectrum into patches
    num_patches: int = 32           # 2048 / 64 = 32 patches
    
    # Transformer architecture
    embed_dim: int = 384            # Embedding dimension (ViT-Small)
    num_heads: int = 6              # Attention heads
    num_layers: int = 12            # Transformer blocks
    mlp_ratio: float = 4.0          # MLP hidden dim ratio
    dropout: float = 0.1            # Dropout rate
    attention_dropout: float = 0.0  # Attention dropout
    
    # Classification
    num_classes: int = 9            # Tomasov dataset classes
    
    # Masked Autoencoder configuration
    mask_ratio: float = 0.75        # Fraction of patches to mask (MAE default)
    decoder_embed_dim: int = 192    # Decoder embedding dimension
    decoder_num_heads: int = 3      # Decoder attention heads
    decoder_num_layers: int = 4     # Decoder transformer blocks
    
    # Visual Prompt Tuning configuration
    num_prompts: int = 10           # Number of learnable prompt tokens
    prompt_dropout: float = 0.1     # Prompt dropout rate
    
    # Training configuration
    learning_rate: float = 1e-4     # Base learning rate
    weight_decay: float = 0.05      # Weight decay
    warmup_epochs: int = 10         # Warmup epochs
    max_epochs: int = 100           # Maximum training epochs
    batch_size: int = 64            # Batch size
    
    # Class names for Tomasov dataset (must match TomasovDASDataset.EXPECTED_CLASSES)
    class_names: List[str] = field(default_factory=lambda: [
        "car", "construction", "fence", "longboard", 
        "manipulation", "openclose", "regular", "running", "walk"
    ])
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.num_classes == len(self.class_names), \
            f"num_classes ({self.num_classes}) must match len(class_names) ({len(self.class_names)})"
        assert self.input_dim % self.patch_size == 0, \
            f"input_dim ({self.input_dim}) must be divisible by patch_size ({self.patch_size})"
        # Update num_patches based on input_dim and patch_size
        self.num_patches = self.input_dim // self.patch_size


# =============================================================================
# Core Model Components
# =============================================================================

class PatchEmbedding1D(nn.Module):
    """1D Patch Embedding for DAS spectral data.
    
    Converts the magnitude spectrum into a sequence of patch embeddings,
    following the Vision Transformer approach adapted for 1D signals.
    """
    
    def __init__(
        self, 
        input_dim: int = 2048,
        patch_size: int = 64, 
        embed_dim: int = 384
    ):
        super().__init__()
        self.input_dim = input_dim
        self.patch_size = patch_size
        self.num_patches = input_dim // patch_size
        self.embed_dim = embed_dim
        
        # Linear projection of flattened patches
        self.projection = nn.Conv1d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, input_dim) or (batch, 1, input_dim)
        Returns:
            Patch embeddings of shape (batch, num_patches, embed_dim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)
        
        x = self.projection(x)  # (batch, embed_dim, num_patches)
        x = x.transpose(1, 2)   # (batch, num_patches, embed_dim)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention with optional attention dropout."""
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        dropout: float = 0.0,
        attention_dropout: float = 0.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(
            batch_size, seq_len, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        if return_attention:
            return x, attn
        return x, None


class TransformerBlock(nn.Module):
    """Transformer encoder block with pre-normalization."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(
            embed_dim, num_heads, dropout, attention_dropout
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(
        self, 
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Self-attention with residual connection
        attn_out, attn_weights = self.attn(self.norm1(x), return_attention)
        x = x + attn_out
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x, attn_weights


class VisualPromptTokens(nn.Module):
    """Visual Prompt Tuning (VPT) tokens for parameter-efficient fine-tuning.
    
    Implements both shallow and deep VPT following:
    "Visual Prompt Tuning" (Jia et al., ECCV 2022)
    
    In deep VPT, learnable prompts are prepended to each transformer layer,
    allowing adaptation with only ~0.3% of parameters (matching MAEPD results).
    """
    
    def __init__(
        self,
        num_prompts: int,
        embed_dim: int,
        num_layers: int,
        deep: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_prompts = num_prompts
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.deep = deep
        
        if deep:
            # Deep VPT: separate prompts for each layer
            self.prompts = nn.ParameterList([
                nn.Parameter(torch.zeros(1, num_prompts, embed_dim))
                for _ in range(num_layers)
            ])
        else:
            # Shallow VPT: prompts only at input
            self.prompts = nn.ParameterList([
                nn.Parameter(torch.zeros(1, num_prompts, embed_dim))
            ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize prompts
        for prompt in self.prompts:
            nn.init.normal_(prompt, std=0.02)
            
    def get_prompts(self, layer_idx: int, batch_size: int) -> torch.Tensor:
        """Get prompts for a specific layer."""
        if self.deep:
            prompts = self.prompts[layer_idx]
        else:
            prompts = self.prompts[0] if layer_idx == 0 else None
            
        if prompts is None:
            return None
            
        prompts = prompts.expand(batch_size, -1, -1)
        return self.dropout(prompts)


# =============================================================================
# Main Model Architecture
# =============================================================================

class DASViTEncoder(nn.Module):
    """Vision Transformer Encoder for DAS signals.
    
    This encoder forms the backbone of the DAS foundation model,
    processing 1D spectral representations into rich feature embeddings.
    """
    
    def __init__(self, config: DASFoundationConfig):
        super().__init__()
        self.config = config
        
        # Patch embedding
        self.patch_embed = PatchEmbedding1D(
            input_dim=config.input_dim,
            patch_size=config.patch_size,
            embed_dim=config.embed_dim
        )
        
        # CLS token for classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        
        # Positional embedding (learnable)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.num_patches + 1, config.embed_dim)
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout
            )
            for _ in range(config.num_layers)
        ])
        
        self.norm = nn.LayerNorm(config.embed_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        
    def forward(
        self,
        x: torch.Tensor,
        vpt_tokens: Optional[VisualPromptTokens] = None,
        return_all_tokens: bool = False,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the encoder.
        
        Args:
            x: Input tensor of shape (batch, input_dim)
            vpt_tokens: Optional Visual Prompt Tokens for VPT fine-tuning
            return_all_tokens: Return all token embeddings (not just CLS)
            return_attention: Return attention weights from each layer
            
        Returns:
            Dictionary containing:
                - 'cls_embedding': CLS token embedding
                - 'patch_embeddings': All patch embeddings (if return_all_tokens)
                - 'attention_weights': Attention weights (if return_attention)
        """
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch, num_patches, embed_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, num_patches + 1, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Store attention weights if requested
        attention_weights = []
        
        # Process through transformer blocks
        for layer_idx, block in enumerate(self.blocks):
            # Add VPT prompts if provided
            if vpt_tokens is not None:
                prompts = vpt_tokens.get_prompts(layer_idx, batch_size)
                if prompts is not None:
                    x = torch.cat([x[:, :1], prompts, x[:, 1:]], dim=1)
            
            x, attn = block(x, return_attention=return_attention)
            
            if return_attention and attn is not None:
                attention_weights.append(attn)
            
            # Remove prompt tokens before next layer (if using deep VPT)
            if vpt_tokens is not None and vpt_tokens.deep:
                prompts = vpt_tokens.get_prompts(layer_idx, batch_size)
                if prompts is not None:
                    num_prompts = prompts.shape[1]
                    x = torch.cat([x[:, :1], x[:, 1 + num_prompts:]], dim=1)
        
        x = self.norm(x)
        
        result = {
            'cls_embedding': x[:, 0],  # CLS token
        }
        
        if return_all_tokens:
            result['patch_embeddings'] = x[:, 1:]
            
        if return_attention:
            result['attention_weights'] = attention_weights
            
        return result


class MaskedAutoencoder(nn.Module):
    """Masked Autoencoder for self-supervised pre-training.
    
    Implements MAE pre-training strategy following:
    - "Masked Autoencoders Are Scalable Vision Learners" (He et al., CVPR 2022)
    - MAEPD adaptation for DAS signals (arXiv 2508.04316)
    
    Key insight: By masking 75% of patches and reconstructing them,
    the encoder learns rich representations of DAS signal patterns.
    """
    
    def __init__(self, config: DASFoundationConfig, encoder: DASViTEncoder):
        super().__init__()
        self.config = config
        self.encoder = encoder
        
        # Decoder embedding projection
        self.decoder_embed = nn.Linear(
            config.embed_dim, 
            config.decoder_embed_dim
        )
        
        # Mask token for reconstruction
        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, config.decoder_embed_dim)
        )
        
        # Decoder positional embedding
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, config.num_patches + 1, config.decoder_embed_dim)
        )
        
        # Decoder transformer blocks
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=config.decoder_embed_dim,
                num_heads=config.decoder_num_heads,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout
            )
            for _ in range(config.decoder_num_layers)
        ])
        
        self.decoder_norm = nn.LayerNorm(config.decoder_embed_dim)
        
        # Prediction head: reconstruct patch values
        self.decoder_pred = nn.Linear(
            config.decoder_embed_dim, 
            config.patch_size
        )
        
        # Initialize
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.decoder_pos_embed, std=0.02)
        
    def random_masking(
        self, 
        x: torch.Tensor, 
        mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform random masking on patch embeddings.
        
        Args:
            x: Patch embeddings (batch, num_patches, embed_dim)
            mask_ratio: Fraction of patches to mask
            
        Returns:
            x_masked: Unmasked patches
            mask: Binary mask (1 = masked)
            ids_restore: Indices to restore original order
        """
        batch_size, num_patches, embed_dim = x.shape
        num_keep = int(num_patches * (1 - mask_ratio))
        
        # Random permutation for each sample
        noise = torch.rand(batch_size, num_patches, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep first num_keep patches
        ids_keep = ids_shuffle[:, :num_keep]
        x_masked = torch.gather(
            x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, embed_dim)
        )
        
        # Generate binary mask: 1 = masked, 0 = kept
        mask = torch.ones(batch_size, num_patches, device=x.device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(
        self, 
        x: torch.Tensor, 
        mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode with random masking."""
        # Get patch embeddings (without CLS token initially)
        x = self.encoder.patch_embed(x)
        
        # Add positional embedding (without CLS position)
        x = x + self.encoder.pos_embed[:, 1:]
        
        # Random masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Add CLS token
        cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Apply transformer blocks
        for block in self.encoder.blocks:
            x, _ = block(x)
        x = self.encoder.norm(x)
        
        return x, mask, ids_restore
    
    def forward_decoder(
        self, 
        x: torch.Tensor, 
        ids_restore: torch.Tensor
    ) -> torch.Tensor:
        """Decode to reconstruct masked patches."""
        # Embed tokens
        x = self.decoder_embed(x)
        
        # Append mask tokens
        mask_tokens = self.mask_token.expand(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], -1
        )
        x_ = torch.cat([x[:, 1:], mask_tokens], dim=1)  # Skip CLS
        
        # Unshuffle to original order
        x_ = torch.gather(
            x_, 1, 
            ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2])
        )
        x = torch.cat([x[:, :1], x_], dim=1)  # Add CLS back
        
        # Add positional embedding
        x = x + self.decoder_pos_embed
        
        # Apply decoder blocks
        for block in self.decoder_blocks:
            x, _ = block(x)
        x = self.decoder_norm(x)
        
        # Predict patch values
        x = self.decoder_pred(x)
        
        # Remove CLS token
        x = x[:, 1:]
        
        return x
    
    def forward(
        self, 
        x: torch.Tensor,
        mask_ratio: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for MAE pre-training.
        
        Args:
            x: Input tensor (batch, input_dim)
            mask_ratio: Override config mask ratio
            
        Returns:
            Dictionary with 'loss', 'pred', 'mask'
        """
        if mask_ratio is None:
            mask_ratio = self.config.mask_ratio
            
        # Encode with masking
        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio)
        
        # Decode to reconstruct
        pred = self.forward_decoder(latent, ids_restore)
        
        # Compute reconstruction loss (MSE on masked patches)
        target = self.patchify(x)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # Mean per patch
        
        # Only compute loss on masked patches
        loss = (loss * mask).sum() / mask.sum()
        
        return {
            'loss': loss,
            'pred': pred,
            'mask': mask,
            'target': target
        }
    
    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert input to patches for reconstruction target."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Reshape to patches
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.config.num_patches, self.config.patch_size)
        return x


class DASFoundationModel(nn.Module):
    """Complete DAS Foundation Model for event classification.
    
    This model combines:
    1. ViT encoder backbone (can be pre-trained with MAE)
    2. Visual Prompt Tuning for parameter-efficient fine-tuning
    3. Classification head for 9-class Tomasov dataset
    
    Supports multiple fine-tuning modes:
    - FULL: Fine-tune all parameters
    - LINEAR_PROBE: Only train classifier head
    - VPT_SHALLOW/VPT_DEEP: Visual Prompt Tuning
    - LORA: Low-Rank Adaptation (future work)
    """
    
    def __init__(
        self, 
        config: DASFoundationConfig,
        fine_tuning_mode: FineTuningMode = FineTuningMode.FULL
    ):
        super().__init__()
        self.config = config
        self.fine_tuning_mode = fine_tuning_mode
        
        # Encoder backbone
        self.encoder = DASViTEncoder(config)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim, config.num_classes)
        )
        
        # Visual Prompt Tokens (initialized if using VPT)
        self.vpt_tokens = None
        if fine_tuning_mode in [FineTuningMode.VPT_SHALLOW, FineTuningMode.VPT_DEEP]:
            self.vpt_tokens = VisualPromptTokens(
                num_prompts=config.num_prompts,
                embed_dim=config.embed_dim,
                num_layers=config.num_layers,
                deep=(fine_tuning_mode == FineTuningMode.VPT_DEEP),
                dropout=config.prompt_dropout
            )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Apply fine-tuning mode
        self._apply_fine_tuning_mode()
        
    def _apply_fine_tuning_mode(self):
        """Configure which parameters are trainable based on fine-tuning mode."""
        if self.fine_tuning_mode == FineTuningMode.LINEAR_PROBE:
            # Freeze encoder, only train classifier
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.classifier.parameters():
                param.requires_grad = True
                
        elif self.fine_tuning_mode in [FineTuningMode.VPT_SHALLOW, FineTuningMode.VPT_DEEP]:
            # Freeze encoder, train prompts and classifier
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.classifier.parameters():
                param.requires_grad = True
            if self.vpt_tokens is not None:
                for param in self.vpt_tokens.parameters():
                    param.requires_grad = True
                    
        elif self.fine_tuning_mode == FineTuningMode.FULL:
            # Train everything
            for param in self.parameters():
                param.requires_grad = True
    
    def get_trainable_params(self) -> int:
        """Count number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        """Count total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def forward(
        self, 
        x: torch.Tensor,
        return_features: bool = False,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for classification.
        
        Args:
            x: Input tensor (batch, input_dim)
            return_features: Return intermediate features
            return_attention: Return attention weights
            
        Returns:
            Dictionary with 'logits' and optionally 'features', 'attention'
        """
        # Encode
        encoder_output = self.encoder(
            x, 
            vpt_tokens=self.vpt_tokens,
            return_all_tokens=return_features,
            return_attention=return_attention
        )
        
        # Classify using CLS token
        cls_embedding = encoder_output['cls_embedding']
        logits = self.classifier(cls_embedding)
        
        result = {'logits': logits}
        
        if return_features:
            result['features'] = cls_embedding
            result['patch_embeddings'] = encoder_output.get('patch_embeddings')
            
        if return_attention:
            result['attention_weights'] = encoder_output.get('attention_weights')
            
        return result
    
    def load_pretrained_encoder(self, checkpoint_path: str, strict: bool = False):
        """Load pre-trained encoder weights from MAE checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        if 'encoder' in checkpoint:
            encoder_state = checkpoint['encoder']
        elif 'model' in checkpoint:
            # Extract encoder weights from full model
            encoder_state = {
                k.replace('encoder.', ''): v 
                for k, v in checkpoint['model'].items() 
                if k.startswith('encoder.')
            }
        else:
            encoder_state = checkpoint
            
        missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=strict)
        
        logger.info(f"Loaded pre-trained encoder from {checkpoint_path}")
        if missing:
            logger.warning(f"Missing keys: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected}")
            
        # Reapply fine-tuning mode
        self._apply_fine_tuning_mode()


# =============================================================================
# Dataset and Data Loading
# =============================================================================

class TomasovDASDataset(Dataset):
    """PyTorch Dataset for the Tomasov DAS dataset.
    
    This dataset class extends the original data_loader.py functionality
    with PyTorch Dataset interface for seamless integration with DataLoader.
    
    Uses stratified splitting to ensure all 9 classes appear in train/val/test splits,
    which is critical since some classes (e.g., construction) have only 1 HDF5 file.
    """
    
    # Expected classes in Tomasov dataset (for consistent ordering)
    EXPECTED_CLASSES = [
        "car", "construction", "fence", "longboard", 
        "manipulation", "openclose", "regular", "running", "walk"
    ]
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        window_size: int = 8192,
        shift: int = 2048,
        sample_len: int = 2048,
        drop_noise: bool = True,
        decimate: Optional[Dict[str, int]] = None,
        split: str = 'train',
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        random_seed: int = 42
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Path to dataset directory with label subdirectories
            transform: Preprocessing transform (default: FFT)
            window_size: Signal window size
            shift: Window shift for overlap
            sample_len: Final sample length after transform
            drop_noise: Whether to filter noisy samples
            decimate: Decimation factors per label
            split: 'train', 'val', or 'test'
            split_ratio: (train, val, test) ratios
            random_seed: Random seed for reproducible splits
        """
        self.data_dir = data_dir
        self.transform = transform or self._default_fft_transform
        self.window_size = window_size
        self.shift = shift
        self.sample_len = sample_len
        self.drop_noise = drop_noise
        self.decimate = decimate or {}
        self.split = split
        self.split_ratio = split_ratio
        self.random_seed = random_seed
        
        # Load and process data
        self.samples, self.labels, self.label_names = self._load_data()
        
        # Compute class weights for imbalanced learning (for all expected classes)
        self._compute_class_weights()
        
    def _default_fft_transform(self, x: np.ndarray) -> np.ndarray:
        """Default FFT preprocessing matching Tomasov paper."""
        x = np.fft.rfft(x)[:, 1:]  # Remove DC
        x = np.abs(x) + 1
        x = np.log10(x)
        return x
    
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load and preprocess the dataset."""
        all_samples = []
        all_labels = []
        file_indices = []
        
        label_dirs = sorted(os.listdir(self.data_dir))
        label_dirs = [d for d in label_dirs if os.path.isdir(
            os.path.join(self.data_dir, d)
        )]
        
        # Validate that found directories match expected classes
        found_labels = set(label_dirs)
        expected_labels = set(self.EXPECTED_CLASSES)
        
        missing = expected_labels - found_labels
        extra = found_labels - expected_labels
        
        if missing:
            logger.warning(f"Expected label directories not found: {missing}")
        if extra:
            logger.warning(f"Unexpected label directories found (will be ignored): {extra}")
            # Only process expected classes
            label_dirs = [d for d in label_dirs if d in expected_labels]
        
        logger.info(f"Processing {len(label_dirs)} label directories: {label_dirs}")
        
        for label in label_dirs:
            label_path = os.path.join(self.data_dir, label)
            h5_files = sorted(glob(os.path.join(label_path, "*.h5")))
            
            for file_idx, h5_file in enumerate(h5_files):
                logger.info(f"Loading {h5_file}")
                
                with h5py.File(h5_file, 'r') as f:
                    data = f["Acquisition"]["Raw[0]"]["RawData"][:]
                    
                # Load bitmap labels
                npy_file = h5_file[:-2] + "npy"
                if os.path.exists(npy_file):
                    bmp = np.load(npy_file)
                else:
                    logger.warning(f"No bitmap file found for {h5_file}")
                    continue
                
                # Extract windows
                decimate_factor = self.decimate.get(label, 1)
                windows = self._extract_windows(data, bmp, decimate_factor)
                
                all_samples.extend(windows)
                all_labels.extend([label] * len(windows))
                file_indices.extend([f"{label}_{file_idx}"] * len(windows))
        
        # Convert to numpy arrays
        samples = np.array(all_samples, dtype=np.float32)
        labels = np.array(all_labels)
        file_indices = np.array(file_indices)
        
        # Apply transform
        logger.info("Applying preprocessing transform...")
        samples = self.transform(samples)
        
        # Trim to sample length
        if self.sample_len and samples.shape[1] > self.sample_len:
            samples = samples[:, :self.sample_len]
        
        # Drop noisy samples
        if self.drop_noise:
            logger.info("Filtering noisy samples...")
            samples, labels, file_indices = self._filter_noise(
                samples, all_samples, labels, file_indices
            )
        
        # Encode labels using all expected classes for consistency
        self.label_encoder = LabelEncoder()
        # Fit on all expected classes to ensure consistent encoding across splits
        self.label_encoder.fit(self.EXPECTED_CLASSES)
        encoded_labels = self.label_encoder.transform(labels)
        label_names = list(self.label_encoder.classes_)
        
        logger.info(f"Label encoding: {dict(zip(label_names, range(len(label_names))))}")
        
        # Split data
        samples, labels = self._apply_split(
            samples, encoded_labels, file_indices
        )
        
        logger.info(f"Dataset split '{self.split}': {len(samples)} samples")
        logger.info(f"Label distribution: {Counter(labels)}")
        
        return samples, labels, label_names
    
    def _extract_windows(
        self, 
        data: np.ndarray, 
        bmp: np.ndarray, 
        decimate: int
    ) -> List[np.ndarray]:
        """Extract signal windows based on bitmap labels."""
        windows = []
        positions = np.transpose(np.where(bmp))
        
        for pos, channel in positions[::decimate]:
            start = pos * self.shift
            end = start + self.window_size
            
            if end <= data.shape[0]:
                window = data[start:end, channel].astype(np.float32)
                windows.append(window)
                
        return windows
    
    def _filter_noise(
        self,
        processed: np.ndarray,
        raw: List[np.ndarray],
        labels: np.ndarray,
        file_indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filter noisy samples using spectral analysis."""
        def spec_condition(x: np.ndarray) -> bool:
            split = int(len(x) * 0.1)
            return np.mean(x[:split]) - np.mean(x[split:]) > 0.05
        
        # Apply FFT to raw for noise detection
        raw = np.array(raw, dtype=np.float32)
        raw_fft = self._default_fft_transform(raw)
        
        # Keep samples that pass spectral condition or are "regular"
        keep_mask = np.apply_along_axis(spec_condition, 1, raw_fft)
        keep_mask |= (labels == "regular")
        
        return processed[keep_mask], labels[keep_mask], file_indices[keep_mask]
    
    def _apply_split(
        self,
        samples: np.ndarray,
        labels: np.ndarray,
        file_indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply stratified train/val/test split to ensure all classes appear in all splits.
        
        This is critical for the Tomasov dataset where some classes (e.g., construction)
        have only 1 HDF5 file - pure file-based splitting would leave some classes
        entirely absent from certain splits.
        """
        np.random.seed(self.random_seed)
        
        # Create indices
        indices = np.arange(len(samples))
        
        # First split: separate test set (stratified by class)
        train_val_ratio = self.split_ratio[0] + self.split_ratio[1]
        test_ratio = self.split_ratio[2]
        
        # Handle edge case where a class might have very few samples
        # Use stratified split with fallback to random if stratification fails
        try:
            train_val_idx, test_idx = train_test_split(
                indices,
                test_size=test_ratio,
                stratify=labels,
                random_state=self.random_seed
            )
        except ValueError as e:
            logger.warning(f"Stratified split failed, using random split: {e}")
            train_val_idx, test_idx = train_test_split(
                indices,
                test_size=test_ratio,
                random_state=self.random_seed
            )
        
        # Second split: separate train and val from train_val
        val_ratio_adjusted = self.split_ratio[1] / train_val_ratio
        
        try:
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=val_ratio_adjusted,
                stratify=labels[train_val_idx],
                random_state=self.random_seed
            )
        except ValueError as e:
            logger.warning(f"Stratified val split failed, using random split: {e}")
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=val_ratio_adjusted,
                random_state=self.random_seed
            )
        
        # Select appropriate split
        if self.split == 'train':
            selected_idx = train_idx
        elif self.split == 'val':
            selected_idx = val_idx
        else:  # test
            selected_idx = test_idx
        
        # Log class distribution for this split
        split_labels = labels[selected_idx]
        unique, counts = np.unique(split_labels, return_counts=True)
        logger.info(f"Split '{self.split}' class distribution: {dict(zip(unique, counts))}")
        
        return samples[selected_idx], labels[selected_idx]
    
    def _compute_class_weights(self):
        """Compute class weights for handling imbalanced data.
        
        Always produces weights for all 9 expected classes to ensure
        consistency with the loss function, even if some classes have
        fewer samples in a particular split.
        """
        num_classes = len(self.EXPECTED_CLASSES)
        
        # Count samples per class in current split
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        label_counts = dict(zip(unique_labels, counts))
        
        # Compute weights for all classes (use median count for missing classes)
        total_samples = len(self.labels)
        median_count = np.median(counts) if len(counts) > 0 else 1
        
        weights = []
        for class_idx in range(num_classes):
            if class_idx in label_counts:
                # Standard balanced weight: total / (num_classes * count)
                weight = total_samples / (num_classes * label_counts[class_idx])
            else:
                # For missing classes, use a reasonable default weight
                weight = total_samples / (num_classes * median_count)
                logger.warning(f"Class {class_idx} ({self.EXPECTED_CLASSES[class_idx]}) "
                             f"not in {self.split} split, using default weight")
            weights.append(weight)
        
        self.class_weights = torch.FloatTensor(weights)
        logger.info(f"Class weights for {self.split}: {self.class_weights.tolist()}")
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = torch.FloatTensor(self.samples[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        return sample, label


# =============================================================================
# Training Utilities
# =============================================================================

class DASTrainer:
    """Trainer class for DAS Foundation Model.
    
    Supports:
    - MAE pre-training
    - Supervised fine-tuning with various strategies
    - Early stopping and checkpointing
    - Comprehensive metrics logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: DASFoundationConfig,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        logger.info(f"Using device: {device}")
        logger.info(f"Total parameters: {model.get_total_params():,}")
        logger.info(f"Trainable parameters: {model.get_trainable_params():,}")
        logger.info(f"Trainable ratio: {100 * model.get_trainable_params() / model.get_total_params():.2f}%")
        
    def pretrain_mae(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        save_path: str = 'mae_pretrained.pt'
    ) -> Dict[str, List[float]]:
        """Pre-train encoder using Masked Autoencoder."""
        mae = MaskedAutoencoder(self.config, self.model.encoder).to(self.device)
        
        optimizer = AdamW(
            mae.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            mae.train()
            train_loss = 0.0
            
            for batch_idx, (x, _) in enumerate(train_loader):
                x = x.to(self.device)
                
                optimizer.zero_grad()
                output = mae(x)
                loss = output['loss']
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validation
            if val_loader is not None:
                mae.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for x, _ in val_loader:
                        x = x.to(self.device)
                        output = mae(x)
                        val_loss += output['loss'].item()
                        
                val_loss /= len(val_loader)
                history['val_loss'].append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'epoch': epoch,
                        'encoder': mae.encoder.state_dict(),
                        'mae': mae.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'val_loss': val_loss
                    }, save_path)
                    
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f} - "
                    f"Val Loss: {val_loss:.4f}"
                )
            else:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")
                
            scheduler.step()
            
        return history
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: Optional[int] = None,
        class_weights: Optional[torch.Tensor] = None,
        save_path: str = 'best_model.pt',
        patience: int = 15
    ) -> Dict[str, List[float]]:
        """Train model for classification."""
        epochs = epochs or self.config.max_epochs
        
        # Setup optimizer (only for trainable params)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate,
            epochs=epochs,
            steps_per_epoch=len(train_loader)
        )
        
        # Loss function with optional class weights
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'val_f1': []
        }
        
        best_val_f1 = 0.0
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(x)
                loss = criterion(output['logits'], y)
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                _, predicted = output['logits'].max(1)
                train_total += y.size(0)
                train_correct += predicted.eq(y).sum().item()
                
            train_loss /= len(train_loader)
            train_acc = 100. * train_correct / train_total
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Validation
            val_metrics = self.evaluate(val_loader)
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['val_f1'].append(val_metrics['f1_macro'])
            
            # Save best model
            if val_metrics['f1_macro'] > best_val_f1:
                best_val_f1 = val_metrics['f1_macro']
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'val_f1': best_val_f1,
                    'config': self.config
                }, save_path)
            else:
                patience_counter += 1
                
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% - "
                f"Val Loss: {val_metrics['loss']:.4f} - Val Acc: {val_metrics['accuracy']:.2f}% - "
                f"Val F1: {val_metrics['f1_macro']:.4f}"
            )
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
                
        return history
    
    def evaluate(
        self, 
        loader: DataLoader,
        return_predictions: bool = False
    ) -> Dict[str, Any]:
        """Evaluate model on a dataset."""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        total_loss = 0.0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                
                output = self.model(x)
                loss = criterion(output['logits'], y)
                total_loss += loss.item()
                
                _, predicted = output['logits'].max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        metrics = {
            'loss': total_loss / len(loader),
            'accuracy': 100. * accuracy_score(all_labels, all_preds),
            'f1_macro': f1_score(all_labels, all_preds, average='macro'),
            'f1_weighted': f1_score(all_labels, all_preds, average='weighted'),
            'confusion_matrix': confusion_matrix(all_labels, all_preds)
        }
        
        if return_predictions:
            metrics['predictions'] = all_preds
            metrics['labels'] = all_labels
            
        return metrics
    
    def get_classification_report(
        self, 
        loader: DataLoader, 
        label_names: List[str]
    ) -> str:
        """Generate detailed classification report."""
        metrics = self.evaluate(loader, return_predictions=True)
        return classification_report(
            metrics['labels'],
            metrics['predictions'],
            target_names=label_names
        )


# =============================================================================
# Main Execution Example
# =============================================================================

def main():
    """
    Example usage demonstrating the complete pipeline:
    1. Load Tomasov dataset
    2. Initialize DAS Foundation Model
    3. (Optional) Pre-train with MAE
    4. Fine-tune with Visual Prompt Tuning
    5. Evaluate on test set
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='DAS Foundation Model Training')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to Tomasov dataset directory')
    parser.add_argument('--mode', type=str, default='vpt_deep',
                        choices=['full', 'linear_probe', 'vpt_shallow', 'vpt_deep'],
                        help='Fine-tuning mode')
    parser.add_argument('--pretrain', action='store_true',
                        help='Run MAE pre-training before fine-tuning')
    parser.add_argument('--pretrain_epochs', type=int, default=100,
                        help='Number of pre-training epochs')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of fine-tuning epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Output directory for checkpoints')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configuration
    config = DASFoundationConfig(
        batch_size=args.batch_size,
        max_epochs=args.epochs
    )
    
    # Determine fine-tuning mode
    mode_map = {
        'full': FineTuningMode.FULL,
        'linear_probe': FineTuningMode.LINEAR_PROBE,
        'vpt_shallow': FineTuningMode.VPT_SHALLOW,
        'vpt_deep': FineTuningMode.VPT_DEEP
    }
    fine_tuning_mode = mode_map[args.mode]
    
    logger.info(f"Loading dataset from {args.data_dir}")
    
    # Load datasets
    train_dataset = TomasovDASDataset(
        data_dir=args.data_dir,
        split='train',
        sample_len=config.input_dim
    )
    
    val_dataset = TomasovDASDataset(
        data_dir=args.data_dir,
        split='val',
        sample_len=config.input_dim
    )
    
    test_dataset = TomasovDASDataset(
        data_dir=args.data_dir,
        split='test',
        sample_len=config.input_dim
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Initialize model
    model = DASFoundationModel(config, fine_tuning_mode=fine_tuning_mode)
    trainer = DASTrainer(model, config)
    
    # Optional: MAE Pre-training
    if args.pretrain:
        logger.info("Starting MAE pre-training...")
        pretrain_history = trainer.pretrain_mae(
            train_loader,
            val_loader,
            epochs=args.pretrain_epochs,
            save_path=os.path.join(args.output_dir, 'mae_pretrained.pt')
        )
        
        # Load pre-trained weights
        model.load_pretrained_encoder(
            os.path.join(args.output_dir, 'mae_pretrained.pt')
        )
    
    # Fine-tuning
    logger.info(f"Starting fine-tuning with mode: {args.mode}")
    history = trainer.train(
        train_loader,
        val_loader,
        class_weights=train_dataset.class_weights,
        save_path=os.path.join(args.output_dir, 'best_model.pt'),
        patience=15
    )
    
    # Load best model and evaluate
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model'])
    
    # Final evaluation
    logger.info("\n" + "="*60)
    logger.info("FINAL EVALUATION ON TEST SET")
    logger.info("="*60)
    
    test_metrics = trainer.evaluate(test_loader)
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
    logger.info(f"Test F1 (Macro): {test_metrics['f1_macro']:.4f}")
    logger.info(f"Test F1 (Weighted): {test_metrics['f1_weighted']:.4f}")
    
    # Classification report - use the consistent label names
    label_names = train_dataset.EXPECTED_CLASSES
    report = trainer.get_classification_report(test_loader, label_names)
    logger.info("\nClassification Report:\n" + report)
    
    # Save confusion matrix
    np.save(
        os.path.join(args.output_dir, 'confusion_matrix.npy'),
        test_metrics['confusion_matrix']
    )
    
    logger.info(f"\nResults saved to {args.output_dir}")
    

if __name__ == '__main__':
    main()