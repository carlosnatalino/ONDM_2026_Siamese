"""
Multi-Similarity Siamese Network for DAS Event Classification

This package implements a multi-similarity Siamese neural network designed for:
- Novelty detection in DAS (Distributed Acoustic Sensing) systems
- N-way K-shot few-shot classification
- Real-world incremental deployment scenarios

The network uses 5 different similarity metrics:
1. L1 Distance (Manhattan)
2. L2 Distance (Euclidean)
3. Cosine Similarity
4. Element-wise Product
5. Learned Attention-weighted Fusion

Modules:
- models: Network architectures (EmbeddingNetwork, MultiSimilaritySiameseNetwork)
- training: Training components (Samplers, Datasets, Loss functions, Trainer)
- evaluation: Evaluation utilities (N-way K-shot, Novelty detection, Simulation)
- visualization: Plotting functions

References:
- Bromley et al., "Signature Verification using a Siamese Time Delay NN" (1993)
- Koch et al., "Siamese Neural Networks for One-shot Image Recognition" (2015)
- Snell et al., "Prototypical Networks for Few-shot Learning" (2017)
- Wang et al., "Multi-Similarity Loss with General Pair Weighting" (2019)
- Tomasov et al., "Enhancing Perimeter Protection using Φ-OTDR and CNN" (2023)

Author: Andrei Campeanu, Carlos Natalino
Date: January 2026
"""

from .models import (
    EmbeddingNetwork,
    MLPEmbeddingNetwork,
    MultiSimilaritySiameseNetwork,
    AttentionSimilarityFusion,
    create_siamese_network
)

from .training import (
    SignalAugmentation,
    EpisodicPairSampler,
    SiamesePairDataset,
    MultiSimilarityBCELoss,
    EpisodicTrainer
)

from .evaluation import (
    evaluate_nway_kshot,
    run_comprehensive_nway_kshot_eval,
    NoveltyDetector,
    RealWorldSimulator,
    run_full_evaluation
)

from .visualization import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_tsne_embeddings,
    plot_nway_kshot_results,
    plot_simulation_results,
    generate_all_plots
)

__version__ = '1.0.0'
__all__ = [
    # Models
    'EmbeddingNetwork',
    'MLPEmbeddingNetwork',
    'MultiSimilaritySiameseNetwork',
    'AttentionSimilarityFusion',
    'create_siamese_network',
    # Training
    'SignalAugmentation',
    'EpisodicPairSampler',
    'SiamesePairDataset',
    'MultiSimilarityBCELoss',
    'EpisodicTrainer',
    # Evaluation
    'evaluate_nway_kshot',
    'run_comprehensive_nway_kshot_eval',
    'NoveltyDetector',
    'RealWorldSimulator',
    'run_full_evaluation',
    # Visualization
    'plot_training_curves',
    'plot_confusion_matrix',
    'plot_tsne_embeddings',
    'plot_nway_kshot_results',
    'plot_simulation_results',
    'generate_all_plots',
]



