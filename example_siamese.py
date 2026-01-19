#!/usr/bin/env python3
"""
Example Usage of Siamese Network for DAS Event Classification

This script demonstrates how to:
1. Load and preprocess DAS data
2. Train a Siamese network
3. Evaluate on N-way K-shot tasks
4. Perform novelty detection
5. Visualize results
"""

import torch
import numpy as np
from pathlib import Path

# Import Siamese network components
from siamese_multisim import (
    create_siamese_network,
    SignalAugmentation,
    EpisodicPairSampler,
    SiamesePairDataset,
    MultiSimilarityBCELoss,
    EpisodicTrainer,
    evaluate_nway_kshot,
    NoveltyDetector,
    plot_training_curves
)

# Data loader
from data_loader import DASDataLoader, fft


def main():
    # ========================================================================
    # Configuration
    # ========================================================================
    
    DATA_DIR = "/path/to/DAS-dataset/data"  # Change this!
    OUTPUT_DIR = "siamese_example_output"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Training classes (5 classes for few-shot learning scenario)
    TRAIN_CLASSES = ['regular', 'walk', 'car', 'manipulation', 'openclose']
    
    # Hyperparameters
    EMBEDDING_DIM = 128
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    EPOCHS = 10  # Use 100 for full training
    
    # ========================================================================
    # Step 1: Load Data
    # ========================================================================
    
    print("=" * 70)
    print("STEP 1: Loading DAS Dataset")
    print("=" * 70)
    
    # Create data loader
    loader = DASDataLoader(
        data_dir=DATA_DIR,
        sample_len=2048,
        transform=fft,
        fsize=8192,
        shift=2048,
        decimate={cls: 50 for cls in TRAIN_CLASSES},  # High decimation for demo
        drop_noise=True
    )
    
    # Load dataset
    x, y_onehot = loader.parse_dataset()
    y = y_onehot.argmax(axis=1)
    class_names = list(loader.encoder.classes_)
    
    print(f"Loaded {len(x)} samples from {len(class_names)} classes")
    print(f"Classes: {class_names}")
    
    # Normalize
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)
    
    # Train/test split (simple 80/20)
    split_idx = int(0.8 * len(x))
    x_train, x_test = x[:split_idx], x[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Train: {len(x_train)}, Test: {len(x_test)}")
    
    # ========================================================================
    # Step 2: Create Model
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("STEP 2: Creating Siamese Network")
    print("=" * 70)
    
    model = create_siamese_network(
        input_dim=2048,
        embedding_dim=EMBEDDING_DIM,
        dropout=0.3,
        use_mlp=False  # Use CNN (set True for MLP)
    )
    
    model = model.to(DEVICE)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # ========================================================================
    # Step 3: Setup Training
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("STEP 3: Setting up Training Components")
    print("=" * 70)
    
    # Create datasets
    augmentation = SignalAugmentation(
        noise_std=0.05,
        scale_range=(0.9, 1.1),
        freq_mask_param=50
    )
    
    train_dataset = SiamesePairDataset(
        x=x_train,
        y=y_train,
        augment=True,
        augmentation=augmentation
    )
    
    # Create sampler
    sampler = EpisodicPairSampler(
        labels=y_train,
        batch_size=BATCH_SIZE,
        positive_ratio=0.5,
        n_batches=50  # Reduced for demo
    )
    
    # Create trainer
    trainer = EpisodicTrainer(
        model=model,
        device=DEVICE,
        output_dir=OUTPUT_DIR,
        regular_class_idx=0
    )
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )
    
    criterion = MultiSimilarityBCELoss(
        margin=0.5,
        alpha=2.0,
        beta=50.0,
        lambda_ms=0.1
    )
    
    print("Training components ready!")
    
    # ========================================================================
    # Step 4: Train Model
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("STEP 4: Training Siamese Network")
    print("=" * 70)
    
    for epoch in range(1, EPOCHS + 1):
        # Train one epoch
        train_metrics = trainer.train_epoch(
            dataset=train_dataset,
            sampler=sampler,
            optimizer=optimizer,
            criterion=criterion,
            grad_clip=1.0
        )
        
        print(f"Epoch {epoch}/{EPOCHS} | "
              f"Loss: {train_metrics['loss']:.4f} | "
              f"Acc: {train_metrics['pair_acc']:.4f} | "
              f"F1: {train_metrics['f1']:.4f}")
    
    print("\nTraining complete!")
    
    # ========================================================================
    # Step 5: Evaluate - N-way K-shot
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("STEP 5: Evaluating N-way K-shot Performance")
    print("=" * 70)
    
    # 5-way 5-shot evaluation
    nway_results = evaluate_nway_kshot(
        model=model,
        x=x_test,
        y=y_test,
        n_way=5,
        k_shot=5,
        n_query=10,
        n_episodes=20,  # Use 100 for full evaluation
        device=DEVICE
    )
    
    print(f"5-way 5-shot Accuracy: {nway_results['accuracy']:.4f} ± {nway_results['accuracy_std']:.4f}")
    print(f"F1 Score (Macro): {nway_results['f1_macro']:.4f}")
    
    # ========================================================================
    # Step 6: Novelty Detection
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("STEP 6: Testing Novelty Detection")
    print("=" * 70)
    
    # Create novelty detector
    detector = NoveltyDetector(model=model, device=DEVICE)
    
    # Fit on known classes (training data)
    detector.fit(x_train, y_train, statistical_k=2.0)
    
    # Detect on test data (includes known classes)
    novel_labels = detector.predict(x_test, method='statistical')
    
    # Compute metrics (assuming test has only known classes)
    n_flagged = (novel_labels == -1).sum()
    print(f"Flagged {n_flagged}/{len(x_test)} samples as novel ({100*n_flagged/len(x_test):.2f}%)")
    
    # ========================================================================
    # Step 7: Visualize Embeddings
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("STEP 7: Visualizing Results")
    print("=" * 70)
    
    # Plot training curves
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    plot_training_curves(
        history=trainer.history,
        output_dir=OUTPUT_DIR,
        filename='training_curves.png'
    )
    print(f"Saved training curves to {OUTPUT_DIR}/training_curves.png")
    
    # ========================================================================
    # Done!
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("Example Complete!")
    print("=" * 70)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print("\nKey takeaways:")
    print("  - Siamese networks learn similarity, not direct classification")
    print("  - They work well with few examples per class (few-shot learning)")
    print("  - Can detect novel/unknown classes during deployment")
    print("  - Training uses episodic sampling for class balance")
    print("\nFor full training, use siamese_multisim/main.py")


if __name__ == '__main__':
    main()
