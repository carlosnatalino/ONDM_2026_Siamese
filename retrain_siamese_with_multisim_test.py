#!/usr/bin/env python3
"""
Retrain Siamese Multi-Similarity Network with full multi-similarity in testing.

This script trains the Siamese network using all similarity metrics (L1, L2, Cosine,
Product + learned attention) in BOTH training and testing phases.

The key difference from before is that the evaluation now uses the full comparison
head instead of just cosine similarity.

Author: Andrei Ribeiro
Date: January 27, 2026
"""

import sys
import logging
from pathlib import Path

# Add siamese_multisim to path
sys.path.insert(0, str(Path(__file__).parent))

from siamese_multisim.main import main

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("="*70)
    logger.info("Retraining Siamese with Multi-Similarity in Testing")
    logger.info("="*70)
    logger.info("Configuration:")
    logger.info("  - Training: Multi-similarity head (L1, L2, Cosine, Product + Attention)")
    logger.info("  - Testing: Multi-similarity head (NOT just cosine)")
    logger.info("  - N-way K-shot: Using full comparison head")
    logger.info("="*70)
    
    main()
