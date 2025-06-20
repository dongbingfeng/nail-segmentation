"""
Training Pipeline Module

Contains training loops, loss functions, and evaluation metrics.
"""

import sys
from pathlib import Path

# Add training root to path for imports
training_root = Path(__file__).parent.parent
sys.path.insert(0, str(training_root))

from .trainer import NailSegmentationTrainer
from losses.segmentation_losses import CombinedLoss, DiceLoss, FocalLoss
from metrics.segmentation_metrics import SegmentationMetrics

__all__ = [
    "NailSegmentationTrainer",
    "CombinedLoss",
    "DiceLoss", 
    "FocalLoss",
    "SegmentationMetrics"
] 