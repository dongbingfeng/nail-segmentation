"""
Nail Segmentation Training Module

This module provides a complete training pipeline for U-Net based nail segmentation models.
Includes data loading, model architecture, training loops, and evaluation metrics.
"""

__version__ = "1.0.0"
__author__ = "Nail Segmentation Team"

from .models.unet import AttentionUNet
from .data.dataset import NailSegmentationDataset, create_data_loaders
from .training.trainer import NailSegmentationTrainer
from .utils.config import TrainingConfig

__all__ = [
    "AttentionUNet",
    "NailSegmentationDataset", 
    "create_data_loaders",
    "NailSegmentationTrainer",
    "TrainingConfig"
] 