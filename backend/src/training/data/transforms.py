"""
Data Augmentation Pipeline for Nail Segmentation

Implements comprehensive data augmentation using Albumentations library,
optimized for small datasets with nail-specific transformations.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from typing import Dict, Any

import sys
from pathlib import Path

# Add training root to path for imports
training_root = Path(__file__).parent.parent
sys.path.insert(0, str(training_root))

from utils.config import TrainingConfig


def get_train_transforms(config: TrainingConfig) -> A.Compose:
    """
    Get training data augmentation pipeline
    
    Args:
        config: Training configuration
        
    Returns:
        Albumentations composition for training
    """
    aug_config = config.augmentation_config
    strength = config.augmentation_strength
    
    transforms = [
        # Geometric transformations
        A.ShiftScaleRotate(
            shift_limit=aug_config.get('shift_scale_rotate', {}).get('shift_limit', 0.1) * strength,
            scale_limit=aug_config.get('shift_scale_rotate', {}).get('scale_limit', 0.1) * strength,
            rotate_limit=aug_config.get('shift_scale_rotate', {}).get('rotate_limit', 15) * strength,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.7
        ),
        
        # Flipping transformations
        A.HorizontalFlip(p=aug_config.get('horizontal_flip', 0.5)),
        A.VerticalFlip(p=aug_config.get('vertical_flip', 0.2)),
        
        # Elastic transformation for natural deformation
        A.ElasticTransform(
            alpha=1,
            sigma=50,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.3 if aug_config.get('elastic_transform', True) else 0
        ),
        
        # Color and brightness transformations
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=aug_config.get('brightness', 0.2) * strength,
                contrast_limit=aug_config.get('contrast', 0.2) * strength,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=20 * strength,
                sat_shift_limit=aug_config.get('saturation', 0.2) * 100 * strength,
                val_shift_limit=20 * strength,
                p=1.0
            ),
        ], p=0.8),
        
        # Noise transformations
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.ISONoise(p=1.0),
        ], p=aug_config.get('gaussian_noise', 0.1)),
        
        # Blur transformations
        A.OneOf([
            A.MotionBlur(blur_limit=7, p=1.0),
            A.MedianBlur(blur_limit=7, p=1.0),
            A.GaussianBlur(blur_limit=7, p=1.0),
        ], p=aug_config.get('blur', 0.1)),
        
        # Cutout for regularization
        A.CoarseDropout(p=0.3),
        
        # Grid distortion for additional geometric variation
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.3 * strength,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.2
        ),
        
        # Optical distortion
        A.OpticalDistortion(
            distort_limit=0.2 * strength,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.2
        ),
        
        # Random crop and resize for scale variation
        A.RandomResizedCrop(
            height=config.image_size if isinstance(config.image_size, int) else config.image_size[1],
            width=config.image_size if isinstance(config.image_size, int) else config.image_size[0],
            scale=(0.8, 1.0),
            ratio=(0.8, 1.2),
            p=aug_config.get('random_crop', 0.3)
        ),
        
        # Ensure final size
        A.Resize(
            height=config.image_size if isinstance(config.image_size, int) else config.image_size[1],
            width=config.image_size if isinstance(config.image_size, int) else config.image_size[0],
            interpolation=cv2.INTER_LINEAR,
            p=1.0
        ),
        
        # Normalization (if not done in dataset)
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0 if not config.normalize_images else 0.0
        ),
    ]
    
    return A.Compose(
        transforms,
        additional_targets={'mask': 'mask'},
        p=1.0
    )


def get_val_transforms(config: TrainingConfig) -> A.Compose:
    """
    Get validation data transforms (minimal augmentation)
    
    Args:
        config: Training configuration
        
    Returns:
        Albumentations composition for validation
    """
    transforms = [
        # Only resize for validation
        A.Resize(
            height=config.image_size if isinstance(config.image_size, int) else config.image_size[1],
            width=config.image_size if isinstance(config.image_size, int) else config.image_size[0],
            interpolation=cv2.INTER_LINEAR,
            p=1.0
        ),
        
        # Normalization (if not done in dataset)
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0 if not config.normalize_images else 0.0
        ),
    ]
    
    return A.Compose(
        transforms,
        additional_targets={'mask': 'mask'},
        p=1.0
    )


def get_test_time_augmentation_transforms(config: TrainingConfig) -> A.Compose:
    """
    Get test-time augmentation transforms for inference
    
    Args:
        config: Training configuration
        
    Returns:
        Albumentations composition for TTA
    """
    transforms = [
        # Light augmentations for TTA
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.3),
        
        # Ensure final size
        A.Resize(
            height=config.image_size[1],
            width=config.image_size[0],
            interpolation=cv2.INTER_LINEAR,
            p=1.0
        ),
        
        # Normalization
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
    ]
    
    return A.Compose(
        transforms,
        additional_targets={'mask': 'mask'},
        p=1.0
    )


def get_nail_specific_transforms(config: TrainingConfig) -> A.Compose:
    """
    Get nail-specific augmentation transforms
    
    These transforms are designed specifically for nail segmentation,
    considering the anatomy and typical variations in nail appearance.
    
    Args:
        config: Training configuration
        
    Returns:
        Albumentations composition for nail-specific augmentation
    """
    transforms = [
        # Nail-specific color variations
        A.OneOf([
            # Simulate different nail polish colors
            A.HueSaturationValue(
                hue_shift_limit=30,
                sat_shift_limit=40,
                val_shift_limit=20,
                p=1.0
            ),
            # Simulate different lighting conditions
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=1.0
            ),
            # Simulate different skin tones
            A.RGBShift(
                r_shift_limit=20,
                g_shift_limit=20,
                b_shift_limit=20,
                p=1.0
            ),
        ], p=0.7),
        
        # Simulate different finger positions and angles
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=20,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            p=0.8
        ),
        
        # Simulate camera shake and motion blur
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=3, p=1.0),
        ], p=0.2),
        
        # Simulate different image qualities
        A.OneOf([
            A.ImageCompression(quality_lower=70, quality_upper=100, p=1.0),
            A.Downscale(scale_min=0.8, scale_max=0.99, p=1.0),
        ], p=0.3),
        
        # Add realistic noise
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 30.0), mean=0, p=1.0),
            A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3), p=1.0),
        ], p=0.2),
        
        # Simulate partial occlusion (jewelry, dirt, etc.)
        A.CoarseDropout(
            max_holes=2,
            max_height=20,
            max_width=20,
            min_holes=1,
            min_height=5,
            min_width=5,
            fill_value=0,
            mask_fill_value=0,
            p=0.2
        ),
    ]
    
    return A.Compose(
        transforms,
        additional_targets={'mask': 'mask'},
        p=1.0
    )


def create_mixup_transforms(alpha: float = 0.2) -> Dict[str, Any]:
    """
    Create MixUp augmentation parameters
    
    Args:
        alpha: MixUp alpha parameter
        
    Returns:
        MixUp configuration dictionary
    """
    return {
        'alpha': alpha,
        'enabled': True,
        'prob': 0.5
    }


def create_cutmix_transforms(alpha: float = 1.0) -> Dict[str, Any]:
    """
    Create CutMix augmentation parameters
    
    Args:
        alpha: CutMix alpha parameter
        
    Returns:
        CutMix configuration dictionary
    """
    return {
        'alpha': alpha,
        'enabled': True,
        'prob': 0.5
    }


def get_progressive_augmentation_schedule(epoch: int, total_epochs: int, config: TrainingConfig) -> A.Compose:
    """
    Get progressive augmentation schedule that increases difficulty over time
    
    Args:
        epoch: Current epoch
        total_epochs: Total number of epochs
        config: Training configuration
        
    Returns:
        Albumentations composition with progressive difficulty
    """
    # Calculate progression factor (0 to 1)
    progress = min(epoch / (total_epochs * 0.7), 1.0)  # Reach full strength at 70% of training
    
    # Adjust augmentation strength based on progress
    adjusted_config = config.augmentation_config.copy()
    base_strength = config.augmentation_strength
    
    # Start with lower strength and gradually increase
    current_strength = base_strength * (0.3 + 0.7 * progress)
    
    # Update config with current strength
    temp_config = TrainingConfig()
    temp_config.augmentation_config = adjusted_config
    temp_config.augmentation_strength = current_strength
    temp_config.image_size = config.image_size
    temp_config.normalize_images = config.normalize_images
    
    return get_train_transforms(temp_config) 