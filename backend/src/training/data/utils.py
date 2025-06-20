"""
Data Utilities for Nail Segmentation Training

Provides utilities for loading annotations, validating data format,
and processing image/mask data.
"""

import json
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def load_annotations(annotations_file: str) -> Dict[str, Any]:
    """
    Load annotations from JSON file
    
    Args:
        annotations_file: Path to annotations JSON file
        
    Returns:
        Dictionary containing annotation data
    """
    try:
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        logger.info(f"Loaded annotations from {annotations_file}")
        logger.info(f"Found {len(annotations)} images with annotations")
        
        return annotations
        
    except FileNotFoundError:
        logger.error(f"Annotations file not found: {annotations_file}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in {annotations_file}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading annotations: {e}")
        raise


def validate_data_format(annotations: Dict[str, Any]) -> bool:
    """
    Validate annotation data format
    
    Args:
        annotations: Annotation data dictionary
        
    Returns:
        True if format is valid
        
    Raises:
        ValueError: If format is invalid
    """
    if not isinstance(annotations, dict):
        raise ValueError("Annotations must be a dictionary")
    
    segmentation_count = 0
    
    for image_id, image_data in annotations.items():
        # Validate image data structure
        if not isinstance(image_data, dict):
            raise ValueError(f"Image data for {image_id} must be a dictionary")
        
        # Check for required fields
        if 'path' not in image_data:
            logger.warning(f"Image {image_id} missing 'path' field")
        
        # Validate annotations
        if 'annotations' in image_data:
            if not isinstance(image_data['annotations'], list):
                raise ValueError(f"Annotations for {image_id} must be a list")
            
            for annotation in image_data['annotations']:
                if not isinstance(annotation, dict):
                    raise ValueError(f"Annotation in {image_id} must be a dictionary")
                
                # Check annotation type
                if annotation.get('type') == 'segmentation':
                    segmentation_count += 1
                    
                    # Validate segmentation data
                    if 'points' not in annotation:
                        logger.warning(f"Segmentation annotation in {image_id} missing 'points'")
                    else:
                        points = annotation['points']
                        if not isinstance(points, list):
                            raise ValueError(f"Points in {image_id} must be a list")
                        
                        for point in points:
                            if not isinstance(point, dict) or 'x' not in point or 'y' not in point:
                                raise ValueError(f"Invalid point format in {image_id}")
    
    logger.info(f"Validation passed: {segmentation_count} segmentation annotations found")
    
    if segmentation_count == 0:
        logger.warning("No segmentation annotations found in dataset")
    
    return True


def create_mask_from_points(points: List[Dict[str, float]], image_size: Tuple[int, int]) -> np.ndarray:
    """
    Create binary mask from list of points
    
    Args:
        points: List of points with 'x' and 'y' coordinates
        image_size: Target image size (width, height)
        
    Returns:
        Binary mask as numpy array
    """
    mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
    
    if len(points) < 3:
        logger.warning("Need at least 3 points to create a polygon mask")
        return mask
    
    # Convert points to numpy array
    polygon_points = np.array([[int(p['x']), int(p['y'])] for p in points], dtype=np.int32)
    
    # Create mask using fillPoly
    cv2.fillPoly(mask, [polygon_points], 255)
    
    return mask


def validate_image_mask_pair(image_path: str, mask: np.ndarray) -> bool:
    """
    Validate image and mask compatibility
    
    Args:
        image_path: Path to image file
        mask: Segmentation mask
        
    Returns:
        True if valid pair
    """
    try:
        # Load image to check dimensions
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Cannot load image: {image_path}")
            return False
        
        # Check if mask has valid content
        if mask.max() == 0:
            logger.warning(f"Empty mask for image: {image_path}")
            return False
        
        # Check mask values
        unique_values = np.unique(mask)
        if not all(val in [0, 255] for val in unique_values):
            logger.warning(f"Mask contains non-binary values for image: {image_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating image-mask pair {image_path}: {e}")
        return False


def compute_dataset_statistics(data_dir: str, annotations: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute dataset statistics
    
    Args:
        data_dir: Directory containing images
        annotations: Annotation data
        
    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        'total_images': len(annotations),
        'segmentation_annotations': 0,
        'image_sizes': [],
        'mask_areas': [],
        'mask_coverage': []
    }
    
    data_path = Path(data_dir)
    
    for image_id, image_data in annotations.items():
        image_path = data_path / image_data.get('path', f"{image_id}.jpg")
        
        if not image_path.exists():
            continue
        
        # Load image to get size
        image = cv2.imread(str(image_path))
        if image is not None:
            h, w = image.shape[:2]
            stats['image_sizes'].append((w, h))
        
        # Process segmentation annotations
        if 'annotations' in image_data:
            for annotation in image_data['annotations']:
                if annotation.get('type') == 'segmentation' and 'points' in annotation:
                    stats['segmentation_annotations'] += 1
                    
                    # Create mask and compute statistics
                    if image is not None:
                        mask = create_mask_from_points(annotation['points'], (w, h))
                        mask_area = np.sum(mask > 0)
                        total_area = w * h
                        
                        stats['mask_areas'].append(mask_area)
                        stats['mask_coverage'].append(mask_area / total_area)
    
    # Compute summary statistics
    if stats['image_sizes']:
        widths, heights = zip(*stats['image_sizes'])
        stats['avg_width'] = np.mean(widths)
        stats['avg_height'] = np.mean(heights)
        stats['min_width'] = np.min(widths)
        stats['max_width'] = np.max(widths)
        stats['min_height'] = np.min(heights)
        stats['max_height'] = np.max(heights)
    
    if stats['mask_areas']:
        stats['avg_mask_area'] = np.mean(stats['mask_areas'])
        stats['min_mask_area'] = np.min(stats['mask_areas'])
        stats['max_mask_area'] = np.max(stats['mask_areas'])
    
    if stats['mask_coverage']:
        stats['avg_mask_coverage'] = np.mean(stats['mask_coverage'])
        stats['min_mask_coverage'] = np.min(stats['mask_coverage'])
        stats['max_mask_coverage'] = np.max(stats['mask_coverage'])
    
    return stats


def split_dataset_by_quality(annotations: Dict[str, Any], quality_threshold: float = 0.1) -> Tuple[List[str], List[str]]:
    """
    Split dataset into high and low quality samples based on mask coverage
    
    Args:
        annotations: Annotation data
        quality_threshold: Minimum mask coverage for high quality
        
    Returns:
        Tuple of (high_quality_ids, low_quality_ids)
    """
    high_quality = []
    low_quality = []
    
    for image_id, image_data in annotations.items():
        if 'annotations' not in image_data:
            low_quality.append(image_id)
            continue
        
        has_good_segmentation = False
        
        for annotation in image_data['annotations']:
            if annotation.get('type') == 'segmentation' and 'points' in annotation:
                # Estimate quality based on number of points
                num_points = len(annotation['points'])
                if num_points >= 10:  # Assume more points = better quality
                    has_good_segmentation = True
                    break
        
        if has_good_segmentation:
            high_quality.append(image_id)
        else:
            low_quality.append(image_id)
    
    logger.info(f"Dataset split: {len(high_quality)} high quality, {len(low_quality)} low quality")
    
    return high_quality, low_quality


def create_curriculum_learning_order(annotations: Dict[str, Any]) -> List[str]:
    """
    Create curriculum learning order from easy to hard samples
    
    Args:
        annotations: Annotation data
        
    Returns:
        List of image IDs ordered from easy to hard
    """
    sample_difficulties = []
    
    for image_id, image_data in annotations.items():
        difficulty_score = 0
        
        if 'annotations' in image_data:
            for annotation in image_data['annotations']:
                if annotation.get('type') == 'segmentation' and 'points' in annotation:
                    num_points = len(annotation['points'])
                    
                    # More points = more complex shape = higher difficulty
                    difficulty_score += num_points
                    
                    # Add other difficulty factors here
                    # e.g., mask area, shape complexity, etc.
        
        sample_difficulties.append((image_id, difficulty_score))
    
    # Sort by difficulty (ascending - easy to hard)
    sample_difficulties.sort(key=lambda x: x[1])
    
    ordered_ids = [item[0] for item in sample_difficulties]
    
    logger.info(f"Created curriculum learning order for {len(ordered_ids)} samples")
    
    return ordered_ids


def preprocess_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Preprocess image for training
    
    Args:
        image: Input image
        target_size: Target size (width, height)
        
    Returns:
        Preprocessed image
    """
    # Resize image
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Convert to RGB if needed
    if len(resized.shape) == 3 and resized.shape[2] == 3:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    return resized


def preprocess_mask(mask: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Preprocess mask for training
    
    Args:
        mask: Input mask
        target_size: Target size (width, height)
        
    Returns:
        Preprocessed mask
    """
    # Resize mask
    resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    
    # Ensure binary values
    resized = (resized > 127).astype(np.uint8) * 255
    
    return resized


def save_dataset_split(train_ids: List[str], val_ids: List[str], output_path: str) -> None:
    """
    Save dataset split to file
    
    Args:
        train_ids: Training image IDs
        val_ids: Validation image IDs
        output_path: Output file path
    """
    split_data = {
        'train': train_ids,
        'val': val_ids,
        'train_count': len(train_ids),
        'val_count': len(val_ids)
    }
    
    with open(output_path, 'w') as f:
        json.dump(split_data, f, indent=2)
    
    logger.info(f"Dataset split saved to {output_path}")


def load_dataset_split(split_path: str) -> Tuple[List[str], List[str]]:
    """
    Load dataset split from file
    
    Args:
        split_path: Path to split file
        
    Returns:
        Tuple of (train_ids, val_ids)
    """
    with open(split_path, 'r') as f:
        split_data = json.load(f)
    
    train_ids = split_data['train']
    val_ids = split_data['val']
    
    logger.info(f"Loaded dataset split: {len(train_ids)} train, {len(val_ids)} val")
    
    return train_ids, val_ids 