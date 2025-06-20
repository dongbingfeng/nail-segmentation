"""
Dataset Implementation for Nail Segmentation Training

Provides PyTorch Dataset class for loading nail images and segmentation masks
with proper data splitting and validation.
"""

import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from sklearn.model_selection import train_test_split
import imageio
import sys
from pathlib import Path as PathLib

# Add training root to path for imports
training_root = PathLib(__file__).parent.parent
sys.path.insert(0, str(training_root))

from utils.config import TrainingConfig
from data.transforms import get_train_transforms, get_val_transforms
from data.utils import load_annotations, validate_data_format


class NailSegmentationDataset(Dataset):
    """
    PyTorch Dataset for nail segmentation training
    
    Loads images and corresponding segmentation masks from annotation data.
    Supports data augmentation and proper train/validation splitting.
    """
    
    def __init__(
        self,
        data_dir: str,
        transform=None,
        split: str = "train",
        config: Optional[TrainingConfig] = None
    ):
        """
        Initialize dataset
        
        Args:
            data_dir: Directory containing images
            transform: Data augmentation transforms
            split: Dataset split ('train' or 'val')
            config: Training configuration
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split
        self.config = config or TrainingConfig()
        
        # Load and validate annotations
        self._validate_dataset()
        
        # Filter annotations for segmentation data
        self.segmentation_data = self._filter_segmentation_annotations()
        
        if len(self.segmentation_data) == 0:
            raise ValueError(f"No segmentation annotations found")
        
        print(f"Loaded {len(self.segmentation_data)} samples for {split} split")
    
    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.segmentation_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get dataset item
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing image and mask tensors
        """
        # Get sample data
        seg_data = self.segmentation_data[idx]
        
        # Load image
        image_path = seg_data['image_path']
        image = self._load_image(str(image_path), 'jpg')
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Load mask
        if 'mask_path' in seg_data:
            mask_path = seg_data['mask_path']
            mask = self._load_image(str(mask_path), 'gif')
        
        elif 'annotation_path' in seg_data:
            annotation_path = seg_data['annotation_path']
            mask = self._load_annotation(str(annotation_path))
        
        else:
            raise ValueError(f"No mask or annotation path found for image: {image_path}")
        
        if mask is None:
            raise ValueError(f"Failed to create mask for image: {mask_path}")
        
        # Validate sample
        if not self._validate_sample(image, mask):
            raise ValueError(f"Invalid sample at index {idx}")
        
        # Resize to target size
        if isinstance(self.config.image_size, int):
            target_size = (self.config.image_size, self.config.image_size)
        else:
            target_size = (self.config.image_size[1], self.config.image_size[0])
        
        if image.shape[:2] != target_size:
            print(f"Warning: Image size mismatch at index {idx}. "
                  f"Expected {target_size}, got ({image.shape[:2][1]}, {image.shape[:2][0]})")
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        
        # Apply transforms
        if self.transform:
            # Albumentations expects HWC format
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Convert to tensors
        if isinstance(image, np.ndarray):
            # Convert from HWC to CHW format
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        if isinstance(mask, np.ndarray):
            # Ensure mask is single channel
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            mask = torch.from_numpy(mask).float()
            # Add channel dimension
            mask = mask.unsqueeze(0)
        
        # Normalize image
        if self.config.normalize_images:
            image = image / 255.0
        
        # Normalize mask to [0, 1]
        mask = mask / 255.0
        
        return {
            'image': image,
            'mask': mask,
            'image_path': str(image_path),
            'annotation_id': seg_data.get('id', idx)
        }
    
    def _load_image(self, image_path: str, image_type: str = 'jpg') -> Optional[np.ndarray]:
        """
        Load image from file
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image as numpy array in RGB format
        """
        try:
            # Load image
            if image_type == 'jpg' or image_type == 'bmp':
                image = cv2.imread(image_path)
            elif image_type == 'gif':
                ## Read the gif from disk to `RGB`s using `imageio.miread` 
                gif = imageio.mimread(image_path)
                assert len(gif) == 1, "Gif should have only one frame"
                # convert form RGB to BGR 
                image = cv2.cvtColor(gif[0], cv2.COLOR_RGB2BGR)
            else:
                raise ValueError(f"Unsupported image type: {image_type}")
            
            if image is None:
                return None
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return image
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def _load_annotation(self, annotation_path: str, image_id: str = None) -> Optional[np.ndarray]:
        """
        Create segmentation mask from annotation data
        
        Args:
            segmentation_data: Segmentation annotation data
            
        Returns:
            Binary mask as numpy array
        """
        try:
            with open(annotation_path, 'r') as f:
                annotation = json.load(f)
            if image_id is not None and annotation['image_id'] != image_id:
                raise ValueError(f"Image ID mismatch: {annotation['image_id']} != {image_id}")
            
            if 'annotations' not in annotation:
                raise ValueError(f"No annotations found in {annotation_path}")
            
            for annotation in annotation['annotations']:
                if not annotation['id'].startswith("mask-"):
                    continue
                points = annotation['coordinates']['points']
                if not points:
                    return None
                
                # Convert points to numpy array
                mask_points = np.array([[p['x'], p['y']] for p in points], dtype=np.int32)
                
                # Create mask from points (assuming they form a polygon)
                # For now, create a simple mask - this should be adapted based on actual data format
                if isinstance(self.config.image_size, int):
                    mask_size = (self.config.image_size, self.config.image_size)
                else:
                    mask_size = (self.config.image_size[1], self.config.image_size[0])
                mask = np.zeros(mask_size, dtype=np.uint8)
                
                if len(mask_points) >= 3:  # Need at least 3 points for a polygon
                    cv2.fillPoly(mask, [mask_points], 255)
                
                # Save mask as binary image file (for debugging)
                # if mask is not None and mask.max() > 0:
                #     mask_filename = f"mask_{image_id}.png" if image_id else "mask.png"
                #     mask_path = os.path.join(os.path.dirname(annotation_path), mask_filename)
                #     cv2.imwrite(mask_path, mask)
            return mask
            
        except Exception as e:
            print(f"Error creating mask: {e}")
            return None

    def _validate_sample(self, image: np.ndarray, mask: np.ndarray) -> bool:
        """
        Validate image and mask pair
        
        Args:
            image: Input image
            mask: Segmentation mask
            
        Returns:
            True if sample is valid
        """
        try:
            # Check if image and mask are not None
            if image is None or mask is None:
                return False
            
            # Check image format
            if len(image.shape) != 3 or image.shape[2] != 3:
                return False
            
            # Check mask format
            if len(mask.shape) not in [2, 3]:
                return False
            
            # Check if mask has valid values
            if mask.max() == 0:  # Empty mask
                return False
            
            return True
            
        except Exception:
            return False
    
    def _filter_segmentation_annotations(self) -> List[Dict[str, Any]]:
        """
        Filter annotations to include only those with segmentation data
        
        Returns:
            List of annotations with segmentation data
        """
        segmentation_data = []
        
        image_dir = os.path.join(self.data_dir, "images")
        mask_dir = os.path.join(self.data_dir, "masks")
        annotations_dir = os.path.join(self.data_dir, "annotations")
        if not os.path.exists(image_dir):
            raise ValueError(f"Image directory does not exist: {image_dir}")
        
        if os.path.exists(mask_dir):
            for image_file in os.listdir(image_dir):
                if not (image_file.endswith('.jpg') or image_file.endswith('.bmp')):
                    continue
                image_id = os.path.splitext(image_file)[0]
                image_path = os.path.join(image_dir, image_file)
                mask_path = os.path.join(mask_dir, f"{image_id}_mask.gif")
                if not os.path.exists(mask_path):
                    print(f"Warning: No mask found for {image_file}: {mask_path}")
                    continue
                    
                segmentation_data.append({
                    'image_id': image_id,
                    'image_path': image_path,
                    'mask_path': mask_path,
                    'id': image_id
                })
        elif os.path.exists(annotations_dir):
            for image_file in os.listdir(image_dir):
                if not (image_file.endswith('.jpg') or image_file.endswith('.bmp')):
                    continue
                image_id = os.path.splitext(image_file)[0]
                image_path = os.path.join(image_dir, image_file)
                annotation_path = os.path.join(annotations_dir, f"{image_id}_annotations.json")
                if not os.path.exists(annotation_path):
                    print(f"Warning: No annotation found for {image_file}: {annotation_path}")
                    continue

                segmentation_data.append({
                    'image_id': image_id,
                    'image_path': image_path,
                    'annotation_path': annotation_path,
                    'id': image_id
                })
        else:
            raise ValueError(f"No image or mask directory or annotations directory found: {self.data_dir}")
        return segmentation_data
    
    def _validate_dataset(self) -> None:
        """Validate dataset integrity"""
        
        # Check if data directory exists
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {self.data_dir}")
        
        # Validate data format
        #validate_data_format(self.annotations)

def create_data_loaders(
    config: TrainingConfig,
    data_dir: str,
    annotations_file: str
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders
    
    Args:
        config: Training configuration
        data_dir: Directory containing images
        annotations_file: Path to annotations file
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Load full dataset
    full_dataset = NailSegmentationDataset(
        data_dir=data_dir,
        annotations_file=annotations_file,
        transform=None,  # Will be set per split
        config=config
    )
    
    # Split dataset
    total_size = len(full_dataset)
    val_size = int(total_size * config.validation_split)
    train_size = total_size - val_size
    
    # Use random split for now - could be improved with stratified splitting
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create separate dataset instances with appropriate transforms
    train_data = NailSegmentationDataset(
        data_dir=data_dir,
        annotations_file=annotations_file,
        transform=get_train_transforms(config),
        split="train",
        config=config
    )
    
    val_data = NailSegmentationDataset(
        data_dir=data_dir,
        annotations_file=annotations_file,
        transform=get_val_transforms(config),
        split="val",
        config=config
    )
    
    # Apply the split indices
    train_data.segmentation_data = [train_data.segmentation_data[i] for i in train_dataset.indices]
    val_data.segmentation_data = [val_data.segmentation_data[i] for i in val_dataset.indices]
    
    # Create data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.get_num_workers(),
        pin_memory=config.pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.get_num_workers(),
        pin_memory=config.pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader
