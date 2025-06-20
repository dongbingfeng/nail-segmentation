"""
Training-Compatible Preprocessing Pipeline

This module provides preprocessing functionality that exactly matches the training pipeline
to ensure consistent model behavior between training and serving.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from typing import Union, List, Tuple, Dict, Any
import logging

# Import training transforms for exact compatibility
try:
    from training.data.transforms import get_train_transforms, get_val_transforms
    TRAINING_TRANSFORMS_AVAILABLE = True
except ImportError:
    TRAINING_TRANSFORMS_AVAILABLE = False
    logging.warning("Training transforms not available, using fallback preprocessing")

from src.config.unet_config import UNetConfig

logger = logging.getLogger(__name__)


class TrainingCompatiblePreprocessor:
    """
    Preprocessing pipeline that maintains exact compatibility with training transforms.
    
    This preprocessor ensures that inference preprocessing matches the training pipeline
    exactly, including normalization statistics, resizing methods, and tensor formats.
    """
    
    def __init__(self, config: UNetConfig):
        """
        Initialize preprocessor with training-compatible transforms.
        
        Args:
            config: U-Net configuration containing preprocessing parameters
        """
        self.config = config
        self.device = torch.device('cpu')  # Will be set by service
        
        # Create preprocessing transforms
        self._create_transforms()
        
        logger.info(f"Preprocessor initialized with image size: {self.config.preprocessing.image_size}")
    
    def _create_transforms(self) -> None:
        """Create preprocessing transform pipeline."""
        image_size = tuple(self.config.preprocessing.image_size)
        
        if TRAINING_TRANSFORMS_AVAILABLE:
            # Use exact training transforms for validation (no augmentation)
            try:
                self.transforms = get_val_transforms(image_size=image_size)
                logger.info("Using training-compatible validation transforms")
                return
            except Exception as e:
                logger.warning(f"Failed to load training transforms: {e}, using fallback")
        
        # Fallback transforms that match training preprocessing
        transform_list = [
            transforms.Resize(image_size, interpolation=Image.BILINEAR),
        ]
        
        if self.config.preprocessing.normalize:
            transform_list.extend([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.config.preprocessing.mean,
                    std=self.config.preprocessing.std
                )
            ])
        else:
            transform_list.append(transforms.ToTensor())
        
        self.transforms = transforms.Compose(transform_list)
        logger.info("Using fallback preprocessing transforms")
    
    def set_device(self, device: torch.device) -> None:
        """
        Set target device for tensor operations.
        
        Args:
            device: Target device for tensor placement
        """
        self.device = device
    
    def _validate_input(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """
        Validate and convert input to PIL Image.
        
        Args:
            image: Input image as PIL Image or numpy array
            
        Returns:
            PIL Image in RGB format
        """
        if isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            if image.ndim == 3:
                if image.shape[2] == 3:
                    # RGB image
                    image = Image.fromarray(image.astype(np.uint8), mode='RGB')
                elif image.shape[2] == 4:
                    # RGBA image, convert to RGB
                    image = Image.fromarray(image.astype(np.uint8), mode='RGBA')
                    image = image.convert('RGB')
                else:
                    raise ValueError(f"Unsupported number of channels: {image.shape[2]}")
            elif image.ndim == 2:
                # Grayscale image, convert to RGB
                image = Image.fromarray(image.astype(np.uint8), mode='L')
                image = image.convert('RGB')
            else:
                raise ValueError(f"Unsupported image dimensions: {image.ndim}")
        
        elif isinstance(image, Image.Image):
            # Ensure RGB format
            if image.mode != 'RGB':
                image = image.convert('RGB')
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        return image
    
    def _preprocess_single_image(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Preprocess a single image.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed tensor with shape (1, C, H, W)
        """
        # Validate and convert input
        pil_image = self._validate_input(image)
        
        # Validate image dimensions
        if pil_image.size[0] == 0 or pil_image.size[1] == 0:
            raise ValueError("Invalid image dimensions")
        
        # Apply transforms
        try:
            tensor = self.transforms(pil_image)
            
            # Ensure correct shape and data type
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)  # Add batch dimension
            
            # Ensure correct data type
            if self.config.preprocessing.dtype == 'float32':
                tensor = tensor.float()
            elif self.config.preprocessing.dtype == 'float16':
                tensor = tensor.half()
            
            return tensor
            
        except Exception as e:
            logger.error(f"Transform failed for image of size {pil_image.size}: {e}")
            raise
    
    def preprocess(self, inputs: Union[Image.Image, List[Image.Image], np.ndarray]) -> torch.Tensor:
        """
        Preprocess input images for inference.
        
        Args:
            inputs: Single image, list of images, or numpy array
            
        Returns:
            Preprocessed tensor with shape (B, C, H, W) ready for inference
        """
        try:
            if isinstance(inputs, list):
                # Batch processing
                if len(inputs) == 0:
                    raise ValueError("Empty image list")
                
                if len(inputs) > self.config.inference.batch_size:
                    raise ValueError(f"Batch size {len(inputs)} exceeds maximum {self.config.inference.batch_size}")
                
                # Process each image
                processed_tensors = []
                for i, image in enumerate(inputs):
                    try:
                        tensor = self._preprocess_single_image(image)
                        processed_tensors.append(tensor)
                    except Exception as e:
                        logger.error(f"Failed to preprocess image {i} in batch: {e}")
                        raise
                
                # Concatenate into batch
                batch_tensor = torch.cat(processed_tensors, dim=0)
                
            else:
                # Single image processing
                batch_tensor = self._preprocess_single_image(inputs)
            
            # Move to target device
            batch_tensor = batch_tensor.to(self.device)
            
            # Validate output tensor
            expected_channels = 3
            expected_height, expected_width = self.config.preprocessing.image_size
            
            if batch_tensor.shape[1] != expected_channels:
                raise ValueError(f"Unexpected channel count: {batch_tensor.shape[1]}, expected {expected_channels}")
            
            if batch_tensor.shape[2] != expected_height or batch_tensor.shape[3] != expected_width:
                raise ValueError(f"Unexpected spatial dimensions: {batch_tensor.shape[2:4]}, expected ({expected_height}, {expected_width})")
            
            logger.debug(f"Preprocessed tensor shape: {batch_tensor.shape}, device: {batch_tensor.device}")
            
            return batch_tensor
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise
    
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """
        Get information about the preprocessing pipeline.
        
        Returns:
            Dictionary containing preprocessing configuration and capabilities
        """
        return {
            'image_size': self.config.preprocessing.image_size,
            'normalize': self.config.preprocessing.normalize,
            'mean': self.config.preprocessing.mean if self.config.preprocessing.normalize else None,
            'std': self.config.preprocessing.std if self.config.preprocessing.normalize else None,
            'dtype': self.config.preprocessing.dtype,
            'max_batch_size': self.config.inference.batch_size,
            'training_compatible': TRAINING_TRANSFORMS_AVAILABLE,
            'device': str(self.device)
        }
    
    def preprocess_for_visualization(self, image: Union[Image.Image, np.ndarray]) -> Tuple[torch.Tensor, Image.Image]:
        """
        Preprocess image and return both tensor and PIL image for visualization.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (preprocessed_tensor, resized_pil_image)
        """
        # Validate input
        pil_image = self._validate_input(image)
        
        # Create resized image for visualization (without normalization)
        target_size = tuple(self.config.preprocessing.image_size)
        resized_image = pil_image.resize(target_size, Image.BILINEAR)
        
        # Create preprocessed tensor
        preprocessed_tensor = self.preprocess(image)
        
        return preprocessed_tensor, resized_image
    
    def denormalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Denormalize a tensor that was normalized during preprocessing.
        
        Args:
            tensor: Normalized tensor with shape (B, C, H, W)
            
        Returns:
            Denormalized tensor in range [0, 1]
        """
        if not self.config.preprocessing.normalize:
            return tensor
        
        mean = torch.tensor(self.config.preprocessing.mean).view(1, 3, 1, 1)
        std = torch.tensor(self.config.preprocessing.std).view(1, 3, 1, 1)
        
        # Move to same device as tensor
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
        
        # Denormalize
        denormalized = tensor * std + mean
        
        # Clamp to valid range
        denormalized = torch.clamp(denormalized, 0.0, 1.0)
        
        return denormalized
    
    def tensor_to_pil(self, tensor: torch.Tensor, denormalize: bool = True) -> Image.Image:
        """
        Convert preprocessed tensor back to PIL Image.
        
        Args:
            tensor: Input tensor with shape (1, C, H, W) or (C, H, W)
            denormalize: Whether to denormalize the tensor first
            
        Returns:
            PIL Image in RGB format
        """
        # Handle batch dimension
        if tensor.dim() == 4:
            if tensor.shape[0] != 1:
                raise ValueError("Batch size must be 1 for PIL conversion")
            tensor = tensor.squeeze(0)
        elif tensor.dim() != 3:
            raise ValueError(f"Expected 3D or 4D tensor, got {tensor.dim()}D")
        
        # Denormalize if needed
        if denormalize and self.config.preprocessing.normalize:
            tensor = tensor.unsqueeze(0)  # Add batch dim for denormalization
            tensor = self.denormalize_tensor(tensor)
            tensor = tensor.squeeze(0)  # Remove batch dim
        
        # Convert to numpy
        numpy_array = tensor.detach().cpu().numpy()
        
        # Transpose from CHW to HWC
        numpy_array = np.transpose(numpy_array, (1, 2, 0))
        
        # Convert to 0-255 range
        numpy_array = (numpy_array * 255).astype(np.uint8)
        
        # Create PIL image
        pil_image = Image.fromarray(numpy_array, mode='RGB')
        
        return pil_image 