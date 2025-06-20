"""
U-Net Model Service Implementation

This module provides the main U-Net serving service that inherits from BaseModelService
and implements U-Net specific model loading, inference, and management.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import threading
import time
import uuid
import asyncio
from datetime import datetime
from contextlib import contextmanager

from src.serving.base_service import BaseModelService
from src.config.unet_config import get_unet_config, UNetConfig
from src.training.models.unet import AttentionUNet
from src.unet.preprocessing import TrainingCompatiblePreprocessor
from src.unet.postprocessing import UNetPostprocessor

logger = logging.getLogger(__name__)


class UNetModelService(BaseModelService):
    """
    U-Net model serving service with eager loading and memory pool management.
    
    This service provides thread-safe U-Net inference with:
    - Eager model loading at initialization
    - Memory pool management for efficient tensor allocation
    - Training-compatible preprocessing pipeline
    - Comprehensive error handling and logging
    """
    
    def __init__(self, config: Optional[UNetConfig] = None, lazy_loading: bool = False):
        """
        Initialize U-Net service with optional lazy model loading.
        
        Args:
            config: U-Net configuration. If None, uses global config.
            lazy_loading: If True, skip eager model loading for testing purposes.
        """
        self.config = config or get_unet_config()
        self.model = None
        self.device = None
        self.preprocessor = None
        self.postprocessor = None
        self._model_loaded = False
        self._inference_lock = threading.RLock()
        self._load_start_time = None
        
        # Initialize base service
        super().__init__(
            config=self.config,
            service_name="unet"
        )
        
        # Conditional model loading
        if not lazy_loading:
            logger.info("Starting eager U-Net model loading...")
            self._load_start_time = time.time()
            try:
                self.load_model()
                self._model_loaded = True
                load_time = time.time() - self._load_start_time
                logger.info(f"U-Net model loaded successfully in {load_time:.2f}s")
            except Exception as e:
                logger.error(f"Failed to load U-Net model during initialization: {e}")
                self._model_loaded = False
                if not self.config.model.fallback_to_cpu:
                    raise
        else:
            logger.info("Lazy loading enabled, skipping model initialization")
    
    def load_model(self) -> bool:
        """
        Load U-Net model from checkpoint with device placement.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            # Determine device
            self.device = self._determine_device()
            logger.info(f"Using device: {self.device}")
            
            # Find best model checkpoint
            model_path = self.config.get_best_model_path()
            if not model_path or not model_path.exists():
                raise FileNotFoundError(f"No valid model checkpoint found in {self.config.model.checkpoint_dir}")
            
            logger.info(f"Loading model from: {model_path}")
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract model parameters
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                model_config = checkpoint.get('model_config', {})
            else:
                # Direct state dict
                state_dict = checkpoint
                model_config = {}
            
            # Create model instance
            self.model = self._create_model_instance(model_config)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            # Initialize preprocessor and postprocessor
            self.preprocessor = TrainingCompatiblePreprocessor(self.config)
            self.postprocessor = UNetPostprocessor(self.config)
            
            # Initialize memory pools (note: this is async but we'll handle it in the base class)
            # self.initialize_memory_pools()  # Skip for now to avoid async warning
            
            # Warm up model with dummy inference
            if self.config.health.dummy_inference_test:
                self._warm_up_model()
            
            logger.info("U-Net model loaded and ready for inference")
            return True
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            if self.config.model.fallback_to_cpu and self.device != 'cpu':
                logger.info("Attempting fallback to CPU...")
                self.config.model.device = 'cpu'
                return self.load_model()
            return False
    
    def _determine_device(self) -> torch.device:
        """Determine the best device for model execution."""
        device_config = self.config.model.device.lower()
        
        if device_config == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                logger.info(f"Auto-selected CUDA device: {torch.cuda.get_device_name()}")
            else:
                device = torch.device('cpu')
                logger.info("Auto-selected CPU device (CUDA not available)")
        elif device_config == 'cuda':
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                if self.config.model.fallback_to_cpu:
                    device = torch.device('cpu')
                    logger.warning("CUDA requested but not available, falling back to CPU")
                else:
                    raise RuntimeError("CUDA requested but not available")
        else:
            device = torch.device('cpu')
        
        return device
    
    def _create_model_instance(self, model_config: Dict[str, Any]) -> AttentionUNet:
        """
        Create U-Net model instance based on configuration.
        
        Args:
            model_config: Model configuration from checkpoint
            
        Returns:
            AttentionUNet instance
        """
        from src.training.utils.config import TrainingConfig
        
        # Create default training config parameters for model creation
        training_config_params = {
            'input_channels': 3,
            'output_channels': 1,
            'base_channels': 64,
            'depth': 4,
            'attention': True,
            'dropout': 0.1,
            'batch_norm': True,
            'image_size': self.config.preprocessing.image_size,
        }
        
        # Override with checkpoint config if available
        if model_config:
            training_config_params.update(model_config)
        
        # Adjust parameters based on model variant
        variant = self.config.model.model_variant
        if variant == 'lightweight':
            training_config_params['base_channels'] = 32
            training_config_params['depth'] = 3
        elif variant == 'deep':
            training_config_params['base_channels'] = 96
            training_config_params['depth'] = 5
        
        # Create TrainingConfig instance
        training_config = TrainingConfig(**training_config_params)
        
        logger.info(f"Creating {variant} U-Net model with config: {training_config_params}")
        return AttentionUNet(training_config)
    
    def _warm_up_model(self) -> None:
        """Warm up model with dummy inference to allocate memory pools."""
        try:
            logger.info("Warming up U-Net model...")
            
            # Create dummy input
            dummy_input = torch.randn(
                1, 3, 
                self.config.preprocessing.image_size[0],
                self.config.preprocessing.image_size[1],
                device=self.device
            )
            
            with torch.no_grad():
                start_time = time.time()
                _ = self.model(dummy_input)
                warm_up_time = time.time() - start_time
                logger.info(f"Model warm-up completed in {warm_up_time:.3f}s")
                
        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}")
    
    def preprocess(self, inputs: Union[Image.Image, List[Image.Image], np.ndarray]) -> torch.Tensor:
        """
        Preprocess input images for U-Net inference.
        
        Args:
            inputs: Single image, list of images, or numpy array
            
        Returns:
            Preprocessed tensor ready for inference
        """
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor not initialized. Model loading may have failed.")
        
        return self.preprocessor.preprocess(inputs)
    
    def inference(self, preprocessed_input: torch.Tensor) -> torch.Tensor:
        """
        Perform thread-safe U-Net inference.
        
        Args:
            preprocessed_input: Preprocessed input tensor
            
        Returns:
            Raw model output tensor
        """
        if not self._model_loaded or self.model is None:
            raise RuntimeError("Model not loaded. Cannot perform inference.")
        
        with self._inference_lock:
            try:
                # Move input to device if needed
                if preprocessed_input.device != self.device:
                    preprocessed_input = preprocessed_input.to(self.device)
                
                # Perform inference
                with torch.no_grad():
                    start_time = time.time()
                    output = self.model(preprocessed_input)
                    inference_time = time.time() - start_time
                    
                    logger.debug(f"Inference completed in {inference_time:.3f}s for batch size {preprocessed_input.shape[0]}")
                    
                return output
                
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"GPU out of memory during inference: {e}")
                if self.config.model.fallback_to_cpu and self.device != torch.device('cpu'):
                    logger.info("Attempting CPU fallback for this inference...")
                    # Move to CPU for this inference
                    preprocessed_input = preprocessed_input.cpu()
                    model_device = next(self.model.parameters()).device
                    self.model.cpu()
                    try:
                        with torch.no_grad():
                            output = self.model(preprocessed_input)
                        return output
                    finally:
                        # Move model back to original device
                        self.model.to(model_device)
                else:
                    raise
            except Exception as e:
                logger.error(f"Inference failed: {e}")
                raise
    
    def postprocess(self, model_output: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """
        Postprocess U-Net model output.
        
        Args:
            model_output: Raw model output tensor
            **kwargs: Additional postprocessing parameters
            
        Returns:
            Dictionary containing processed results
        """
        if self.postprocessor is None:
            raise RuntimeError("Postprocessor not initialized. Model loading may have failed.")
        
        return self.postprocessor.postprocess(model_output, **kwargs)
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check.
        
        Returns:
            Dictionary containing health status information
        """
        health_status = {
            'service_name': 'unet',
            'model_variant': self.config.model.model_variant,
            'model_loaded': self._model_loaded,
            'device': str(self.device) if self.device else None,
            'memory_pools_initialized': self.memory_manager is not None,
            'healthy': False,
            'startup_time': None,
            'last_check': time.time()
        }
        
        if self._load_start_time:
            health_status['startup_time'] = time.time() - self._load_start_time
        
        try:
            # Check if model is loaded and functional
            if not self._model_loaded or self.model is None:
                health_status['error'] = 'Model not loaded'
                return health_status
            
            # Check device availability
            if self.device.type == 'cuda' and not torch.cuda.is_available():
                health_status['error'] = 'CUDA device not available'
                return health_status
            
            # Check memory pools
            if self.memory_manager:
                pool_status = self.memory_manager.get_pool_status()
                health_status['memory_pools'] = pool_status
                
            # Perform dummy inference if enabled
            if self.config.health.dummy_inference_test:
                try:
                    dummy_input = torch.randn(1, 3, 64, 64, device=self.device)
                    with torch.no_grad():
                        _ = self.model(dummy_input)
                    health_status['inference_test'] = 'passed'
                except Exception as e:
                    health_status['inference_test'] = f'failed: {str(e)}'
                    health_status['error'] = f'Inference test failed: {str(e)}'
                    return health_status
            
            health_status['healthy'] = True
            
        except Exception as e:
            health_status['error'] = f'Health check failed: {str(e)}'
            logger.error(f"Health check error: {e}")
        
        return health_status
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed model information.
        
        Returns:
            Dictionary containing model metadata
        """
        info = {
            'model_name': 'unet',
            'model_variant': self.config.model.model_variant,
            'model_loaded': self._model_loaded,
            'device': str(self.device) if self.device else None,
            'checkpoint_dir': self.config.model.checkpoint_dir,
            'preprocessing': {
                'image_size': self.config.preprocessing.image_size,
                'normalize': self.config.preprocessing.normalize,
                'mean': self.config.preprocessing.mean,
                'std': self.config.preprocessing.std,
            },
            'inference': {
                'batch_size': self.config.inference.batch_size,
                'threshold': self.config.inference.threshold,
                'enable_confidence': self.config.inference.enable_confidence,
            }
        }
        
        if self.model:
            # Count model parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            info.update({
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            })
        
        # Add best model path if available
        best_model_path = self.config.get_best_model_path()
        if best_model_path:
            info['current_model_path'] = str(best_model_path)
            info['model_file_size_mb'] = best_model_path.stat().st_size / (1024 * 1024)
        
        return info
    
    @contextmanager
    def inference_context(self):
        """Context manager for inference operations with proper resource management."""
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            logger.debug(f"Starting inference context {request_id}")
            yield request_id
        finally:
            duration = time.time() - start_time
            logger.debug(f"Inference context {request_id} completed in {duration:.3f}s")
    
    def process_single_image(self, image: Union[Image.Image, np.ndarray]) -> Dict[str, Any]:
        """
        Process a single image through the complete pipeline.
        
        Args:
            image: Input image as PIL Image or numpy array
            
        Returns:
            Dictionary containing segmentation results
        """
        with self.inference_context() as request_id:
            try:
                # Preprocess
                preprocessed = self.preprocess(image)
                
                # Inference
                output = self.inference(preprocessed)
                
                # Postprocess
                result = self.postprocess(output)
                result['request_id'] = request_id
                
                return result
                
            except Exception as e:
                logger.error(f"Single image processing failed for request {request_id}: {e}")
                raise
    
    def process_batch(self, images: List[Union[Image.Image, np.ndarray]]) -> List[Dict[str, Any]]:
        """
        Process a batch of images.
        
        Args:
            images: List of input images
            
        Returns:
            List of segmentation results
        """
        if not self.config.inference.enable_batch_processing:
            # Process individually
            return [self.process_single_image(img) for img in images]
        
        with self.inference_context() as request_id:
            try:
                # Preprocess batch
                preprocessed = self.preprocess(images)
                
                # Inference
                output = self.inference(preprocessed)
                
                # Postprocess each output
                results = []
                for i in range(output.shape[0]):
                    single_output = output[i:i+1]  # Keep batch dimension
                    result = self.postprocess(single_output)
                    result['request_id'] = f"{request_id}_{i}"
                    results.append(result)
                
                return results
                
            except Exception as e:
                logger.error(f"Batch processing failed for request {request_id}: {e}")
                raise
    
    # Async methods for FastAPI routes
    
    async def segment_async(
        self, 
        image_data: str, 
        threshold: Optional[float] = None,
        return_confidence: bool = False,
        return_contours: bool = False,
        return_visualizations: bool = False
    ) -> Dict[str, Any]:
        """
        Async wrapper for single image segmentation.
        
        Args:
            image_data: Base64 encoded image data
            threshold: Segmentation threshold
            return_confidence: Whether to return confidence scores
            return_contours: Whether to return contours
            return_visualizations: Whether to return visualizations
            
        Returns:
            Segmentation result dictionary
        """
        loop = asyncio.get_event_loop()
        
        # Decode image from base64
        import base64
        import io
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Run inference in thread pool to avoid blocking
        result = await loop.run_in_executor(
            None, 
            self._process_image_with_options,
            image, threshold, return_confidence, return_contours, return_visualizations
        )
        
        return result
    
    async def segment_batch_async(
        self,
        images_data: List[str],
        threshold: Optional[float] = None,
        return_confidence: bool = False,
        return_contours: bool = False,
        return_visualizations: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Async wrapper for batch image segmentation.
        
        Args:
            images_data: List of base64 encoded image data
            threshold: Segmentation threshold
            return_confidence: Whether to return confidence scores
            return_contours: Whether to return contours
            return_visualizations: Whether to return visualizations
            
        Returns:
            List of segmentation results
        """
        loop = asyncio.get_event_loop()
        
        # Decode images from base64
        import base64
        import io
        images = []
        for img_data in images_data:
            image_bytes = base64.b64decode(img_data)
            image = Image.open(io.BytesIO(image_bytes))
            images.append(image)
        
        # Run batch inference in thread pool
        results = await loop.run_in_executor(
            None,
            self._process_batch_with_options,
            images, threshold, return_confidence, return_contours, return_visualizations
        )
        
        return results
    
    async def get_health_status(self) -> Dict[str, Any]:
        """
        Async wrapper for health status check.
        
        Returns:
            Health status dictionary with required fields for API
        """
        loop = asyncio.get_event_loop()
        health = await loop.run_in_executor(None, self.health_check)
        
        # Convert to expected API format
        device_str = health.get("device", "") or ""
        return {
            "service_healthy": health.get("healthy", False),
            "model_loaded": health.get("model_loaded", False),
            "memory_pools_ready": health.get("memory_pools_initialized", False),
            "gpu_available": device_str.startswith("cuda"),
            "last_health_check": datetime.now(),
            "model_info": {
                "model_name": health.get("service_name"),
                "model_variant": health.get("model_variant"),
                "device": health.get("device")
            },
            "memory_usage": health.get("memory_pools", {}),
            "capabilities": ["segmentation", "batch_processing", "confidence_scoring"]
        }
    
    async def get_model_info(self) -> Dict[str, Any]:
        """
        Async wrapper for model information.
        
        Returns:
            Model information dictionary with required fields for API
        """
        loop = asyncio.get_event_loop()
        info = await loop.run_in_executor(None, self.get_model_info)
        
        # Convert to expected API format
        return {
            "model_name": info.get("model_name", "unet"),
            "model_variant": info.get("model_variant", "standard"),
            "checkpoint_path": info.get("current_model_path", ""),
            "model_parameters": info.get("total_parameters", 0),
            "input_shape": [1, 3] + self.config.preprocessing.image_size,
            "output_shape": [1, 1] + self.config.preprocessing.image_size,
            "device": info.get("device", "unknown"),
            "architecture_details": {
                "model_type": "AttentionUNet",
                "base_channels": getattr(self.model, 'base_channels', 64) if self.model else 64,
                "depth": getattr(self.model, 'depth', 4) if self.model else 4,
                "use_attention": True
            },
            "training_metadata": {},
            "preprocessing_config": info.get("preprocessing", {}),
            "supported_formats": ["JPEG", "PNG", "BMP"]
        }
    
    async def warm_up(self) -> None:
        """
        Async wrapper for model warm-up.
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._warm_up_model)
    
    def _process_image_with_options(
        self, 
        image: Image.Image,
        threshold: Optional[float] = None,
        return_confidence: bool = False,
        return_contours: bool = False,
        return_visualizations: bool = False
    ) -> Dict[str, Any]:
        """
        Process single image with specific options.
        
        Args:
            image: PIL Image
            threshold: Segmentation threshold
            return_confidence: Whether to return confidence scores
            return_contours: Whether to return contours
            return_visualizations: Whether to return visualizations
            
        Returns:
            Segmentation result dictionary
        """
        # Set threshold if provided
        original_threshold = self.config.inference.threshold
        if threshold is not None:
            self.config.inference.threshold = threshold
        
        try:
            # Process image
            result = self.process_single_image(image)
            
            # Add model info
            result["model_info"] = {
                "model_name": "unet",
                "model_variant": self.config.model.model_variant,
                "device": str(self.device)
            }
            
            # Convert result to expected format
            formatted_result = {
                "mask_data": result.get("binary_mask", ""),
                "confidence_scores": result.get("confidence_scores") if return_confidence else None,
                "contours": result.get("contours") if return_contours else None,
                "visualizations": result.get("visualizations") if return_visualizations else None,
                "model_info": result["model_info"]
            }
            
            return formatted_result
            
        finally:
            # Restore original threshold
            self.config.inference.threshold = original_threshold
    
    def _process_batch_with_options(
        self,
        images: List[Image.Image],
        threshold: Optional[float] = None,
        return_confidence: bool = False,
        return_contours: bool = False,
        return_visualizations: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Process batch of images with specific options.
        
        Args:
            images: List of PIL Images
            threshold: Segmentation threshold
            return_confidence: Whether to return confidence scores
            return_contours: Whether to return contours
            return_visualizations: Whether to return visualizations
            
        Returns:
            List of segmentation results
        """
        # Set threshold if provided
        original_threshold = self.config.inference.threshold
        if threshold is not None:
            self.config.inference.threshold = threshold
        
        try:
            # Process batch
            results = self.process_batch(images)
            
            # Format results
            formatted_results = []
            for result in results:
                # Add model info
                result["model_info"] = {
                    "model_name": "unet",
                    "model_variant": self.config.model.model_variant,
                    "device": str(self.device)
                }
                
                # Convert to expected format
                formatted_result = {
                    "mask_data": result.get("binary_mask", ""),
                    "confidence_scores": result.get("confidence_scores") if return_confidence else None,
                    "contours": result.get("contours") if return_contours else None,
                    "visualizations": result.get("visualizations") if return_visualizations else None,
                    "model_info": result["model_info"]
                }
                formatted_results.append(formatted_result)
            
            return formatted_results
            
        finally:
            # Restore original threshold
            self.config.inference.threshold = original_threshold 