"""
Base Model Service Abstract Class

Provides a foundation for implementing model serving services with standardized
patterns for model loading, inference, health checking, and memory management.
"""

import time
import logging
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from pathlib import Path

import torch
import numpy as np

from .memory_manager import MemoryPoolManager


class BaseModelService(ABC):
    """
    Abstract base class for model serving services
    
    Provides common functionality for model management, memory pools,
    health monitoring, and thread-safe inference coordination.
    """
    
    def __init__(self, config: Any, service_name: str):
        """
        Initialize base model service
        
        Args:
            config: Service configuration object
            service_name: Name of the service for logging and identification
        """
        self.config = config
        self.service_name = service_name
        self.logger = logging.getLogger(f"serving.{service_name}")
        
        # Service state
        self._initialized = False
        self._model_loaded = False
        self._memory_pools_ready = False
        
        # Thread safety
        self._inference_lock = threading.RLock()
        self._initialization_lock = threading.RLock()
        
        # Model and memory management
        self.model: Optional[torch.nn.Module] = None
        self.device: torch.device = None
        self.memory_manager: Optional[MemoryPoolManager] = None
        
        # Performance tracking
        self._inference_count = 0
        self._total_inference_time = 0.0
        self._last_inference_time = None
        
        self.logger.info(f"Initializing {service_name} service")
    
    @abstractmethod
    async def load_model(self) -> bool:
        """
        Load the model and prepare for inference
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def preprocess(self, input_data: Any) -> torch.Tensor:
        """
        Preprocess input data for model inference
        
        Args:
            input_data: Raw input data
            
        Returns:
            torch.Tensor: Preprocessed tensor ready for inference
        """
        pass
    
    @abstractmethod
    def inference(self, preprocessed_input: torch.Tensor) -> torch.Tensor:
        """
        Run model inference on preprocessed input
        
        Args:
            preprocessed_input: Preprocessed input tensor
            
        Returns:
            torch.Tensor: Model output tensor
        """
        pass
    
    @abstractmethod
    def postprocess(self, model_output: torch.Tensor, metadata: Dict[str, Any] = None) -> Any:
        """
        Postprocess model output to final result format
        
        Args:
            model_output: Raw model output tensor
            metadata: Additional metadata for postprocessing
            
        Returns:
            Any: Final processed result
        """
        pass
    
    async def initialize_service(self) -> bool:
        """
        Initialize the complete service including model loading and memory pools
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        with self._initialization_lock:
            if self._initialized:
                return True
            
            try:
                self.logger.info(f"Starting {self.service_name} service initialization")
                
                # Initialize device
                self._initialize_device()
                
                # Initialize memory pools
                if await self.initialize_memory_pools():
                    self._memory_pools_ready = True
                    self.logger.info("Memory pools initialized successfully")
                else:
                    self.logger.error("Failed to initialize memory pools")
                    return False
                
                # Load model
                if await self.load_model():
                    self._model_loaded = True
                    self.logger.info("Model loaded successfully")
                else:
                    self.logger.error("Failed to load model")
                    return False
                
                # Perform startup validation
                if await self._startup_validation():
                    self._initialized = True
                    self.logger.info(f"{self.service_name} service initialized successfully")
                    return True
                else:
                    self.logger.error("Startup validation failed")
                    return False
                    
            except Exception as e:
                self.logger.error(f"Service initialization failed: {e}")
                return False
    
    async def initialize_memory_pools(self) -> bool:
        """
        Initialize memory pools for efficient tensor management
        
        Returns:
            bool: True if memory pools initialized successfully
        """
        try:
            # Get memory pool configuration from subclass
            pool_config = self._get_memory_pool_config()
            
            if pool_config:
                self.memory_manager = MemoryPoolManager(
                    device=self.device,
                    config=pool_config,
                    logger=self.logger
                )
                return await self.memory_manager.initialize_pools()
            else:
                self.logger.info("No memory pool configuration provided, skipping pool initialization")
                return True
                
        except Exception as e:
            self.logger.error(f"Memory pool initialization failed: {e}")
            return False
    
    def _get_memory_pool_config(self) -> Optional[Dict[str, Any]]:
        """
        Get memory pool configuration from service config
        Subclasses should override this to provide specific pool configurations
        
        Returns:
            Optional[Dict[str, Any]]: Memory pool configuration or None
        """
        return getattr(self.config, 'memory_pool_config', None)
    
    def _initialize_device(self) -> None:
        """Initialize compute device (GPU/CPU)"""
        device_config = getattr(self.config, 'device', 'auto')
        
        if device_config == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_config)
        
        self.logger.info(f"Using device: {self.device}")
    
    async def _startup_validation(self) -> bool:
        """
        Perform startup validation to ensure service is ready
        
        Returns:
            bool: True if validation passes
        """
        try:
            # Test basic inference capability with dummy data
            dummy_result = await self._test_inference_capability()
            return dummy_result is not None
            
        except Exception as e:
            self.logger.error(f"Startup validation failed: {e}")
            return False
    
    async def _test_inference_capability(self) -> Optional[Any]:
        """
        Test inference capability with dummy data
        Subclasses should override to provide appropriate dummy data
        
        Returns:
            Optional[Any]: Test result or None if test failed
        """
        self.logger.info("Dummy inference test not implemented for this service")
        return True  # Default to passing if not implemented
    
    async def process_request(self, input_data: Any, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a complete inference request with timing and error handling
        
        Args:
            input_data: Input data for inference
            metadata: Additional metadata for processing
            
        Returns:
            Dict[str, Any]: Response with results and metadata
        """
        start_time = time.time()
        
        if not self._initialized:
            return {
                "success": False,
                "error": {
                    "code": "SERVICE_NOT_INITIALIZED",
                    "message": "Service is not initialized",
                    "details": f"{self.service_name} service must be initialized before processing requests"
                }
            }
        
        try:
            with self._inference_lock:
                # Preprocess input
                preprocessed_input = self.preprocess(input_data)
                
                # Run inference
                model_output = self.inference(preprocessed_input)
                
                # Postprocess output
                final_result = self.postprocess(model_output, metadata or {})
                
                # Update performance metrics
                inference_time = time.time() - start_time
                self._update_performance_metrics(inference_time)
                
                return {
                    "success": True,
                    "result": final_result,
                    "metadata": {
                        "processing_time": inference_time,
                        "inference_count": self._inference_count,
                        "service_name": self.service_name
                    }
                }
                
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Request processing failed: {e}")
            
            return {
                "success": False,
                "error": {
                    "code": "INFERENCE_FAILED",
                    "message": "Inference processing failed",
                    "details": str(e)
                },
                "metadata": {
                    "processing_time": processing_time,
                    "service_name": self.service_name
                }
            }
    
    def _update_performance_metrics(self, inference_time: float) -> None:
        """Update internal performance metrics"""
        self._inference_count += 1
        self._total_inference_time += inference_time
        self._last_inference_time = inference_time
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check service health status
        
        Returns:
            Dict[str, Any]: Health status information
        """
        health_info = {
            "service_name": self.service_name,
            "status": "healthy" if self._initialized else "unhealthy",
            "initialized": self._initialized,
            "model_loaded": self._model_loaded,
            "memory_pools_ready": self._memory_pools_ready,
            "device": str(self.device) if self.device else None,
            "inference_count": self._inference_count,
            "average_inference_time": (
                self._total_inference_time / self._inference_count 
                if self._inference_count > 0 else None
            ),
            "last_inference_time": self._last_inference_time
        }
        
        # Add GPU memory info if available
        if self.device and self.device.type == 'cuda':
            try:
                health_info.update({
                    "gpu_memory_allocated": torch.cuda.memory_allocated(self.device),
                    "gpu_memory_reserved": torch.cuda.memory_reserved(self.device),
                    "gpu_memory_available": True
                })
            except Exception:
                health_info["gpu_memory_available"] = False
        
        # Add memory pool status if available
        if self.memory_manager:
            health_info["memory_pools"] = self.memory_manager.get_pool_status()
        
        return health_info
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and capabilities
        
        Returns:
            Dict[str, Any]: Model information
        """
        info = {
            "service_name": self.service_name,
            "model_loaded": self._model_loaded,
            "device": str(self.device) if self.device else None,
            "initialized": self._initialized
        }
        
        # Add model-specific info if model is loaded
        if self.model and self._model_loaded:
            try:
                # Get parameter count
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                
                info.update({
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params,
                    "model_size_mb": total_params * 4 / (1024 * 1024)  # Assuming float32
                })
                
                # Add model-specific info from subclass
                model_specific_info = self._get_model_specific_info()
                if model_specific_info:
                    info.update(model_specific_info)
                    
            except Exception as e:
                self.logger.warning(f"Could not get model information: {e}")
        
        return info
    
    def _get_model_specific_info(self) -> Optional[Dict[str, Any]]:
        """
        Get model-specific information
        Subclasses should override to provide specific model details
        
        Returns:
            Optional[Dict[str, Any]]: Model-specific information or None
        """
        return None
    
    def shutdown(self) -> None:
        """Shutdown the service and cleanup resources"""
        self.logger.info(f"Shutting down {self.service_name} service")
        
        with self._initialization_lock:
            if self.memory_manager:
                self.memory_manager.cleanup()
            
            if self.model and self.device and self.device.type == 'cuda':
                # Clear GPU memory
                try:
                    del self.model
                    torch.cuda.empty_cache()
                except Exception as e:
                    self.logger.warning(f"Error during GPU cleanup: {e}")
            
            self._initialized = False
            self._model_loaded = False
            self._memory_pools_ready = False
        
        self.logger.info(f"{self.service_name} service shutdown complete") 