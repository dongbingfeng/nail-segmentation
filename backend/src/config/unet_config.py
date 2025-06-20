"""
U-Net Model Serving Configuration

This module provides configuration management for the U-Net model serving system,
including model checkpoint discovery, preprocessing parameters, memory management,
and environment variable support.
"""

import os
import yaml
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model loading and management configuration."""
    checkpoint_dir: str = "../../models/unet/checkpoints/"
    model_variant: str = "standard"  # standard, lightweight, deep
    device: str = "auto"  # auto, cuda, cpu
    model_name: Optional[str] = None  # Specific model file, if None auto-discover best
    fallback_to_cpu: bool = True
    
    def __post_init__(self):
        # Convert relative path to absolute
        if not os.path.isabs(self.checkpoint_dir):
            # Relative to config file location
            config_dir = Path(__file__).parent
            self.checkpoint_dir = str((config_dir / self.checkpoint_dir).resolve())


@dataclass
class PreprocessingConfig:
    """Image preprocessing configuration matching training pipeline."""
    image_size: List[int] = field(default_factory=lambda: [256, 256])
    normalize: bool = True
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    dtype: str = "float32"
    
    def __post_init__(self):
        if len(self.image_size) != 2:
            raise ValueError("image_size must be [height, width]")
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError("mean and std must have 3 values for RGB channels")


@dataclass
class InferenceConfig:
    """Inference execution configuration."""
    batch_size: int = 8
    threshold: float = 0.5
    enable_confidence: bool = True
    max_concurrent_requests: int = 10
    inference_timeout_seconds: int = 30
    enable_batch_processing: bool = True
    
    def __post_init__(self):
        if self.threshold < 0.0 or self.threshold > 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")


@dataclass 
class MemoryConfig:
    """Memory pool management configuration."""
    pool_size_multiplier: int = 10
    enable_gpu_pools: bool = True
    max_pool_size_gb: float = 4.0
    cleanup_threshold: float = 0.8  # Cleanup when pools reach 80% capacity
    preallocate_on_startup: bool = True
    
    def __post_init__(self):
        if self.pool_size_multiplier < 1:
            raise ValueError("pool_size_multiplier must be >= 1")
        if self.max_pool_size_gb <= 0:
            raise ValueError("max_pool_size_gb must be > 0")


@dataclass
class HealthConfig:
    """Health check and monitoring configuration."""
    startup_validation: bool = True
    dummy_inference_test: bool = True
    health_check_interval: int = 30
    max_startup_time_seconds: int = 60
    enable_detailed_metrics: bool = False
    
    def __post_init__(self):
        if self.health_check_interval < 5:
            raise ValueError("health_check_interval must be >= 5 seconds")


class UNetConfig:
    """
    Complete U-Net serving configuration with automatic checkpoint discovery,
    environment variable support, and validation.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration from YAML file and environment variables.
        
        Args:
            config_path: Path to YAML config file. If None, uses default config.
        """
        self.model = ModelConfig()
        self.preprocessing = PreprocessingConfig()
        self.inference = InferenceConfig()
        self.memory = MemoryConfig()
        self.health = HealthConfig()
        
        # Load from YAML if provided
        if config_path:
            self.load_from_yaml(config_path)
        
        # Override with environment variables
        self.load_from_environment()
        
        # Validate configuration
        self.validate()
    
    def load_from_yaml(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        try:
            config_path = Path(config_path)
            if not config_path.exists():
                logger.warning(f"Config file not found: {config_path}")
                return
            
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            if not config_data:
                logger.warning(f"Empty config file: {config_path}")
                return
                
            unet_config = config_data.get('unet', {})
            
            # Update configurations from YAML
            if 'model' in unet_config:
                self._update_dataclass(self.model, unet_config['model'])
            
            if 'preprocessing' in unet_config:
                self._update_dataclass(self.preprocessing, unet_config['preprocessing'])
                
            if 'inference' in unet_config:
                self._update_dataclass(self.inference, unet_config['inference'])
                
            if 'memory' in unet_config:
                self._update_dataclass(self.memory, unet_config['memory'])
                
            if 'health' in unet_config:
                self._update_dataclass(self.health, unet_config['health'])
                
            logger.info(f"Configuration loaded from: {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise
    
    def load_from_environment(self) -> None:
        """Load configuration from environment variables with UNET_ prefix."""
        env_mappings = {
            # Model config
            'UNET_CHECKPOINT_DIR': ('model', 'checkpoint_dir'),
            'UNET_MODEL_VARIANT': ('model', 'model_variant'),
            'UNET_DEVICE': ('model', 'device'),
            'UNET_MODEL_NAME': ('model', 'model_name'),
            'UNET_FALLBACK_TO_CPU': ('model', 'fallback_to_cpu'),
            
            # Preprocessing config
            'UNET_IMAGE_SIZE': ('preprocessing', 'image_size'),
            'UNET_NORMALIZE': ('preprocessing', 'normalize'),
            'UNET_MEAN': ('preprocessing', 'mean'),
            'UNET_STD': ('preprocessing', 'std'),
            
            # Inference config
            'UNET_BATCH_SIZE': ('inference', 'batch_size'),
            'UNET_THRESHOLD': ('inference', 'threshold'),
            'UNET_ENABLE_CONFIDENCE': ('inference', 'enable_confidence'),
            'UNET_MAX_CONCURRENT_REQUESTS': ('inference', 'max_concurrent_requests'),
            'UNET_INFERENCE_TIMEOUT': ('inference', 'inference_timeout_seconds'),
            
            # Memory config
            'UNET_POOL_SIZE_MULTIPLIER': ('memory', 'pool_size_multiplier'),
            'UNET_ENABLE_GPU_POOLS': ('memory', 'enable_gpu_pools'),
            'UNET_MAX_POOL_SIZE_GB': ('memory', 'max_pool_size_gb'),
            
            # Health config
            'UNET_STARTUP_VALIDATION': ('health', 'startup_validation'),
            'UNET_DUMMY_INFERENCE_TEST': ('health', 'dummy_inference_test'),
            'UNET_HEALTH_CHECK_INTERVAL': ('health', 'health_check_interval'),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                config_obj = getattr(self, section)
                try:
                    # Type conversion based on current value type
                    current_value = getattr(config_obj, key)
                    if isinstance(current_value, bool):
                        converted_value = value.lower() in ('true', '1', 'yes', 'on')
                    elif isinstance(current_value, int):
                        converted_value = int(value)
                    elif isinstance(current_value, float):
                        converted_value = float(value)
                    elif isinstance(current_value, list):
                        # Handle list parsing for image_size, mean, std
                        if ',' in value:
                            if key in ['mean', 'std']:
                                converted_value = [float(x.strip()) for x in value.split(',')]
                            elif key == 'image_size':
                                converted_value = [int(x.strip()) for x in value.split(',')]
                            else:
                                converted_value = [x.strip() for x in value.split(',')]
                        else:
                            converted_value = [value]
                    else:
                        converted_value = value
                    
                    setattr(config_obj, key, converted_value)
                    logger.info(f"Environment override: {env_var} = {converted_value}")
                    
                except (ValueError, TypeError) as e:
                    logger.error(f"Failed to convert environment variable {env_var}={value}: {e}")
                    raise
    
    def validate(self) -> None:
        """Validate complete configuration and dependencies."""
        try:
            # Validate checkpoint directory exists
            checkpoint_path = Path(self.model.checkpoint_dir)
            if not checkpoint_path.exists():
                logger.warning(f"Checkpoint directory does not exist: {checkpoint_path}")
                # Create directory if possible
                try:
                    checkpoint_path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created checkpoint directory: {checkpoint_path}")
                except Exception as e:
                    logger.error(f"Cannot create checkpoint directory: {e}")
            
            # Validate model variant
            valid_variants = ['standard', 'lightweight', 'deep']
            if self.model.model_variant not in valid_variants:
                raise ValueError(f"model_variant must be one of {valid_variants}")
            
            # Validate device
            valid_devices = ['auto', 'cuda', 'cpu']
            if self.model.device not in valid_devices:
                raise ValueError(f"device must be one of {valid_devices}")
            
            # Validate image size is power of 2 for efficient processing
            h, w = self.preprocessing.image_size
            if h != w:
                logger.warning(f"Non-square image size {h}x{w} may affect model performance")
            
            # Validate memory configuration doesn't exceed reasonable limits
            if self.memory.max_pool_size_gb > 16:
                logger.warning(f"Large memory pool size: {self.memory.max_pool_size_gb}GB")
            
            # Validate inference configuration
            if self.inference.batch_size > 32:
                logger.warning(f"Large batch size {self.inference.batch_size} may cause memory issues")
            
            logger.info("Configuration validation completed successfully")
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    def discover_model_checkpoints(self) -> List[Path]:
        """
        Discover available model checkpoints in the configured directory.
        
        Returns:
            List of checkpoint file paths sorted by modification time (newest first)
        """
        checkpoint_dir = Path(self.model.checkpoint_dir)
        if not checkpoint_dir.exists():
            logger.warning(f"Checkpoint directory not found: {checkpoint_dir}")
            return []
        
        # Look for .pth, .pt, .ckpt files
        patterns = ['*.pth', '*.pt', '*.ckpt']
        checkpoints = []
        
        for pattern in patterns:
            checkpoints.extend(checkpoint_dir.glob(pattern))
        
        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        logger.info(f"Found {len(checkpoints)} checkpoints in {checkpoint_dir}")
        return checkpoints
    
    def get_best_model_path(self) -> Optional[Path]:
        """
        Get the best available model checkpoint path.
        
        Returns:
            Path to best model checkpoint or None if no models found
        """
        if self.model.model_name:
            # Use specific model if specified
            specific_path = Path(self.model.checkpoint_dir) / self.model.model_name
            if specific_path.exists():
                return specific_path
            else:
                logger.error(f"Specified model not found: {specific_path}")
                return None
        
        # Auto-discover best model
        checkpoints = self.discover_model_checkpoints()
        if not checkpoints:
            logger.error("No model checkpoints found")
            return None
        
        # Look for best model indicators
        for checkpoint in checkpoints:
            name_lower = checkpoint.name.lower()
            if 'best' in name_lower or 'final' in name_lower:
                logger.info(f"Selected best model: {checkpoint}")
                return checkpoint
        
        # Fallback to newest model
        best_checkpoint = checkpoints[0]
        logger.info(f"Selected newest model: {best_checkpoint}")
        return best_checkpoint
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            'model': self.model.__dict__,
            'preprocessing': self.preprocessing.__dict__,
            'inference': self.inference.__dict__,
            'memory': self.memory.__dict__,
            'health': self.health.__dict__,
        }
    
    def _update_dataclass(self, dataclass_obj: object, updates: Dict[str, Any]) -> None:
        """Update dataclass fields from dictionary."""
        for key, value in updates.items():
            if hasattr(dataclass_obj, key):
                setattr(dataclass_obj, key, value)
            else:
                logger.warning(f"Unknown configuration key: {key}")


# Global configuration instance
_config_instance: Optional[UNetConfig] = None


def get_unet_config(config_path: Optional[str] = None, force_reload: bool = False) -> UNetConfig:
    """
    Get global U-Net configuration instance (singleton pattern).
    
    Args:
        config_path: Path to YAML config file for initial load
        force_reload: Force reload configuration even if already loaded
    
    Returns:
        UNetConfig instance
    """
    global _config_instance
    
    if _config_instance is None or force_reload:
        _config_instance = UNetConfig(config_path)
    
    return _config_instance


def create_default_config_file(output_path: str) -> None:
    """
    Create a default configuration file with all options documented.
    
    Args:
        output_path: Path where to save the default config YAML
    """
    default_config = {
        'unet': {
            'model': {
                'checkpoint_dir': '../../models/unet/checkpoints/',
                'model_variant': 'standard',  # standard, lightweight, deep
                'device': 'auto',  # auto, cuda, cpu
                'model_name': None,  # null for auto-discovery
                'fallback_to_cpu': True,
            },
            'preprocessing': {
                'image_size': [256, 256],
                'normalize': True,
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'dtype': 'float32',
            },
            'inference': {
                'batch_size': 8,
                'threshold': 0.5,
                'enable_confidence': True,
                'max_concurrent_requests': 10,
                'inference_timeout_seconds': 30,
                'enable_batch_processing': True,
            },
            'memory': {
                'pool_size_multiplier': 10,
                'enable_gpu_pools': True,
                'max_pool_size_gb': 4.0,
                'cleanup_threshold': 0.8,
                'preallocate_on_startup': True,
            },
            'health': {
                'startup_validation': True,
                'dummy_inference_test': True,
                'health_check_interval': 30,
                'max_startup_time_seconds': 60,
                'enable_detailed_metrics': False,
            }
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)
    
    logger.info(f"Default configuration saved to: {output_path}")


if __name__ == "__main__":
    # Example usage and testing
    print("Testing U-Net Configuration System")
    
    # Create default config
    create_default_config_file("unet_config_default.yaml")
    
    # Test configuration loading
    config = UNetConfig()
    print(f"Default config loaded:")
    print(f"  Checkpoint dir: {config.model.checkpoint_dir}")
    print(f"  Image size: {config.preprocessing.image_size}")
    print(f"  Batch size: {config.inference.batch_size}")
    
    # Test checkpoint discovery
    checkpoints = config.discover_model_checkpoints()
    print(f"Found checkpoints: {[str(p) for p in checkpoints]}")
    
    best_model = config.get_best_model_path()
    print(f"Best model: {best_model}") 