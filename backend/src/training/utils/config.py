"""
Configuration Management for Training Pipeline

Provides comprehensive configuration management with YAML loading,
environment variable overrides, and hardware-aware settings.
"""

import os
import yaml
import torch
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional
from pathlib import Path


@dataclass
class TrainingConfig:
    """Comprehensive training configuration with hardware-aware defaults"""
    
    # Model configuration
    model_type: str = "attention_unet"
    input_channels: int = 3
    output_channels: int = 1
    base_channels: int = 64
    depth: int = 4
    attention: bool = True
    dropout: float = 0.2
    batch_norm: bool = True
    
    # Training configuration
    batch_size: int = 8
    learning_rate: float = 1e-4
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 15
    lr_scheduler: str = "cosine"
    weight_decay: float = 1e-5
    gradient_clip_val: float = 1.0
    
    # Data configuration  
    image_size: Tuple[int, int] = (1918, 1280)
    augmentation_strength: float = 0.5
    normalize_images: bool = True
    
    # Loss configuration
    loss_type: str = "combined"
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    focal_alpha: float = 1.0
    focal_gamma: float = 2.0
    
    # Hardware configuration
    device: str = "auto"
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = True
    
    # Logging configuration
    log_interval: int = 10
    save_interval: int = 5
    tensorboard: bool = True
    wandb: bool = False
    
    # Paths
    data_dir: str = "./data"
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    
    # Augmentation configuration
    augmentation_config: Dict[str, Any] = field(default_factory=lambda: {
        "rotation": 15,
        "horizontal_flip": 0.5,
        "vertical_flip": 0.2,
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "elastic_transform": True,
        "gaussian_noise": 0.1
    })
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'TrainingConfig':
        """Load configuration from YAML file"""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Flatten nested configuration
        flattened_config = cls._flatten_config(config_dict)
        
        # Create config instance
        return cls(**flattened_config)
    
    @classmethod
    def from_env(cls) -> 'TrainingConfig':
        """Create configuration from environment variables"""
        env_config = {}
        
        # Model configuration
        if os.getenv("MODEL_TYPE"):
            env_config["model_type"] = os.getenv("MODEL_TYPE")
        if os.getenv("BASE_CHANNELS"):
            env_config["base_channels"] = int(os.getenv("BASE_CHANNELS"))
        if os.getenv("MODEL_DEPTH"):
            env_config["depth"] = int(os.getenv("MODEL_DEPTH"))
        
        # Training configuration
        if os.getenv("BATCH_SIZE"):
            env_config["batch_size"] = int(os.getenv("BATCH_SIZE"))
        if os.getenv("LEARNING_RATE"):
            env_config["learning_rate"] = float(os.getenv("LEARNING_RATE"))
        if os.getenv("EPOCHS"):
            env_config["epochs"] = int(os.getenv("EPOCHS"))
        
        # Hardware configuration
        if os.getenv("DEVICE"):
            env_config["device"] = os.getenv("DEVICE")
        if os.getenv("NUM_WORKERS"):
            env_config["num_workers"] = int(os.getenv("NUM_WORKERS"))
        
        # Paths
        if os.getenv("DATA_DIR"):
            env_config["data_dir"] = os.getenv("DATA_DIR")
        if os.getenv("OUTPUT_DIR"):
            env_config["output_dir"] = os.getenv("OUTPUT_DIR")
        
        return cls(**env_config)
    
    @staticmethod
    def _flatten_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested configuration dictionary"""
        flattened = {}
        
        for key, value in config_dict.items():
            if key == "model" and isinstance(value, dict):
                for model_key, model_value in value.items():
                    if model_key == "type":
                        flattened["model_type"] = model_value
                    else:
                        flattened[model_key] = model_value
            elif key == "training" and isinstance(value, dict):
                flattened.update(value)
            elif key == "data" and isinstance(value, dict):
                for data_key, data_value in value.items():
                    if data_key == "image_size" and isinstance(data_value, list) and len(data_value) >= 2:
                        # Use the smaller dimension for square images
                        flattened["image_size"] = data_value
                    else:
                        flattened[data_key] = data_value
            elif key == "loss" and isinstance(value, dict):
                flattened.update(value)
            elif key == "hardware" and isinstance(value, dict):
                flattened.update(value)
            elif key == "logging" and isinstance(value, dict):
                flattened.update(value)
            elif key == "paths" and isinstance(value, dict):
                flattened.update(value)
            elif key == "augmentation" and isinstance(value, dict):
                flattened["augmentation_config"] = value
            # Skip unknown sections like small_dataset_optimizations
            elif key not in ["small_dataset_optimizations"]:
                flattened[key] = value
        
        return flattened
    
    def get_device(self) -> torch.device:
        """Get PyTorch device based on configuration"""
        if self.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(self.device)
        
        return device
    
    def get_num_workers(self) -> int:
        """Get optimal number of workers based on hardware"""
        if self.num_workers == -1:
            return min(8, os.cpu_count() or 1)
        return self.num_workers
    
    def save(self, path: str) -> None:
        """Save configuration to YAML file"""
        config_dict = {
            "model": {
                "type": self.model_type,
                "input_channels": self.input_channels,
                "output_channels": self.output_channels,
                "base_channels": self.base_channels,
                "depth": self.depth,
                "attention": self.attention,
                "dropout": self.dropout,
                "batch_norm": self.batch_norm
            },
            "training": {
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "validation_split": self.validation_split,
                "early_stopping_patience": self.early_stopping_patience,
                "lr_scheduler": self.lr_scheduler,
                "weight_decay": self.weight_decay,
                "gradient_clip_val": self.gradient_clip_val
            },
            "data": {
                "image_size": [self.image_size, self.image_size] if isinstance(self.image_size, int) else list(self.image_size),
                "augmentation_strength": self.augmentation_strength,
                "normalize_images": self.normalize_images
            },
            "loss": {
                "loss_type": self.loss_type,
                "bce_weight": self.bce_weight,
                "dice_weight": self.dice_weight,
                "focal_alpha": self.focal_alpha,
                "focal_gamma": self.focal_gamma
            },
            "hardware": {
                "device": self.device,
                "num_workers": self.num_workers,
                "pin_memory": self.pin_memory,
                "mixed_precision": self.mixed_precision
            },
            "logging": {
                "log_interval": self.log_interval,
                "save_interval": self.save_interval,
                "tensorboard": self.tensorboard,
                "wandb": self.wandb
            },
            "paths": {
                "data_dir": self.data_dir,
                "output_dir": self.output_dir,
                "checkpoint_dir": self.checkpoint_dir
            },
            "augmentation": self.augmentation_config
        }
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def validate(self) -> None:
        """Validate configuration parameters"""
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        if self.epochs <= 0:
            raise ValueError("Number of epochs must be positive")
        
        if not 0 < self.validation_split < 1:
            raise ValueError("Validation split must be between 0 and 1")
        
        if len(self.image_size) != 2 or any(s <= 0 for s in self.image_size):
            raise ValueError("Image size must be a tuple of two positive integers")
        
        if self.depth <= 0:
            raise ValueError("Model depth must be positive")
        
        if self.base_channels <= 0:
            raise ValueError("Base channels must be positive")
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get model-specific parameters"""
        return {
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "base_channels": self.base_channels,
            "depth": self.depth,
            "attention": self.attention,
            "dropout": self.dropout,
            "batch_norm": self.batch_norm
        }
    
    def get_optimizer_params(self) -> Dict[str, Any]:
        """Get optimizer parameters"""
        return {
            "lr": self.learning_rate,
            "weight_decay": self.weight_decay
        }
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return f"TrainingConfig(model={self.model_type}, batch_size={self.batch_size}, lr={self.learning_rate}, device={self.device})" 