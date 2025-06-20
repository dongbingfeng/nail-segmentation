#!/usr/bin/env python3
"""
Nail Segmentation Training Script

Main entry point for training nail segmentation models with configuration
management, logging, and comprehensive error handling.

Usage:
    python train.py --config config.yaml --data-dir /path/to/data
    python train.py --config config.yaml --data-dir /path/to/data --resume checkpoint.pth
"""

import argparse
import sys
import os
import traceback
import torch
from pathlib import Path

# Add parent directories to path for imports
training_root = str(Path(__file__).parent.parent)
sys.path.insert(0, training_root)

# Import directly from the modules
from training.trainer import NailSegmentationTrainer
from utils.config import TrainingConfig
from utils.logging import setup_logging, get_logger
from utils.checkpoint import CheckpointManager


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train nail segmentation model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to training data directory"
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    
    # Training control
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)"
    )
    
    # Hardware configuration
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for training"
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of data loader workers (overrides config)"
    )
    
    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for models and logs (overrides config)"
    )
    
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Checkpoint directory (overrides config)"
    )
    
    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    parser.add_argument(
        "--disable-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging"
    )
    
    # Model configuration
    parser.add_argument(
        "--model-variant",
        type=str,
        default="standard",
        choices=["standard", "lightweight", "deep"],
        help="Model variant to use"
    )
    
    # Validation
    parser.add_argument(
        "--validation-split",
        type=float,
        default=None,
        help="Validation split ratio (overrides config)"
    )
    
    # Performance
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="Disable automatic mixed precision"
    )
    
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling for performance analysis"
    )
    
    # Debugging
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with additional logging"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run setup without actual training"
    )
    
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command line arguments"""
    # Check data directory
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    
    # Check config file
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
    
    # Check resume checkpoint
    if args.resume and not os.path.exists(args.resume):
        raise FileNotFoundError(f"Checkpoint file not found: {args.resume}")
    
    # Validate numeric arguments
    if args.epochs is not None and args.epochs <= 0:
        raise ValueError("Number of epochs must be positive")
    
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("Batch size must be positive")
    
    if args.learning_rate is not None and args.learning_rate <= 0:
        raise ValueError("Learning rate must be positive")
    
    if args.validation_split is not None and not (0 < args.validation_split < 1):
        raise ValueError("Validation split must be between 0 and 1")


def setup_configuration(args: argparse.Namespace) -> TrainingConfig:
    """Setup training configuration from file and arguments"""
    # Load base configuration
    config = TrainingConfig.from_yaml(args.config)
    
    # Override with command line arguments
    if args.epochs is not None:
        config.epochs = args.epochs
    
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    
    if args.device != "auto":
        config.device = args.device
    
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir
    
    if args.validation_split is not None:
        config.validation_split = args.validation_split
    
    if args.disable_tensorboard:
        config.tensorboard = False
    
    if args.disable_amp:
        config.mixed_precision = False
    
    # Validate configuration
    config.validate()
    
    return config


def setup_script_logging(args: argparse.Namespace) -> None:
    """Setup logging configuration"""
    import logging
    
    # Set logging level
    level = getattr(logging, args.log_level.upper())
    
    # Setup training logger
    logger = setup_logging(log_level=args.log_level.upper())
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")


def print_system_info(logger) -> None:
    """Print system information for debugging"""
    import platform
    import psutil
    
    logger.info("=== System Information ===")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # MPS availability (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.info("MPS (Metal Performance Shaders) available")
    
    # Memory information
    memory = psutil.virtual_memory()
    logger.info(f"Total RAM: {memory.total / 1e9:.1f} GB")
    logger.info(f"Available RAM: {memory.available / 1e9:.1f} GB")
    
    # CPU information
    logger.info(f"CPU cores: {psutil.cpu_count()}")
    logger.info("=" * 30)


def run_training(args: argparse.Namespace) -> None:
    """Run the training process"""
    logger = get_logger("train_script")
    
    try:
        # Setup configuration
        logger.info("Setting up configuration...")
        config = setup_configuration(args)
        logger.info(f"Configuration loaded: {config}")
        
        # Print system information
        print_system_info(logger)
        
        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = NailSegmentationTrainer(config)
        
        if args.dry_run:
            logger.info("Dry run completed successfully")
            return
        
        # Start training
        if args.resume:
            logger.info(f"Resuming training from: {args.resume}")
            results = trainer.resume_training(args.resume, args.data_dir)
        else:
            logger.info("Starting new training...")
            results = trainer.train(args.data_dir)
        
        # Print results
        logger.info("=== Training Results ===")
        logger.info(f"Best validation IoU: {results['best_val_iou']:.4f}")
        logger.info(f"Best validation loss: {results['best_val_loss']:.4f}")
        logger.info(f"Total epochs: {results['total_epochs']}")
        logger.info(f"Total time: {results['total_time']:.2f} seconds")
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        if args.debug:
            logger.error("Full traceback:")
            logger.error(traceback.format_exc())
        sys.exit(1)


def main():
    """Main entry point"""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_script_logging(args)
    logger = get_logger("train_script")
    
    try:
        # Validate arguments
        validate_arguments(args)
        
        # Check if profiling is enabled
        if args.profile:
            import cProfile
            import pstats
            
            logger.info("Running with profiling enabled...")
            profiler = cProfile.Profile()
            profiler.enable()
            
            run_training(args)
            
            profiler.disable()
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            stats.print_stats(20)  # Print top 20 functions
        else:
            run_training(args)
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        if args.debug:
            logger.error("Full traceback:")
            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main() 