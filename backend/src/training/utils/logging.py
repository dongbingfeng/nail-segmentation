"""
Logging Utilities for Training Pipeline

Provides structured logging, TensorBoard integration, and training monitoring.
"""

import os
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from torch.utils.tensorboard import SummaryWriter


def setup_logging(
    log_dir: str = "./logs",
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True
) -> logging.Logger:
    """
    Set up structured logging for training pipeline
    
    Args:
        log_dir: Directory to save log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
    
    Returns:
        Configured logger instance
    """
    # Create log directory
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("nail_segmentation_training")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler
    if log_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"training_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "nail_segmentation_training") -> logging.Logger:
    """Get existing logger instance"""
    return logging.getLogger(name)


class TrainingLogger:
    """
    Comprehensive training logger with TensorBoard integration
    """
    
    def __init__(
        self,
        log_dir: str = "./logs",
        tensorboard_dir: str = "./runs",
        experiment_name: Optional[str] = None
    ):
        """
        Initialize training logger
        
        Args:
            log_dir: Directory for text logs
            tensorboard_dir: Directory for TensorBoard logs
            experiment_name: Name of the experiment
        """
        self.log_dir = Path(log_dir)
        self.tensorboard_dir = Path(tensorboard_dir)
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"nail_segmentation_{timestamp}"
        
        self.experiment_name = experiment_name
        
        # Set up text logger
        self.logger = setup_logging(
            log_dir=str(self.log_dir),
            log_level="INFO",
            log_to_file=True,
            log_to_console=True
        )
        
        # Set up TensorBoard writer
        tensorboard_path = self.tensorboard_dir / experiment_name
        self.writer = SummaryWriter(log_dir=str(tensorboard_path))
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        
        self.logger.info(f"Training logger initialized for experiment: {experiment_name}")
        self.logger.info(f"TensorBoard logs: {tensorboard_path}")
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log training configuration"""
        self.logger.info("Training Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
        
        # Log config to TensorBoard as text
        config_text = "\n".join([f"{k}: {v}" for k, v in config.items()])
        self.writer.add_text("config", config_text, 0)
    
    def log_model_info(self, model_info: Dict[str, Any]) -> None:
        """Log model architecture information"""
        self.logger.info("Model Information:")
        for key, value in model_info.items():
            self.logger.info(f"  {key}: {value}")
        
        # Log to TensorBoard
        if "total_params" in model_info:
            self.writer.add_scalar("model/total_params", model_info["total_params"], 0)
        if "trainable_params" in model_info:
            self.writer.add_scalar("model/trainable_params", model_info["trainable_params"], 0)
    
    def log_epoch_start(self, epoch: int) -> None:
        """Log start of training epoch"""
        self.current_epoch = epoch
        self.logger.info(f"Starting epoch {epoch}")
    
    def log_epoch_end(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
        """Log end of training epoch with metrics"""
        self.logger.info(f"Epoch {epoch} completed:")
        
        # Log training metrics
        self.logger.info("Training metrics:")
        for metric, value in train_metrics.items():
            self.logger.info(f"  train_{metric}: {value:.6f}")
            self.writer.add_scalar(f"train/{metric}", value, epoch)
        
        # Log validation metrics
        self.logger.info("Validation metrics:")
        for metric, value in val_metrics.items():
            self.logger.info(f"  val_{metric}: {value:.6f}")
            self.writer.add_scalar(f"val/{metric}", value, epoch)
        
        self.logger.info("-" * 50)
    
    def log_batch_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log batch-level metrics"""
        self.global_step = step
        
        # Log to TensorBoard
        for metric, value in metrics.items():
            self.writer.add_scalar(f"batch/{metric}", value, step)
    
    def log_learning_rate(self, lr: float, epoch: int) -> None:
        """Log current learning rate"""
        self.logger.info(f"Learning rate: {lr:.8f}")
        self.writer.add_scalar("train/learning_rate", lr, epoch)
    
    def log_checkpoint_save(self, checkpoint_path: str, epoch: int, metrics: Dict[str, float]) -> None:
        """Log checkpoint saving"""
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        self.logger.info(f"Epoch: {epoch}, Metrics: {metrics}")
    
    def log_early_stopping(self, epoch: int, patience: int, best_metric: float) -> None:
        """Log early stopping trigger"""
        self.logger.warning(f"Early stopping triggered at epoch {epoch}")
        self.logger.warning(f"No improvement for {patience} epochs")
        self.logger.warning(f"Best metric: {best_metric:.6f}")
    
    def log_training_complete(self, total_epochs: int, best_metrics: Dict[str, float]) -> None:
        """Log training completion"""
        self.logger.info("Training completed!")
        self.logger.info(f"Total epochs: {total_epochs}")
        self.logger.info("Best metrics:")
        for metric, value in best_metrics.items():
            self.logger.info(f"  {metric}: {value:.6f}")
    
    def log_error(self, error: Exception, context: str = "") -> None:
        """Log training errors"""
        if context:
            self.logger.error(f"Error in {context}: {str(error)}")
        else:
            self.logger.error(f"Training error: {str(error)}")
        
        # Log error to TensorBoard
        self.writer.add_text("errors", f"{context}: {str(error)}", self.current_epoch)
    
    def log_gpu_memory(self, allocated: float, cached: float, epoch: int) -> None:
        """Log GPU memory usage"""
        self.logger.info(f"GPU Memory - Allocated: {allocated:.2f}MB, Cached: {cached:.2f}MB")
        self.writer.add_scalar("system/gpu_memory_allocated", allocated, epoch)
        self.writer.add_scalar("system/gpu_memory_cached", cached, epoch)
    
    def log_training_time(self, epoch_time: float, total_time: float, epoch: int) -> None:
        """Log training time metrics"""
        self.logger.info(f"Epoch time: {epoch_time:.2f}s, Total time: {total_time:.2f}s")
        self.writer.add_scalar("time/epoch_time", epoch_time, epoch)
        self.writer.add_scalar("time/total_time", total_time, epoch)
    
    def add_image(self, tag: str, image, step: int) -> None:
        """Add image to TensorBoard"""
        self.writer.add_image(tag, image, step)
    
    def add_histogram(self, tag: str, values, step: int) -> None:
        """Add histogram to TensorBoard"""
        self.writer.add_histogram(tag, values, step)
    
    def close(self) -> None:
        """Close logger and TensorBoard writer"""
        self.logger.info("Closing training logger")
        self.writer.close()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


class MetricsTracker:
    """
    Track and compute running averages of training metrics
    """
    
    def __init__(self):
        self.metrics = {}
        self.counts = {}
    
    def update(self, metrics: Dict[str, float]) -> None:
        """Update metrics with new values"""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0
            
            self.metrics[key] += value
            self.counts[key] += 1
    
    def get_averages(self) -> Dict[str, float]:
        """Get average values for all metrics"""
        averages = {}
        for key in self.metrics:
            if self.counts[key] > 0:
                averages[key] = self.metrics[key] / self.counts[key]
            else:
                averages[key] = 0.0
        return averages
    
    def reset(self) -> None:
        """Reset all metrics"""
        self.metrics.clear()
        self.counts.clear()
    
    def get_current_values(self) -> Dict[str, float]:
        """Get current accumulated values"""
        return self.metrics.copy() 