"""
Nail Segmentation Training Loop

Implements a comprehensive training system for nail segmentation models with
validation, early stopping, checkpointing, and metric tracking.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional, Tuple, List
import logging
from pathlib import Path
import json
from tqdm import tqdm

import sys
from pathlib import Path

# Add the training root to path for imports
training_root = Path(__file__).parent.parent
sys.path.insert(0, str(training_root))

from models.unet import AttentionUNet, create_model
from data.dataset import NailSegmentationDataset
from data.transforms import get_train_transforms, get_val_transforms
from losses.segmentation_losses import CombinedLoss
from metrics.segmentation_metrics import SegmentationMetrics
from utils.config import TrainingConfig
from utils.checkpoint import CheckpointManager
from utils.logging import setup_logging


class NailSegmentationTrainer:
    """
    Comprehensive training system for nail segmentation models
    
    Features:
    - Automatic train/validation splitting
    - Early stopping with patience
    - Learning rate scheduling
    - Gradient clipping
    - Mixed precision training
    - Comprehensive logging
    - Automatic checkpointing
    """
    
    def __init__(self, config: TrainingConfig, output_dir: str = None):
        """
        Initialize trainer with configuration
        
        Args:
            config: Training configuration
            output_dir: Output directory (overrides config if provided)
        """
        self.config = config
        
        # Override output directory if provided
        if output_dir:
            self.config.output_dir = output_dir
            self.config.checkpoint_dir = os.path.join(output_dir, 'checkpoints')
        
        self.device = config.get_device()
        
        # Setup logging
        self.logger = logging.getLogger("trainer")
        self.logger.info(f"Initializing trainer on device: {self.device}")
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_iou = 0.0
        self.early_stopping_counter = 0
        self.training_history = {
            'train_loss': [], 'val_loss': [], 'train_iou': [], 'val_iou': [],
            'train_dice': [], 'val_dice': [], 'learning_rate': []
        }
        
        # Setup directories
        self._setup_directories()
        
        # Setup tensorboard if enabled
        if config.tensorboard:
            self.writer = SummaryWriter(log_dir=os.path.join(config.output_dir, 'tensorboard'))
        else:
            self.writer = None
        
        # Setup checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.checkpoint_dir,
            max_checkpoints=5
        )
        
        # Setup metrics tracker
        self.metrics = SegmentationMetrics()
        
        # Setup mixed precision if enabled
        if config.mixed_precision and self.device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
            self.use_amp = True
        else:
            self.scaler = None
            self.use_amp = False
    
    def _setup_directories(self) -> None:
        """Create necessary directories"""
        directories = [
            self.config.output_dir,
            self.config.checkpoint_dir,
            os.path.join(self.config.output_dir, 'tensorboard'),
            os.path.join(self.config.output_dir, 'predictions')
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def setup_model(self) -> None:
        """Initialize model, optimizer, and loss function"""
        try:
            self.logger.info("Setting up model...")
            
            # Create model
            try:
                self.model = create_model(self.config, model_variant="standard")
                self.model = self.model.to(self.device)
            except Exception as e:
                self.logger.error(f"Failed to create or move model to device: {e}")
                raise RuntimeError(f"Model creation failed: {e}") from e
            
            # Log model info
            try:
                model_info = self.model.get_model_info()
                self.logger.info(f"Model created: {model_info}")
            except Exception as e:
                self.logger.warning(f"Could not get model info: {e}")
            
            # Setup loss function
            try:
                self.criterion = CombinedLoss(
                    bce_weight=self.config.bce_weight,
                    dice_weight=self.config.dice_weight,
                    focal_alpha=self.config.focal_alpha,
                    focal_gamma=self.config.focal_gamma
                ).to(self.device)
            except Exception as e:
                self.logger.error(f"Failed to create loss function: {e}")
                raise RuntimeError(f"Loss function creation failed: {e}") from e
            
            # Setup optimizer
            try:
                self.optimizer = optim.Adam(
                    self.model.parameters(),
                    **self.config.get_optimizer_params()
                )
            except Exception as e:
                self.logger.error(f"Failed to create optimizer: {e}")
                raise RuntimeError(f"Optimizer creation failed: {e}") from e
            
            # Setup learning rate scheduler
            try:
                if self.config.lr_scheduler == "cosine":
                    self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        self.optimizer, T_max=self.config.epochs
                    )
                elif self.config.lr_scheduler == "step":
                    self.scheduler = optim.lr_scheduler.StepLR(
                        self.optimizer, step_size=30, gamma=0.1
                    )
                elif self.config.lr_scheduler == "plateau":
                    self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        self.optimizer, mode='min', patience=10, factor=0.5
                    )
                else:
                    self.scheduler = None
                    
                if self.scheduler:
                    self.logger.info(f"Learning rate scheduler: {self.config.lr_scheduler}")
            except Exception as e:
                self.logger.error(f"Failed to create learning rate scheduler: {e}")
                raise RuntimeError(f"Scheduler creation failed: {e}") from e
                
        except Exception as e:
            self.logger.error(f"Model setup failed: {e}")
            raise
    
    def setup_data(self, data_dir: str) -> Tuple[DataLoader, DataLoader]:
        """
        Setup training and validation data loaders
        
        Args:
            data_dir: Path to data directory
            
        Returns:
            Tuple of (train_loader, val_loader)
            
        Raises:
            RuntimeError: If data setup fails
            ValueError: If data directory is invalid or empty
        """
        try:
            self.logger.info(f"Setting up data loaders from: {data_dir}")
            
            # Validate data directory
            if not os.path.exists(data_dir):
                raise ValueError(f"Data directory does not exist: {data_dir}")
            
            # Create dataset with training transforms
            try:
                # Look for annotations file in data directory
                #annotations_file = os.path.join(data_dir, "annotations.json")
                #if not os.path.exists(annotations_file):
                #    raise FileNotFoundError(f"Annotations file not found: {annotations_file}")
                
                full_dataset = NailSegmentationDataset(
                    data_dir=data_dir,
                #    annotations_file=annotations_file,
                    transform=get_train_transforms(self.config),
                    split='train',
                    config=self.config
                )
            except Exception as e:
                self.logger.error(f"Failed to create dataset: {e}")
                raise RuntimeError(f"Dataset creation failed: {e}") from e
            
            # Check dataset size
            dataset_size = len(full_dataset)
            if dataset_size == 0:
                raise ValueError(f"Dataset is empty. Check data directory: {data_dir}")
            
            if dataset_size < 2:
                raise ValueError(f"Dataset too small ({dataset_size} samples). Need at least 2 samples.")
            
            # Split dataset
            val_size = int(dataset_size * self.config.validation_split)
            train_size = dataset_size - val_size
            
            if train_size == 0 or val_size == 0:
                raise ValueError(f"Invalid dataset split. Train: {train_size}, Val: {val_size}")
            
            try:
                train_dataset, val_dataset = random_split(
                    full_dataset, [train_size, val_size],
                    generator=torch.Generator().manual_seed(42)
                )
            except Exception as e:
                self.logger.error(f"Failed to split dataset: {e}")
                raise RuntimeError(f"Dataset splitting failed: {e}") from e
            
            # Update validation dataset transform
            try:
                val_dataset.dataset.transform = get_val_transforms(self.config)
                val_dataset.dataset.split = 'val'
            except Exception as e:
                self.logger.warning(f"Could not update validation transforms: {e}")
            
            # Create data loaders
            try:
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=True,
                    num_workers=self.config.get_num_workers(),
                    pin_memory=self.config.pin_memory,
                    drop_last=True
                )
                
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    num_workers=self.config.get_num_workers(),
                    pin_memory=self.config.pin_memory
                )
            except Exception as e:
                self.logger.error(f"Failed to create data loaders: {e}")
                raise RuntimeError(f"DataLoader creation failed: {e}") from e
            
            self.logger.info(f"Dataset split: {train_size} training, {val_size} validation")
            return train_loader, val_loader
            
        except Exception as e:
            self.logger.error(f"Data setup failed: {e}")
            raise
    
    def prepare_data(self, data_dir: str) -> None:
        """
        Prepare training and validation datasets
        
        Args:
            data_dir: Path to data directory
        """
        self.train_loader, self.val_loader = self.setup_data(data_dir)
        
        # Store dataset references for convenience
        self.train_dataset = self.train_loader.dataset
        self.val_dataset = self.val_loader.dataset
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary of training metrics
            
        Raises:
            RuntimeError: If training epoch fails
        """
        try:
            self.model.train()
            total_loss = 0.0
            total_iou = 0.0
            total_dice = 0.0
            num_batches = 0
            
            progress_bar = tqdm(
                train_loader, 
                desc=f"Epoch {self.current_epoch + 1}/{self.config.epochs}",
                leave=False
            )
            
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    images = batch['image'].to(self.device, non_blocking=True)
                    masks = batch['mask'].to(self.device, non_blocking=True)
                    
                    self.optimizer.zero_grad()
                    
                    # Forward pass with mixed precision
                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(images)
                            loss = self.criterion(outputs, masks)
                        
                        # Backward pass
                        self.scaler.scale(loss).backward()
                        
                        # Gradient clipping
                        if self.config.gradient_clip_val > 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), 
                                self.config.gradient_clip_val
                            )
                        
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)
                        loss.backward()
                        
                        # Gradient clipping
                        if self.config.gradient_clip_val > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), 
                                self.config.gradient_clip_val
                            )
                        
                        self.optimizer.step()
                    
                    # Update metrics
                    total_loss += loss.item()
                    
                    # Calculate IoU and Dice
                    with torch.no_grad():
                        pred_masks = torch.sigmoid(outputs) > 0.5
                        intersection = (pred_masks * masks).sum()
                        union = pred_masks.sum() + masks.sum() - intersection
                        batch_iou = (intersection / (union + 1e-8)).item()
                        batch_dice = (2 * intersection / (pred_masks.sum() + masks.sum() + 1e-8)).item()
                        
                        total_iou += batch_iou
                        total_dice += batch_dice
                        num_batches += 1
                    
                    # Update progress bar
                    if batch_idx % self.config.log_interval == 0:
                        current_lr = self.optimizer.param_groups[0]['lr']
                        progress_bar.set_postfix({
                            'Loss': f"{loss.item():.4f}",
                            'IoU': f"{batch_iou:.4f}",
                            'LR': f"{current_lr:.2e}"
                        })
                        
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        self.logger.error(f"GPU out of memory at batch {batch_idx}. Try reducing batch size.")
                        # Clear cache and continue
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        raise RuntimeError(f"GPU out of memory: {e}") from e
                    else:
                        self.logger.error(f"Training batch {batch_idx} failed: {e}")
                        raise RuntimeError(f"Training batch failed: {e}") from e
                        
                except Exception as e:
                    self.logger.error(f"Unexpected error in training batch {batch_idx}: {e}")
                    raise RuntimeError(f"Training batch failed: {e}") from e
            
            # Calculate average metrics
            avg_loss = total_loss / len(train_loader)
            avg_iou = total_iou / num_batches if num_batches > 0 else 0.0
            avg_dice = total_dice / num_batches if num_batches > 0 else 0.0
            
            return {
                'loss': avg_loss,
                'iou': avg_iou,
                'dice': avg_dice
            }
            
        except Exception as e:
            self.logger.error(f"Training epoch failed: {e}")
            raise
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate for one epoch
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_iou = 0.0
        total_dice = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                images = batch['image'].to(self.device, non_blocking=True)
                masks = batch['mask'].to(self.device, non_blocking=True)
                
                # Forward pass
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                # Update metrics
                total_loss += loss.item()
                
                # Calculate IoU and Dice
                pred_masks = torch.sigmoid(outputs) > 0.5
                intersection = (pred_masks * masks).sum()
                union = pred_masks.sum() + masks.sum() - intersection
                batch_iou = (intersection / (union + 1e-8)).item()
                batch_dice = (2 * intersection / (pred_masks.sum() + masks.sum() + 1e-8)).item()
                
                total_iou += batch_iou
                total_dice += batch_dice
                num_batches += 1
        
        # Calculate average metrics
        avg_loss = total_loss / len(val_loader)
        avg_iou = total_iou / num_batches if num_batches > 0 else 0.0
        avg_dice = total_dice / num_batches if num_batches > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'iou': avg_iou,
            'dice': avg_dice
        }
    
    def train(self, data_dir: str) -> Dict[str, Any]:
        """
        Complete training loop
        
        Args:
            data_dir: Path to training data
            
        Returns:
            Training history and final metrics
        """
        self.logger.info("Starting training...")
        
        # Setup model
        self.setup_model()
        
        # Setup data if not already prepared
        if not hasattr(self, 'train_loader') or not hasattr(self, 'val_loader'):
            train_loader, val_loader = self.setup_data(data_dir)
        else:
            train_loader, val_loader = self.train_loader, self.val_loader
        
        # Save initial configuration
        config_path = os.path.join(self.config.output_dir, 'config.yaml')
        self.config.save(config_path)
        
        start_time = time.time()
        
        try:
            for epoch in range(self.config.epochs):
                self.current_epoch = epoch
                epoch_start = time.time()
                
                # Training phase
                train_metrics = self.train_epoch(train_loader)
                
                # Validation phase
                val_metrics = self.validate_epoch(val_loader)
                
                # Learning rate scheduling
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['loss'])
                    else:
                        self.scheduler.step()
                
                # Log metrics
                current_lr = self.optimizer.param_groups[0]['lr']
                self._log_epoch_metrics(train_metrics, val_metrics, current_lr)
                
                # Check for best model
                if val_metrics['iou'] > self.best_val_iou:
                    self.best_val_iou = val_metrics['iou']
                    self.best_val_loss = val_metrics['loss']
                    self.early_stopping_counter = 0
                    
                    # Save best model
                    self.checkpoint_manager.save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        metrics=val_metrics,
                        config=self.config,
                        is_best=True
                    )
                else:
                    self.early_stopping_counter += 1
                
                # Regular checkpoint saving
                if (epoch + 1) % self.config.save_interval == 0:
                    self.checkpoint_manager.save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        metrics=val_metrics,
                        config=self.config,
                        is_best=False
                    )
                
                # Early stopping check
                if self.early_stopping_counter >= self.config.early_stopping_patience:
                    self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
                
                epoch_time = time.time() - epoch_start
                self.logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed with error: {str(e)}")
            raise
        
        finally:
            # Save final checkpoint
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=self.current_epoch,
                metrics=val_metrics if 'val_metrics' in locals() else {},
                config=self.config,
                is_best=False,
                filename="final_checkpoint.pth"
            )
            
            # Close tensorboard writer
            if self.writer:
                self.writer.close()
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f}s")
        self.logger.info(f"Best validation IoU: {self.best_val_iou:.4f}")
        
        # Save training history
        history_path = os.path.join(self.config.output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        return {
            'history': self.training_history,
            'best_val_iou': self.best_val_iou,
            'best_val_loss': self.best_val_loss,
            'total_epochs': self.current_epoch + 1,
            'total_time': total_time
        }
    
    def _log_epoch_metrics(self, train_metrics: Dict[str, float], 
                          val_metrics: Dict[str, float], lr: float) -> None:
        """Log metrics for current epoch"""
        # Update training history
        self.training_history['train_loss'].append(train_metrics['loss'])
        self.training_history['val_loss'].append(val_metrics['loss'])
        self.training_history['train_iou'].append(train_metrics['iou'])
        self.training_history['val_iou'].append(val_metrics['iou'])
        self.training_history['train_dice'].append(train_metrics['dice'])
        self.training_history['val_dice'].append(val_metrics['dice'])
        self.training_history['learning_rate'].append(lr)
        
        # Console logging
        self.logger.info(
            f"Epoch {self.current_epoch + 1:3d} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Train IoU: {train_metrics['iou']:.4f} | "
            f"Val IoU: {val_metrics['iou']:.4f} | "
            f"LR: {lr:.2e}"
        )
        
        # Tensorboard logging
        if self.writer:
            epoch = self.current_epoch
            
            # Loss metrics
            self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/Validation', val_metrics['loss'], epoch)
            
            # IoU metrics
            self.writer.add_scalar('IoU/Train', train_metrics['iou'], epoch)
            self.writer.add_scalar('IoU/Validation', val_metrics['iou'], epoch)
            
            # Dice metrics
            self.writer.add_scalar('Dice/Train', train_metrics['dice'], epoch)
            self.writer.add_scalar('Dice/Validation', val_metrics['dice'], epoch)
            
            # Learning rate
            self.writer.add_scalar('Learning_Rate', lr, epoch)
            
            # Other metrics
            for metric_name, metric_value in train_metrics.items():
                if metric_name not in ['loss', 'iou', 'dice']:
                    self.writer.add_scalar(f'Train/{metric_name}', metric_value, epoch)
            
            for metric_name, metric_value in val_metrics.items():
                if metric_name not in ['loss', 'iou', 'dice']:
                    self.writer.add_scalar(f'Validation/{metric_name}', metric_value, epoch)
    
    def resume_training(self, checkpoint_path: str, data_dir: str) -> Dict[str, Any]:
        """
        Resume training from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            data_dir: Path to training data
            
        Returns:
            Training results
        """
        self.logger.info(f"Resuming training from: {checkpoint_path}")
        
        # Setup model and data
        self.setup_model()
        train_loader, val_loader = self.setup_data(data_dir)
        
        # Load checkpoint
        checkpoint_data = self.checkpoint_manager.load_checkpoint(
            checkpoint_path, self.model, self.optimizer, self.scheduler
        )
        
        # Restore training state
        self.current_epoch = checkpoint_data['epoch'] + 1
        self.training_history = checkpoint_data.get('history', self.training_history)
        
        # Update config epochs to continue from checkpoint
        remaining_epochs = self.config.epochs - self.current_epoch
        if remaining_epochs <= 0:
            self.logger.warning("Checkpoint epoch >= configured epochs. No training needed.")
            return {'message': 'Training already completed'}
        
        self.logger.info(f"Resuming from epoch {self.current_epoch + 1}")
        
        # Continue training
        return self.train(data_dir) 