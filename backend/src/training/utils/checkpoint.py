"""
Checkpoint Management for Training Pipeline

Provides comprehensive model checkpointing with versioning, best model tracking,
and automatic cleanup of old checkpoints.
"""

import os
import torch
import json
import glob
import shutil
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging
from datetime import datetime

from .config import TrainingConfig


class CheckpointManager:
    """
    Manages model checkpoints with automatic versioning and cleanup
    
    Features:
    - Automatic checkpoint versioning
    - Best model tracking
    - Configurable retention policy
    - Metadata storage
    - Resume training support
    """
    
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        """
        Initialize checkpoint manager
        
        Args:
            checkpoint_dir: Directory to store checkpoints
            max_checkpoints: Maximum number of regular checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.logger = logging.getLogger(__name__)
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Best model tracking
        self.best_model_path = self.checkpoint_dir / "best_model.pth"
        self.metadata_path = self.checkpoint_dir / "checkpoint_metadata.json"
        
        # Load existing metadata
        self.metadata = self._load_metadata()
    
    def save_checkpoint(self, 
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                       epoch: int,
                       metrics: Dict[str, float],
                       config: TrainingConfig,
                       is_best: bool = False,
                       filename: Optional[str] = None) -> str:
        """
        Save model checkpoint
        
        Args:
            model: PyTorch model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler state
            epoch: Current epoch number
            metrics: Training metrics
            config: Training configuration
            is_best: Whether this is the best model so far
            filename: Custom filename (optional)
            
        Returns:
            Path to saved checkpoint
        """
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'config': config.__dict__,
            'timestamp': datetime.now().isoformat(),
            'model_info': model.get_model_info() if hasattr(model, 'get_model_info') else {}
        }
        
        if filename:
            checkpoint_path = self.checkpoint_dir / filename
        else:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}_{timestamp}.pth"
        
        # Save checkpoint
        try:
            torch.save(checkpoint_data, checkpoint_path)
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Update metadata
            self._update_metadata(checkpoint_path, epoch, metrics, is_best)
            
            # Save as best model if applicable
            if is_best:
                shutil.copy2(checkpoint_path, self.best_model_path)
                self.logger.info(f"Best model updated: {self.best_model_path}")
            
            # Cleanup old checkpoints (but keep best model)
            if not is_best and not filename:
                self._cleanup_old_checkpoints()
            
            return str(checkpoint_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {str(e)}")
            raise
    
    def load_checkpoint(self, 
                       checkpoint_path: str,
                       model: torch.nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                       map_location: Optional[str] = None) -> Dict[str, Any]:
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            map_location: Device mapping for loading (optional)
            
        Returns:
            Dictionary containing loaded checkpoint data
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            # Load checkpoint data
            if map_location:
                checkpoint_data = torch.load(checkpoint_path, map_location=map_location)
            else:
                checkpoint_data = torch.load(checkpoint_path)
            
            # Load model state
            model.load_state_dict(checkpoint_data['model_state_dict'])
            self.logger.info(f"Model state loaded from: {checkpoint_path}")
            
            # Load optimizer state if provided
            if optimizer and 'optimizer_state_dict' in checkpoint_data:
                optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                self.logger.info("Optimizer state loaded")
            
            # Load scheduler state if provided
            if scheduler and 'scheduler_state_dict' in checkpoint_data:
                if checkpoint_data['scheduler_state_dict'] is not None:
                    scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
                    self.logger.info("Scheduler state loaded")
            
            return checkpoint_data
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {str(e)}")
            raise
    
    def load_best_model(self, 
                       model: torch.nn.Module,
                       map_location: Optional[str] = None) -> Dict[str, Any]:
        """
        Load the best model checkpoint
        
        Args:
            model: Model to load state into
            map_location: Device mapping for loading (optional)
            
        Returns:
            Dictionary containing loaded checkpoint data
        """
        if not self.best_model_path.exists():
            raise FileNotFoundError("No best model checkpoint found")
        
        return self.load_checkpoint(str(self.best_model_path), model, map_location=map_location)
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get path to the most recent checkpoint
        
        Returns:
            Path to latest checkpoint or None if no checkpoints exist
        """
        checkpoint_files = glob.glob(str(self.checkpoint_dir / "checkpoint_epoch_*.pth"))
        
        if not checkpoint_files:
            return None
        
        # Sort by modification time
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        return latest_checkpoint
    
    def list_checkpoints(self) -> Dict[str, Any]:
        """
        List all available checkpoints with metadata
        
        Returns:
            Dictionary containing checkpoint information
        """
        checkpoints = []
        
        # Regular checkpoints
        checkpoint_files = glob.glob(str(self.checkpoint_dir / "checkpoint_epoch_*.pth"))
        
        for checkpoint_path in sorted(checkpoint_files):
            try:
                checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
                checkpoints.append({
                    'path': checkpoint_path,
                    'epoch': checkpoint_data.get('epoch', 0),
                    'metrics': checkpoint_data.get('metrics', {}),
                    'timestamp': checkpoint_data.get('timestamp', ''),
                    'size_mb': os.path.getsize(checkpoint_path) / (1024 * 1024)
                })
            except Exception as e:
                self.logger.warning(f"Could not read checkpoint {checkpoint_path}: {str(e)}")
        
        return {
            'regular_checkpoints': checkpoints,
            'best_model_exists': self.best_model_path.exists(),
            'best_model_path': str(self.best_model_path) if self.best_model_path.exists() else None,
            'metadata': self.metadata
        }
    
    def delete_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Delete a specific checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint to delete
            
        Returns:
            True if deletion was successful
        """
        checkpoint_path = Path(checkpoint_path)
        
        try:
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                self.logger.info(f"Deleted checkpoint: {checkpoint_path}")
                
                # Update metadata
                self._remove_from_metadata(str(checkpoint_path))
                return True
            else:
                self.logger.warning(f"Checkpoint not found: {checkpoint_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to delete checkpoint: {str(e)}")
            return False
    
    def cleanup_all_checkpoints(self, keep_best: bool = True) -> None:
        """
        Delete all checkpoints
        
        Args:
            keep_best: Whether to keep the best model checkpoint
        """
        checkpoint_files = glob.glob(str(self.checkpoint_dir / "checkpoint_epoch_*.pth"))
        
        for checkpoint_path in checkpoint_files:
            try:
                os.remove(checkpoint_path)
                self.logger.info(f"Deleted checkpoint: {checkpoint_path}")
            except Exception as e:
                self.logger.error(f"Failed to delete {checkpoint_path}: {str(e)}")
        
        if not keep_best and self.best_model_path.exists():
            try:
                self.best_model_path.unlink()
                self.logger.info("Deleted best model checkpoint")
            except Exception as e:
                self.logger.error(f"Failed to delete best model: {str(e)}")
        
        # Reset metadata
        self.metadata = {'checkpoints': [], 'best_model': None}
        self._save_metadata()
    
    def export_model(self, 
                    checkpoint_path: str,
                    export_path: str,
                    export_format: str = 'state_dict') -> str:
        """
        Export model for deployment
        
        Args:
            checkpoint_path: Path to checkpoint to export
            export_path: Path for exported model
            export_format: Export format ('state_dict', 'full_model', 'onnx')
            
        Returns:
            Path to exported model
        """
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            
            if export_format == 'state_dict':
                # Export only model state dict
                torch.save(checkpoint_data['model_state_dict'], export_path)
                
            elif export_format == 'full_model':
                # Export full model with architecture
                # Note: This requires the model class to be available
                raise NotImplementedError("Full model export requires model reconstruction")
                
            elif export_format == 'onnx':
                # Export to ONNX format
                # Note: This requires example input
                raise NotImplementedError("ONNX export requires example input tensor")
                
            else:
                raise ValueError(f"Unsupported export format: {export_format}")
            
            self.logger.info(f"Model exported to: {export_path}")
            return str(export_path)
            
        except Exception as e:
            self.logger.error(f"Failed to export model: {str(e)}")
            raise
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load checkpoint metadata"""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load metadata: {str(e)}")
        
        return {'checkpoints': [], 'best_model': None}
    
    def _save_metadata(self) -> None:
        """Save checkpoint metadata"""
        try:
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {str(e)}")
    
    def _update_metadata(self, 
                        checkpoint_path: Path, 
                        epoch: int, 
                        metrics: Dict[str, float], 
                        is_best: bool) -> None:
        """Update checkpoint metadata"""
        checkpoint_info = {
            'path': str(checkpoint_path),
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'is_best': is_best
        }
        
        self.metadata['checkpoints'].append(checkpoint_info)
        
        if is_best:
            self.metadata['best_model'] = checkpoint_info
        
        self._save_metadata()
    
    def _remove_from_metadata(self, checkpoint_path: str) -> None:
        """Remove checkpoint from metadata"""
        self.metadata['checkpoints'] = [
            cp for cp in self.metadata['checkpoints'] 
            if cp['path'] != checkpoint_path
        ]
        self._save_metadata()
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints keeping only the most recent ones"""
        checkpoint_files = glob.glob(str(self.checkpoint_dir / "checkpoint_epoch_*.pth"))
        
        if len(checkpoint_files) <= self.max_checkpoints:
            return
        
        # Sort by modification time (oldest first)
        checkpoint_files.sort(key=os.path.getmtime)
        
        # Remove oldest checkpoints
        checkpoints_to_remove = checkpoint_files[:-self.max_checkpoints]
        
        for checkpoint_path in checkpoints_to_remove:
            try:
                os.remove(checkpoint_path)
                self.logger.info(f"Cleaned up old checkpoint: {checkpoint_path}")
                self._remove_from_metadata(checkpoint_path)
            except Exception as e:
                self.logger.error(f"Failed to cleanup {checkpoint_path}: {str(e)}")


class ModelVersionManager:
    """
    Manages model versions for production deployment
    
    Features:
    - Semantic versioning
    - Model metadata tracking
    - Deployment history
    """
    
    def __init__(self, models_dir: str):
        """
        Initialize model version manager
        
        Args:
            models_dir: Directory to store versioned models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        self.versions_file = self.models_dir / "versions.json"
        self.versions = self._load_versions()
    
    def save_version(self, 
                    model_path: str,
                    version: str,
                    description: str = "",
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save a new model version
        
        Args:
            model_path: Path to model file
            version: Version string (e.g., "1.0.0")
            description: Version description
            metadata: Additional metadata
            
        Returns:
            Path to versioned model
        """
        version_dir = self.models_dir / f"v{version}"
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model file
        model_filename = Path(model_path).name
        versioned_model_path = version_dir / model_filename
        shutil.copy2(model_path, versioned_model_path)
        
        # Save version metadata
        version_info = {
            'version': version,
            'description': description,
            'model_path': str(versioned_model_path),
            'original_path': model_path,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.versions[version] = version_info
        self._save_versions()
        
        self.logger.info(f"Model version {version} saved: {versioned_model_path}")
        return str(versioned_model_path)
    
    def get_version(self, version: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific version"""
        return self.versions.get(version)
    
    def list_versions(self) -> Dict[str, Any]:
        """List all available versions"""
        return self.versions
    
    def get_latest_version(self) -> Optional[str]:
        """Get the latest version string"""
        if not self.versions:
            return None
        
        # Simple version sorting (assumes semantic versioning)
        versions = list(self.versions.keys())
        versions.sort(key=lambda v: [int(x) for x in v.split('.')])
        return versions[-1]
    
    def _load_versions(self) -> Dict[str, Any]:
        """Load version metadata"""
        if self.versions_file.exists():
            try:
                with open(self.versions_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load versions: {str(e)}")
        
        return {}
    
    def _save_versions(self) -> None:
        """Save version metadata"""
        try:
            with open(self.versions_file, 'w') as f:
                json.dump(self.versions, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save versions: {str(e)}") 