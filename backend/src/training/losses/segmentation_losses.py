"""
Segmentation Loss Functions for Nail Segmentation

Implements various loss functions optimized for binary segmentation tasks,
particularly effective for small datasets and imbalanced classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation
    
    Particularly effective for segmentation tasks with class imbalance
    """
    
    def __init__(self, smooth: float = 1e-6, reduction: str = 'mean'):
        """
        Initialize Dice Loss
        
        Args:
            smooth: Smoothing factor to avoid division by zero
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice Loss
        
        Args:
            predictions: Predicted segmentation masks (B, C, H, W)
            targets: Ground truth masks (B, C, H, W)
            
        Returns:
            Dice loss value
        """
        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Calculate intersection and union
        intersection = (predictions * targets).sum()
        dice_score = (2.0 * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )
        
        dice_loss = 1.0 - dice_score
        
        if self.reduction == 'mean':
            return dice_loss
        elif self.reduction == 'sum':
            return dice_loss * predictions.numel()
        else:
            return dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    Focuses learning on hard examples by down-weighting easy examples
    """
    
    def __init__(
        self, 
        alpha: float = 1.0, 
        gamma: float = 2.0, 
        reduction: str = 'mean'
    ):
        """
        Initialize Focal Loss
        
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate Focal Loss
        
        Args:
            predictions: Predicted probabilities (B, C, H, W)
            targets: Ground truth masks (B, C, H, W)
            
        Returns:
            Focal loss value
        """
        # Calculate binary cross entropy
        bce_loss = F.binary_cross_entropy(predictions, targets, reduction='none')
        
        # Calculate p_t
        p_t = predictions * targets + (1 - predictions) * (1 - targets)
        
        # Calculate alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Calculate focal weight
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice Loss
    
    Allows for different weighting of false positives and false negatives
    """
    
    def __init__(
        self, 
        alpha: float = 0.3, 
        beta: float = 0.7, 
        smooth: float = 1e-6,
        reduction: str = 'mean'
    ):
        """
        Initialize Tversky Loss
        
        Args:
            alpha: Weight for false positives
            beta: Weight for false negatives
            smooth: Smoothing factor
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate Tversky Loss
        
        Args:
            predictions: Predicted segmentation masks (B, C, H, W)
            targets: Ground truth masks (B, C, H, W)
            
        Returns:
            Tversky loss value
        """
        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Calculate true positives, false positives, false negatives
        true_pos = (predictions * targets).sum()
        false_pos = (predictions * (1 - targets)).sum()
        false_neg = ((1 - predictions) * targets).sum()
        
        # Calculate Tversky index
        tversky_index = (true_pos + self.smooth) / (
            true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth
        )
        
        tversky_loss = 1.0 - tversky_index
        
        if self.reduction == 'mean':
            return tversky_loss
        elif self.reduction == 'sum':
            return tversky_loss * predictions.numel()
        else:
            return tversky_loss


class IoULoss(nn.Module):
    """
    Intersection over Union (IoU) Loss
    
    Directly optimizes the IoU metric
    """
    
    def __init__(self, smooth: float = 1e-6, reduction: str = 'mean'):
        """
        Initialize IoU Loss
        
        Args:
            smooth: Smoothing factor to avoid division by zero
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(IoULoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate IoU Loss
        
        Args:
            predictions: Predicted segmentation masks (B, C, H, W)
            targets: Ground truth masks (B, C, H, W)
            
        Returns:
            IoU loss value
        """
        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Calculate intersection and union
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum() - intersection
        
        # Calculate IoU
        iou = (intersection + self.smooth) / (union + self.smooth)
        iou_loss = 1.0 - iou
        
        if self.reduction == 'mean':
            return iou_loss
        elif self.reduction == 'sum':
            return iou_loss * predictions.numel()
        else:
            return iou_loss


class CombinedLoss(nn.Module):
    """
    Combined loss function for better segmentation performance
    
    Combines multiple loss functions with configurable weights
    """
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_weight: float = 0.3,
        bce_weight: float = 0.2,
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0,
        dice_smooth: float = 1e-6
    ):
        """
        Initialize Combined Loss
        
        Args:
            dice_weight: Weight for Dice loss
            focal_weight: Weight for Focal loss
            bce_weight: Weight for BCE loss
            focal_alpha: Alpha parameter for Focal loss
            focal_gamma: Gamma parameter for Focal loss
            dice_smooth: Smoothing factor for Dice loss
        """
        super(CombinedLoss, self).__init__()
        
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.bce_weight = bce_weight
        
        self.dice_loss = DiceLoss(smooth=dice_smooth)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.bce_loss = nn.BCELoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate Combined Loss
        
        Args:
            predictions: Predicted segmentation masks (B, C, H, W)
            targets: Ground truth masks (B, C, H, W)
            
        Returns:
            Combined loss value
        """
        dice = self.dice_loss(predictions, targets)
        focal = self.focal_loss(predictions, targets)
        bce = self.bce_loss(predictions, targets)
        
        combined_loss = (
            self.dice_weight * dice +
            self.focal_weight * focal +
            self.bce_weight * bce
        )
        
        return combined_loss


class BoundaryLoss(nn.Module):
    """
    Boundary Loss for better edge segmentation
    
    Focuses on boundary pixels for improved edge detection
    """
    
    def __init__(self, theta0: float = 3.0, theta: float = 5.0):
        """
        Initialize Boundary Loss
        
        Args:
            theta0: Distance threshold for boundary
            theta: Scaling factor
        """
        super(BoundaryLoss, self).__init__()
        self.theta0 = theta0
        self.theta = theta
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate Boundary Loss
        
        Args:
            predictions: Predicted segmentation masks (B, C, H, W)
            targets: Ground truth masks (B, C, H, W)
            
        Returns:
            Boundary loss value
        """
        # Calculate distance transform (simplified version)
        # In practice, you might want to use scipy.ndimage.distance_transform_edt
        
        # Create boundary mask
        kernel = torch.ones(3, 3, device=targets.device)
        boundary_targets = F.conv2d(
            targets, 
            kernel.unsqueeze(0).unsqueeze(0), 
            padding=1
        )
        boundary_targets = (boundary_targets > 0) & (boundary_targets < 9)
        boundary_targets = boundary_targets.float()
        
        # Calculate boundary loss
        boundary_loss = F.binary_cross_entropy(
            predictions * boundary_targets,
            targets * boundary_targets
        )
        
        return boundary_loss


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss
    
    Applies different weights to positive and negative classes
    """
    
    def __init__(self, pos_weight: Optional[float] = None):
        """
        Initialize Weighted BCE Loss
        
        Args:
            pos_weight: Weight for positive class
        """
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate Weighted BCE Loss
        
        Args:
            predictions: Predicted probabilities (B, C, H, W)
            targets: Ground truth masks (B, C, H, W)
            
        Returns:
            Weighted BCE loss value
        """
        if self.pos_weight is not None:
            pos_weight = torch.tensor(self.pos_weight, device=predictions.device)
            return F.binary_cross_entropy_with_logits(
                predictions, targets, pos_weight=pos_weight
            )
        else:
            return F.binary_cross_entropy(predictions, targets)


class AdaptiveLoss(nn.Module):
    """
    Adaptive loss that adjusts weights based on training progress
    
    Starts with BCE for stability, gradually shifts to Dice for better IoU
    """
    
    def __init__(self, max_epochs: int = 100):
        """
        Initialize Adaptive Loss
        
        Args:
            max_epochs: Maximum number of training epochs
        """
        super(AdaptiveLoss, self).__init__()
        self.max_epochs = max_epochs
        self.current_epoch = 0
        
        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceLoss()
    
    def set_epoch(self, epoch: int):
        """Set current epoch for adaptive weighting"""
        self.current_epoch = epoch
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate Adaptive Loss
        
        Args:
            predictions: Predicted segmentation masks (B, C, H, W)
            targets: Ground truth masks (B, C, H, W)
            
        Returns:
            Adaptive loss value
        """
        # Calculate adaptive weights
        progress = min(self.current_epoch / self.max_epochs, 1.0)
        bce_weight = 1.0 - progress
        dice_weight = progress
        
        bce = self.bce_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets)
        
        adaptive_loss = bce_weight * bce + dice_weight * dice
        
        return adaptive_loss


def create_loss_function(
    loss_type: str = 'combined',
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss functions
    
    Args:
        loss_type: Type of loss function
        **kwargs: Additional arguments for loss function
        
    Returns:
        Loss function instance
    """
    if loss_type == 'dice':
        return DiceLoss(**kwargs)
    elif loss_type == 'focal':
        return FocalLoss(**kwargs)
    elif loss_type == 'tversky':
        return TverskyLoss(**kwargs)
    elif loss_type == 'iou':
        return IoULoss(**kwargs)
    elif loss_type == 'combined':
        return CombinedLoss(**kwargs)
    elif loss_type == 'boundary':
        return BoundaryLoss(**kwargs)
    elif loss_type == 'weighted_bce':
        return WeightedBCELoss(**kwargs)
    elif loss_type == 'adaptive':
        return AdaptiveLoss(**kwargs)
    elif loss_type == 'bce':
        return nn.BCELoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def calculate_class_weights(targets: torch.Tensor) -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets
    
    Args:
        targets: Ground truth masks
        
    Returns:
        Class weights tensor
    """
    # Calculate positive and negative pixel counts
    pos_count = targets.sum()
    neg_count = targets.numel() - pos_count
    
    # Calculate weights (inverse frequency)
    total_count = targets.numel()
    pos_weight = total_count / (2.0 * pos_count) if pos_count > 0 else 1.0
    neg_weight = total_count / (2.0 * neg_count) if neg_count > 0 else 1.0
    
    return torch.tensor([neg_weight, pos_weight])


def get_loss_weights(dataset_stats: dict) -> dict:
    """
    Get recommended loss weights based on dataset statistics
    
    Args:
        dataset_stats: Dataset statistics dictionary
        
    Returns:
        Dictionary with recommended loss weights
    """
    # Extract statistics
    pos_ratio = dataset_stats.get('positive_pixel_ratio', 0.1)
    
    # Calculate weights based on class imbalance
    if pos_ratio < 0.05:  # Very imbalanced
        weights = {
            'dice_weight': 0.6,
            'focal_weight': 0.4,
            'bce_weight': 0.0,
            'focal_alpha': 0.75,
            'focal_gamma': 3.0
        }
    elif pos_ratio < 0.2:  # Moderately imbalanced
        weights = {
            'dice_weight': 0.5,
            'focal_weight': 0.3,
            'bce_weight': 0.2,
            'focal_alpha': 0.5,
            'focal_gamma': 2.0
        }
    else:  # Relatively balanced
        weights = {
            'dice_weight': 0.4,
            'focal_weight': 0.2,
            'bce_weight': 0.4,
            'focal_alpha': 0.25,
            'focal_gamma': 1.5
        }
    
    return weights 