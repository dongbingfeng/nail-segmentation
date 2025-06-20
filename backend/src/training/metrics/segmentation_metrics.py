"""
Segmentation Metrics for Nail Segmentation Evaluation

Implements comprehensive metrics for evaluating binary segmentation performance,
including IoU, Dice coefficient, precision, recall, and boundary metrics.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
import cv2


class SegmentationMetrics:
    """
    Comprehensive segmentation metrics calculator
    """
    
    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6):
        """
        Initialize segmentation metrics
        
        Args:
            threshold: Threshold for binary predictions
            smooth: Smoothing factor for numerical stability
        """
        self.threshold = threshold
        self.smooth = smooth
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics"""
        self.predictions = []
        self.targets = []
        self.scores = []
    
    def update(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        scores: Optional[torch.Tensor] = None
    ):
        """
        Update metrics with new batch
        
        Args:
            predictions: Predicted masks (B, C, H, W) or (B, H, W)
            targets: Ground truth masks (B, C, H, W) or (B, H, W)
            scores: Raw prediction scores before thresholding
        """
        # Convert to numpy and flatten
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        if scores is not None and isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        
        # Ensure binary predictions
        binary_preds = (predictions > self.threshold).astype(np.float32)
        binary_targets = (targets > 0.5).astype(np.float32)
        
        # Store for batch computation
        self.predictions.extend(binary_preds.flatten())
        self.targets.extend(binary_targets.flatten())
        
        if scores is not None:
            self.scores.extend(scores.flatten())
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics
        
        Returns:
            Dictionary of computed metrics
        """
        if not self.predictions:
            return {}
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Basic metrics
        metrics = {}
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()
        
        # Basic classification metrics
        metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # F1 Score
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1_score'] = 0.0
        
        # Dice coefficient
        intersection = (predictions * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        metrics['dice'] = float(dice)
        
        # IoU (Jaccard Index)
        union = predictions.sum() + targets.sum() - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        metrics['iou'] = float(iou)
        
        # Sensitivity and Specificity
        metrics['sensitivity'] = metrics['recall']  # Same as recall
        
        # Balanced accuracy
        metrics['balanced_accuracy'] = (metrics['sensitivity'] + metrics['specificity']) / 2.0
        
        # Matthews Correlation Coefficient
        mcc_num = (tp * tn) - (fp * fn)
        mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        metrics['mcc'] = mcc_num / mcc_den if mcc_den > 0 else 0.0
        
        # AUC metrics if scores are available
        if self.scores:
            scores = np.array(self.scores)
            try:
                metrics['auc_roc'] = roc_auc_score(targets, scores)
                metrics['auc_pr'] = average_precision_score(targets, scores)
            except ValueError:
                metrics['auc_roc'] = 0.0
                metrics['auc_pr'] = 0.0
        
        return metrics


class BatchMetrics:
    """
    Calculate metrics for individual batches
    """
    
    @staticmethod
    def dice_coefficient(
        predictions: torch.Tensor, 
        targets: torch.Tensor, 
        smooth: float = 1e-6
    ) -> torch.Tensor:
        """
        Calculate Dice coefficient for batch
        
        Args:
            predictions: Predicted masks (B, C, H, W)
            targets: Ground truth masks (B, C, H, W)
            smooth: Smoothing factor
            
        Returns:
            Dice coefficient per sample in batch
        """
        # Flatten spatial dimensions
        predictions = predictions.view(predictions.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        
        # Calculate intersection and union
        intersection = (predictions * targets).sum(dim=1)
        dice = (2.0 * intersection + smooth) / (
            predictions.sum(dim=1) + targets.sum(dim=1) + smooth
        )
        
        return dice
    
    @staticmethod
    def iou_score(
        predictions: torch.Tensor, 
        targets: torch.Tensor, 
        smooth: float = 1e-6
    ) -> torch.Tensor:
        """
        Calculate IoU score for batch
        
        Args:
            predictions: Predicted masks (B, C, H, W)
            targets: Ground truth masks (B, C, H, W)
            smooth: Smoothing factor
            
        Returns:
            IoU score per sample in batch
        """
        # Flatten spatial dimensions
        predictions = predictions.view(predictions.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        
        # Calculate intersection and union
        intersection = (predictions * targets).sum(dim=1)
        union = predictions.sum(dim=1) + targets.sum(dim=1) - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        
        return iou
    
    @staticmethod
    def pixel_accuracy(
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate pixel accuracy for batch
        
        Args:
            predictions: Predicted masks (B, C, H, W)
            targets: Ground truth masks (B, C, H, W)
            
        Returns:
            Pixel accuracy per sample in batch
        """
        # Flatten spatial dimensions
        predictions = predictions.view(predictions.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        
        # Calculate accuracy
        correct = (predictions == targets).float()
        accuracy = correct.mean(dim=1)
        
        return accuracy


class BoundaryMetrics:
    """
    Metrics specifically for boundary evaluation
    """
    
    @staticmethod
    def boundary_iou(
        predictions: np.ndarray, 
        targets: np.ndarray, 
        dilation_radius: int = 2
    ) -> float:
        """
        Calculate IoU specifically for boundary regions
        
        Args:
            predictions: Predicted binary mask
            targets: Ground truth binary mask
            dilation_radius: Radius for boundary dilation
            
        Returns:
            Boundary IoU score
        """
        # Create boundary masks
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_radius*2+1, dilation_radius*2+1))
        
        # Get boundaries
        pred_boundary = cv2.morphologyEx(predictions.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
        target_boundary = cv2.morphologyEx(targets.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
        
        # Calculate IoU on boundaries
        intersection = (pred_boundary * target_boundary).sum()
        union = pred_boundary.sum() + target_boundary.sum() - intersection
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def hausdorff_distance(
        predictions: np.ndarray, 
        targets: np.ndarray
    ) -> float:
        """
        Calculate Hausdorff distance between boundaries
        
        Args:
            predictions: Predicted binary mask
            targets: Ground truth binary mask
            
        Returns:
            Hausdorff distance
        """
        # Find contours
        pred_contours, _ = cv2.findContours(
            predictions.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        target_contours, _ = cv2.findContours(
            targets.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not pred_contours or not target_contours:
            return float('inf')
        
        # Get boundary points
        pred_points = np.vstack([contour.reshape(-1, 2) for contour in pred_contours])
        target_points = np.vstack([contour.reshape(-1, 2) for contour in target_contours])
        
        # Calculate distances
        def directed_hausdorff(points1, points2):
            distances = []
            for p1 in points1:
                min_dist = np.min(np.sqrt(np.sum((points2 - p1)**2, axis=1)))
                distances.append(min_dist)
            return np.max(distances)
        
        # Bidirectional Hausdorff distance
        h1 = directed_hausdorff(pred_points, target_points)
        h2 = directed_hausdorff(target_points, pred_points)
        
        return max(h1, h2)
    
    @staticmethod
    def average_surface_distance(
        predictions: np.ndarray, 
        targets: np.ndarray
    ) -> float:
        """
        Calculate average surface distance
        
        Args:
            predictions: Predicted binary mask
            targets: Ground truth binary mask
            
        Returns:
            Average surface distance
        """
        # Find contours
        pred_contours, _ = cv2.findContours(
            predictions.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        target_contours, _ = cv2.findContours(
            targets.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not pred_contours or not target_contours:
            return float('inf')
        
        # Get boundary points
        pred_points = np.vstack([contour.reshape(-1, 2) for contour in pred_contours])
        target_points = np.vstack([contour.reshape(-1, 2) for contour in target_contours])
        
        # Calculate average distances
        distances = []
        for p1 in pred_points:
            min_dist = np.min(np.sqrt(np.sum((target_points - p1)**2, axis=1)))
            distances.append(min_dist)
        
        for p2 in target_points:
            min_dist = np.min(np.sqrt(np.sum((pred_points - p2)**2, axis=1)))
            distances.append(min_dist)
        
        return np.mean(distances)


class MetricsTracker:
    """
    Track metrics across training epochs
    """
    
    def __init__(self, metrics_list: List[str]):
        """
        Initialize metrics tracker
        
        Args:
            metrics_list: List of metric names to track
        """
        self.metrics_list = metrics_list
        self.history = {metric: [] for metric in metrics_list}
        self.best_values = {metric: 0.0 for metric in metrics_list}
        self.best_epochs = {metric: 0 for metric in metrics_list}
    
    def update(self, epoch: int, metrics: Dict[str, float]):
        """
        Update metrics for current epoch
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of metric values
        """
        for metric in self.metrics_list:
            if metric in metrics:
                value = metrics[metric]
                self.history[metric].append(value)
                
                # Update best values (assuming higher is better for most metrics)
                if value > self.best_values[metric]:
                    self.best_values[metric] = value
                    self.best_epochs[metric] = epoch
    
    def get_best_metrics(self) -> Dict[str, Tuple[float, int]]:
        """
        Get best metric values and their epochs
        
        Returns:
            Dictionary mapping metric names to (best_value, best_epoch) tuples
        """
        return {
            metric: (self.best_values[metric], self.best_epochs[metric])
            for metric in self.metrics_list
        }
    
    def get_current_metrics(self) -> Dict[str, float]:
        """
        Get current (latest) metric values
        
        Returns:
            Dictionary of current metric values
        """
        return {
            metric: self.history[metric][-1] if self.history[metric] else 0.0
            for metric in self.metrics_list
        }
    
    def get_metric_history(self, metric: str) -> List[float]:
        """
        Get history for specific metric
        
        Args:
            metric: Metric name
            
        Returns:
            List of metric values across epochs
        """
        return self.history.get(metric, [])


def calculate_comprehensive_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate comprehensive metrics for a batch
    
    Args:
        predictions: Predicted masks (B, C, H, W)
        targets: Ground truth masks (B, C, H, W)
        threshold: Threshold for binary predictions
        
    Returns:
        Dictionary of computed metrics
    """
    # Convert to binary
    binary_preds = (predictions > threshold).float()
    binary_targets = (targets > 0.5).float()
    
    # Calculate batch metrics
    dice_scores = BatchMetrics.dice_coefficient(binary_preds, binary_targets)
    iou_scores = BatchMetrics.iou_score(binary_preds, binary_targets)
    pixel_acc = BatchMetrics.pixel_accuracy(binary_preds, binary_targets)
    
    # Aggregate metrics
    metrics = {
        'dice': dice_scores.mean().item(),
        'iou': iou_scores.mean().item(),
        'pixel_accuracy': pixel_acc.mean().item(),
        'dice_std': dice_scores.std().item(),
        'iou_std': iou_scores.std().item()
    }
    
    return metrics


def evaluate_model_performance(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate model performance on a dataset
    
    Args:
        model: Trained model
        dataloader: Data loader for evaluation
        device: Device to run evaluation on
        threshold: Threshold for binary predictions
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    metrics_calculator = SegmentationMetrics(threshold=threshold)
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Update metrics
            metrics_calculator.update(outputs, masks, outputs)
    
    # Compute final metrics
    final_metrics = metrics_calculator.compute()
    
    return final_metrics


class MultiThresholdMetrics:
    """
    Calculate metrics across multiple thresholds for optimal threshold selection
    """
    
    def __init__(self, thresholds: List[float] = None):
        """
        Initialize multi-threshold metrics
        
        Args:
            thresholds: List of thresholds to evaluate
        """
        if thresholds is None:
            self.thresholds = [i * 0.1 for i in range(1, 10)]  # 0.1 to 0.9
        else:
            self.thresholds = thresholds
        
        self.metrics_per_threshold = {}
    
    def evaluate(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> Dict[float, Dict[str, float]]:
        """
        Evaluate metrics across all thresholds
        
        Args:
            predictions: Predicted probability maps
            targets: Ground truth masks
            
        Returns:
            Dictionary mapping thresholds to their metrics
        """
        results = {}
        
        for threshold in self.thresholds:
            metrics_calc = SegmentationMetrics(threshold=threshold)
            metrics_calc.update(predictions, targets, predictions)
            metrics = metrics_calc.compute()
            results[threshold] = metrics
        
        self.metrics_per_threshold = results
        return results
    
    def get_optimal_threshold(self, metric: str = 'f1_score') -> Tuple[float, float]:
        """
        Get optimal threshold based on specified metric
        
        Args:
            metric: Metric to optimize for
            
        Returns:
            Tuple of (optimal_threshold, best_metric_value)
        """
        if not self.metrics_per_threshold:
            raise ValueError("Must call evaluate() first")
        
        best_threshold = None
        best_value = -1
        
        for threshold, metrics in self.metrics_per_threshold.items():
            if metric in metrics and metrics[metric] > best_value:
                best_value = metrics[metric]
                best_threshold = threshold
        
        return best_threshold, best_value 