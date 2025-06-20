"""
U-Net Postprocessing Pipeline

This module provides postprocessing functionality for U-Net model outputs including
threshold application, mask refinement, confidence scoring, and format conversion.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from scipy import ndimage
from skimage import measure, morphology

from src.config.unet_config import UNetConfig

logger = logging.getLogger(__name__)


class UNetPostprocessor:
    """
    Comprehensive U-Net output postprocessing with multiple output formats.
    
    This postprocessor provides:
    - Sigmoid threshold application with configurable thresholds
    - Mask refinement using morphological operations
    - Confidence score calculation per pixel and overall
    - Multiple output formats: binary mask, confidence heatmap, contours
    - Batch postprocessing support
    """
    
    def __init__(self, config: UNetConfig):
        """
        Initialize postprocessor with configuration.
        
        Args:
            config: U-Net configuration containing postprocessing parameters
        """
        self.config = config
        self.threshold = config.inference.threshold
        self.enable_confidence = config.inference.enable_confidence
        
        logger.info(f"Postprocessor initialized with threshold: {self.threshold}")
    
    def postprocess(self, model_output: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """
        Postprocess U-Net model output with comprehensive analysis.
        
        Args:
            model_output: Raw model output tensor with shape (B, 1, H, W)
            **kwargs: Additional postprocessing parameters:
                - threshold: Override default threshold
                - return_confidence: Override confidence calculation
                - refine_mask: Enable/disable mask refinement
                - return_contours: Enable/disable contour extraction
                
        Returns:
            Dictionary containing processed results
        """
        # Extract parameters with defaults
        threshold = kwargs.get('threshold', self.threshold)
        return_confidence = kwargs.get('return_confidence', self.enable_confidence)
        refine_mask = kwargs.get('refine_mask', True)
        return_contours = kwargs.get('return_contours', True)
        
        try:
            # Validate input
            if model_output.dim() != 4:
                raise ValueError(f"Expected 4D tensor (B, C, H, W), got {model_output.dim()}D")
            
            if model_output.shape[1] != 1:
                raise ValueError(f"Expected single channel output, got {model_output.shape[1]} channels")
            
            batch_size = model_output.shape[0]
            
            # Process each item in batch
            results = []
            for i in range(batch_size):
                single_output = model_output[i:i+1]  # Keep batch dimension
                result = self._postprocess_single(
                    single_output,
                    threshold=threshold,
                    return_confidence=return_confidence,
                    refine_mask=refine_mask,
                    return_contours=return_contours
                )
                results.append(result)
            
            # Return single result if batch size is 1, otherwise return list
            if batch_size == 1:
                return results[0]
            else:
                return {'batch_results': results, 'batch_size': batch_size}
                
        except Exception as e:
            logger.error(f"Postprocessing failed: {e}")
            raise
    
    def _postprocess_single(self, 
                          output: torch.Tensor,
                          threshold: float,
                          return_confidence: bool,
                          refine_mask: bool,
                          return_contours: bool) -> Dict[str, Any]:
        """
        Postprocess a single model output.
        
        Args:
            output: Single model output with shape (1, 1, H, W)
            threshold: Sigmoid threshold for binary mask
            return_confidence: Whether to calculate confidence scores
            refine_mask: Whether to apply mask refinement
            return_contours: Whether to extract contours
            
        Returns:
            Dictionary containing processed results for single image
        """
        # Apply sigmoid activation
        sigmoid_output = torch.sigmoid(output)
        
        # Remove batch and channel dimensions for processing
        sigmoid_map = sigmoid_output.squeeze().detach().cpu().numpy()  # Shape: (H, W)
        
        # Create binary mask
        binary_mask = (sigmoid_map >= threshold).astype(np.uint8)
        
        # Apply mask refinement if enabled
        if refine_mask:
            binary_mask = self._refine_mask(binary_mask)
        
        # Prepare results
        result = {
            'binary_mask': binary_mask,
            'threshold_used': threshold,
            'mask_area': np.sum(binary_mask),
            'mask_area_ratio': np.sum(binary_mask) / (binary_mask.shape[0] * binary_mask.shape[1])
        }
        
        # Add confidence information if requested
        if return_confidence:
            confidence_info = self._calculate_confidence(sigmoid_map, binary_mask)
            result.update(confidence_info)
        
        # Add contour information if requested
        if return_contours:
            contour_info = self._extract_contours(binary_mask)
            result.update(contour_info)
        
        # Add raw outputs for advanced use cases
        result['raw_output'] = {
            'sigmoid_map': sigmoid_map,
            'original_shape': sigmoid_map.shape
        }
        
        return result
    
    def _refine_mask(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to refine the binary mask.
        
        Args:
            binary_mask: Binary mask array with shape (H, W)
            
        Returns:
            Refined binary mask
        """
        try:
            # Create morphological kernel
            kernel_size = max(3, min(binary_mask.shape) // 100)  # Adaptive kernel size
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            
            # Remove small noise with opening
            refined_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            
            # Fill small holes with closing
            refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
            
            # Remove small connected components
            min_area = max(100, (binary_mask.shape[0] * binary_mask.shape[1]) // 1000)
            refined_mask = self._remove_small_components(refined_mask, min_area)
            
            logger.debug(f"Mask refinement: {np.sum(binary_mask)} -> {np.sum(refined_mask)} pixels")
            
            return refined_mask
            
        except Exception as e:
            logger.warning(f"Mask refinement failed: {e}, returning original mask")
            return binary_mask
    
    def _remove_small_components(self, binary_mask: np.ndarray, min_area: int) -> np.ndarray:
        """
        Remove small connected components from binary mask.
        
        Args:
            binary_mask: Binary mask array
            min_area: Minimum area for components to keep
            
        Returns:
            Cleaned binary mask
        """
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        
        # Create output mask
        cleaned_mask = np.zeros_like(binary_mask)
        
        # Keep components that meet size criteria
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                cleaned_mask[labels == i] = 1
        
        return cleaned_mask
    
    def _calculate_confidence(self, sigmoid_map: np.ndarray, binary_mask: np.ndarray) -> Dict[str, Any]:
        """
        Calculate confidence scores for the segmentation.
        
        Args:
            sigmoid_map: Sigmoid probability map with shape (H, W)
            binary_mask: Binary mask with shape (H, W)
            
        Returns:
            Dictionary containing confidence metrics
        """
        try:
            # Overall confidence metrics
            mean_confidence = np.mean(sigmoid_map)
            max_confidence = np.max(sigmoid_map)
            min_confidence = np.min(sigmoid_map)
            
            # Confidence within predicted mask
            if np.any(binary_mask):
                mask_confidence = np.mean(sigmoid_map[binary_mask == 1])
                mask_min_confidence = np.min(sigmoid_map[binary_mask == 1])
                mask_max_confidence = np.max(sigmoid_map[binary_mask == 1])
            else:
                mask_confidence = 0.0
                mask_min_confidence = 0.0
                mask_max_confidence = 0.0
            
            # Confidence outside predicted mask (should be low for good predictions)
            if np.any(binary_mask == 0):
                background_confidence = np.mean(sigmoid_map[binary_mask == 0])
                background_max_confidence = np.max(sigmoid_map[binary_mask == 0])
            else:
                background_confidence = 0.0
                background_max_confidence = 0.0
            
            # Prediction certainty (how far from threshold)
            threshold_distance = np.abs(sigmoid_map - self.threshold)
            mean_certainty = np.mean(threshold_distance)
            
            # Edge confidence (confidence at mask boundaries)
            edge_confidence = self._calculate_edge_confidence(sigmoid_map, binary_mask)
            
            return {
                'confidence_scores': {
                    'overall_mean': float(mean_confidence),
                    'overall_max': float(max_confidence),
                    'overall_min': float(min_confidence),
                    'mask_mean': float(mask_confidence),
                    'mask_min': float(mask_min_confidence),
                    'mask_max': float(mask_max_confidence),
                    'background_mean': float(background_confidence),
                    'background_max': float(background_max_confidence),
                    'mean_certainty': float(mean_certainty),
                    'edge_confidence': float(edge_confidence)
                },
                'confidence_map': sigmoid_map  # Full confidence heatmap
            }
            
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            return {
                'confidence_scores': {},
                'confidence_map': sigmoid_map
            }
    
    def _calculate_edge_confidence(self, sigmoid_map: np.ndarray, binary_mask: np.ndarray) -> float:
        """
        Calculate confidence at mask edges.
        
        Args:
            sigmoid_map: Sigmoid probability map
            binary_mask: Binary mask
            
        Returns:
            Mean confidence at mask edges
        """
        try:
            # Find mask edges
            edges = cv2.Canny(binary_mask.astype(np.uint8) * 255, 50, 150)
            edge_pixels = edges > 0
            
            if np.any(edge_pixels):
                edge_confidence = np.mean(sigmoid_map[edge_pixels])
                return edge_confidence
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Edge confidence calculation failed: {e}")
            return 0.0
    
    def _extract_contours(self, binary_mask: np.ndarray) -> Dict[str, Any]:
        """
        Extract contours from binary mask.
        
        Args:
            binary_mask: Binary mask with shape (H, W)
            
        Returns:
            Dictionary containing contour information
        """
        try:
            # Find contours
            contours, hierarchy = cv2.findContours(
                binary_mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                return {
                    'contours': [],
                    'num_contours': 0,
                    'contour_areas': [],
                    'largest_contour_area': 0,
                    'bounding_boxes': []
                }
            
            # Process contours
            contour_data = []
            areas = []
            bounding_boxes = []
            
            for contour in contours:
                # Calculate area
                area = cv2.contourArea(contour)
                areas.append(area)
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                bounding_boxes.append({'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)})
                
                # Convert contour to list of points
                contour_points = contour.reshape(-1, 2).tolist()
                contour_data.append({
                    'points': contour_points,
                    'area': float(area),
                    'bounding_box': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
                })
            
            # Sort contours by area (largest first)
            sorted_indices = np.argsort(areas)[::-1]
            contour_data = [contour_data[i] for i in sorted_indices]
            areas = [areas[i] for i in sorted_indices]
            bounding_boxes = [bounding_boxes[i] for i in sorted_indices]
            
            return {
                'contours': contour_data,
                'num_contours': len(contours),
                'contour_areas': areas,
                'largest_contour_area': float(max(areas)) if areas else 0.0,
                'bounding_boxes': bounding_boxes
            }
            
        except Exception as e:
            logger.warning(f"Contour extraction failed: {e}")
            return {
                'contours': [],
                'num_contours': 0,
                'contour_areas': [],
                'largest_contour_area': 0,
                'bounding_boxes': []
            }
    
    def create_visualization_outputs(self, result: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Create visualization outputs from postprocessing results.
        
        Args:
            result: Postprocessing result dictionary
            
        Returns:
            Dictionary containing visualization arrays
        """
        try:
            binary_mask = result['binary_mask']
            confidence_map = result.get('confidence_map')
            
            visualizations = {}
            
            # Binary mask visualization (0-255 grayscale)
            visualizations['binary_mask_vis'] = (binary_mask * 255).astype(np.uint8)
            
            # Confidence heatmap visualization
            if confidence_map is not None:
                confidence_vis = (confidence_map * 255).astype(np.uint8)
                visualizations['confidence_heatmap'] = cv2.applyColorMap(confidence_vis, cv2.COLORMAP_JET)
            
            # Contour visualization
            if 'contours' in result and result['contours']:
                contour_vis = np.zeros_like(binary_mask, dtype=np.uint8)
                for contour_data in result['contours']:
                    points = np.array(contour_data['points'], dtype=np.int32)
                    cv2.fillPoly(contour_vis, [points], 255)
                visualizations['contour_mask'] = contour_vis
            
            # Combined overlay (binary mask + contours)
            if 'contours' in result and result['contours']:
                combined = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 3), dtype=np.uint8)
                # Binary mask in blue channel
                combined[:, :, 2] = (binary_mask * 255).astype(np.uint8)
                # Contours in red channel
                for contour_data in result['contours']:
                    points = np.array(contour_data['points'], dtype=np.int32)
                    cv2.polylines(combined, [points], True, (255, 0, 0), 2)
                visualizations['combined_overlay'] = combined
            
            return visualizations
            
        except Exception as e:
            logger.warning(f"Visualization creation failed: {e}")
            return {}
    
    def get_postprocessing_info(self) -> Dict[str, Any]:
        """
        Get information about postprocessing configuration.
        
        Returns:
            Dictionary containing postprocessing settings
        """
        return {
            'default_threshold': self.threshold,
            'enable_confidence': self.enable_confidence,
            'supported_outputs': [
                'binary_mask',
                'confidence_scores',
                'confidence_map',
                'contours',
                'bounding_boxes',
                'visualization_outputs'
            ],
            'refinement_features': [
                'morphological_operations',
                'small_component_removal',
                'edge_confidence_analysis'
            ]
        }
    
    def batch_postprocess(self, model_outputs: torch.Tensor, **kwargs) -> List[Dict[str, Any]]:
        """
        Postprocess a batch of model outputs efficiently.
        
        Args:
            model_outputs: Batch of model outputs with shape (B, 1, H, W)
            **kwargs: Postprocessing parameters
            
        Returns:
            List of postprocessed results
        """
        results = []
        batch_size = model_outputs.shape[0]
        
        for i in range(batch_size):
            single_output = model_outputs[i:i+1]
            result = self.postprocess(single_output, **kwargs)
            results.append(result)
        
        return results 