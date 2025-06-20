"""
U-Net API Models and Schemas

This module defines pydantic models for U-Net API requests and responses,
providing type safety and automatic validation for the serving endpoints.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict
import base64
import numpy as np
from datetime import datetime


class UNetSegmentationRequest(BaseModel):
    """Request model for single image U-Net segmentation."""
    
    model_config = ConfigDict(protected_namespaces=())
    
    image_data: str = Field(
        ..., 
        description="Base64 encoded image data (PNG, JPEG, etc.)"
    )
    
    threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Sigmoid threshold for binary mask generation (0.0-1.0)"
    )
    
    return_confidence: Optional[bool] = Field(
        None,
        description="Whether to return confidence scores and heatmap"
    )
    
    return_contours: Optional[bool] = Field(
        None,
        description="Whether to extract and return contours"
    )
    
    refine_mask: Optional[bool] = Field(
        None,
        description="Whether to apply morphological mask refinement"
    )
    
    return_visualizations: Optional[bool] = Field(
        False,
        description="Whether to return visualization images"
    )
    
    @field_validator('image_data')
    @classmethod
    def validate_image_data(cls, v):
        """Validate base64 image data."""
        try:
            # Check if it's valid base64
            base64.b64decode(v)
            return v
        except Exception:
            raise ValueError("Invalid base64 image data")


class UNetBatchImageRequest(BaseModel):
    """Individual image in batch request."""
    image_data: str = Field(description="Base64 encoded image data")

class UNetBatchRequest(BaseModel):
    """Request model for batch U-Net segmentation."""
    
    model_config = ConfigDict(protected_namespaces=())
    
    images: List[UNetBatchImageRequest] = Field(
        ...,
        min_length=1,
        max_length=32,  # Configurable limit
        description="List of images for batch processing"
    )
    
    threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Sigmoid threshold for binary mask generation"
    )
    
    return_confidence: Optional[bool] = Field(
        None,
        description="Whether to return confidence scores"
    )
    
    return_contours: Optional[bool] = Field(
        None,
        description="Whether to extract contours"
    )
    
    refine_mask: Optional[bool] = Field(
        None,
        description="Whether to apply mask refinement"
    )
    
    return_visualizations: Optional[bool] = Field(
        False,
        description="Whether to return visualization images"
    )
    
    @field_validator('images')
    @classmethod
    def validate_images(cls, v):
        """Validate all image data in batch."""
        for i, img_obj in enumerate(v):
            try:
                base64.b64decode(img_obj.image_data)
            except Exception:
                raise ValueError(f"Invalid base64 data for image {i}")
        return v


class ConfidenceScores(BaseModel):
    """Model for confidence score metrics."""
    
    overall_mean: float = Field(description="Mean confidence across entire image")
    overall_max: float = Field(description="Maximum confidence in image")
    overall_min: float = Field(description="Minimum confidence in image")
    mask_mean: float = Field(description="Mean confidence within predicted mask")
    mask_min: float = Field(description="Minimum confidence within mask")
    mask_max: float = Field(description="Maximum confidence within mask")
    background_mean: float = Field(description="Mean confidence in background")
    background_max: float = Field(description="Maximum confidence in background")
    mean_certainty: float = Field(description="Mean distance from threshold")


class ContourData(BaseModel):
    """Model for individual contour information."""
    
    points: List[List[int]] = Field(description="Contour points as [x, y] coordinates")
    area: float = Field(description="Contour area in pixels")
    bounding_box: Dict[str, int] = Field(description="Bounding box: x, y, width, height")


class BoundingBox(BaseModel):
    """Model for bounding box information."""
    
    x: int = Field(description="Left coordinate")
    y: int = Field(description="Top coordinate") 
    width: int = Field(description="Box width")
    height: int = Field(description="Box height")


class VisualizationOutputs(BaseModel):
    """Model for visualization outputs."""
    
    binary_mask_vis: Optional[str] = Field(
        None,
        description="Base64 encoded binary mask visualization"
    )
    confidence_heatmap: Optional[str] = Field(
        None,
        description="Base64 encoded confidence heatmap"
    )
    contour_mask: Optional[str] = Field(
        None,
        description="Base64 encoded contour visualization"
    )
    combined_overlay: Optional[str] = Field(
        None,
        description="Base64 encoded combined overlay visualization"
    )


class UNetSegmentationResponse(BaseModel):
    """Response model for U-Net segmentation results."""
    
    model_config = ConfigDict(protected_namespaces=())
    
    success: bool = Field(description="Whether segmentation was successful")
    request_id: Optional[str] = Field(description="Unique request identifier")
    processing_time_ms: Optional[float] = Field(description="Processing time in milliseconds")
    
    # Core segmentation results  
    mask_data: Optional[str] = Field(
        None,
        description="Base64 encoded binary mask (PNG format)"
    )
    binary_mask: Optional[str] = Field(
        None,
        description="Base64 encoded binary mask (PNG format)"
    )
    mask_area: Optional[int] = Field(description="Number of mask pixels")
    mask_area_ratio: Optional[float] = Field(description="Ratio of mask to total pixels")
    threshold_used: Optional[float] = Field(description="Threshold used for binary mask")
    model_info: Optional[Dict[str, Any]] = Field(description="Model metadata")
    
    # Confidence information (optional)
    confidence_scores: Optional[ConfidenceScores] = Field(
        None,
        description="Detailed confidence metrics"
    )
    confidence_map: Optional[str] = Field(
        None,
        description="Base64 encoded confidence heatmap"
    )
    
    # Contour information (optional)
    contours: Optional[List[ContourData]] = Field(
        None,
        description="Extracted contours with metadata"
    )
    largest_contour_area: Optional[float] = Field(description="Area of largest contour")
    bounding_boxes: Optional[List[BoundingBox]] = Field(
        None,
        description="Bounding boxes for contours"
    )
    
    # Visualization outputs (optional)
    visualizations: Optional[VisualizationOutputs] = Field(
        None,
        description="Visualization images"
    )
    
    # Error information
    error: Optional[Dict[str, Any]] = Field(
        None,
        description="Error details if processing failed"
    )


class UNetBatchResponse(BaseModel):
    """Response model for batch U-Net segmentation."""
    
    success: bool = Field(description="Whether batch processing was successful")
    results: List[UNetSegmentationResponse] = Field(description="Individual segmentation results")
    total_processing_time_ms: int = Field(description="Total processing time")
    batch_size: int = Field(description="Number of images processed")
    request_id: str = Field(description="Request identifier")
    
    # Batch-level error information
    error_message: Optional[str] = Field(None, description="Batch-level error message")


class ModelInfo(BaseModel):
    """Model information details."""
    
    model_config = ConfigDict(protected_namespaces=())
    
    model_name: str = Field(description="Model identifier")
    model_variant: str = Field(description="Model variant (standard/lightweight/deep)")
    model_loaded: bool = Field(description="Whether model is loaded and ready")
    device: Optional[str] = Field(description="Device being used (cuda/cpu)")
    
    # Model specifications
    total_parameters: Optional[int] = Field(description="Total model parameters")
    trainable_parameters: Optional[int] = Field(description="Trainable parameters")
    model_size_mb: Optional[float] = Field(description="Model size in megabytes")
    
    # File information
    checkpoint_dir: Optional[str] = Field(description="Checkpoint directory")
    current_model_path: Optional[str] = Field(description="Current model file path")
    model_file_size_mb: Optional[float] = Field(description="Model file size in MB")
    
    # Configuration
    preprocessing: Optional[Dict[str, Any]] = Field(description="Preprocessing config")
    inference: Optional[Dict[str, Any]] = Field(description="Inference config")


class MemoryPoolStatus(BaseModel):
    """Memory pool status information."""
    
    pool_name: str = Field(description="Memory pool identifier")
    allocated_tensors: int = Field(description="Number of allocated tensors")
    available_tensors: int = Field(description="Number of available tensors")
    total_capacity: int = Field(description="Total pool capacity")
    memory_usage_mb: float = Field(description="Memory usage in megabytes")
    utilization_ratio: float = Field(description="Pool utilization ratio")


class UNetHealthResponse(BaseModel):
    """Response model for U-Net service health check."""
    
    model_config = ConfigDict(protected_namespaces=())
    
    service_healthy: bool = Field(description="Overall service health status")
    model_loaded: bool = Field(description="Whether model is loaded")
    memory_pools_ready: bool = Field(description="Memory pools status")
    gpu_available: bool = Field(description="GPU availability status")
    last_health_check: datetime = Field(description="Timestamp of last health check")
    
    # Optional detailed information
    model_info: Optional[Dict[str, Any]] = Field(None, description="Model metadata")
    memory_usage: Optional[Dict[str, Any]] = Field(None, description="Memory usage info")
    capabilities: List[str] = Field(default_factory=list, description="Service capabilities")
    
    # Error information
    error_message: Optional[str] = Field(None, description="Error message if unhealthy")


class UNetInitializeRequest(BaseModel):
    """Request model for manual U-Net service initialization."""
    
    force_reload: Optional[bool] = Field(
        False,
        description="Force reload even if already initialized"
    )
    
    warm_up: Optional[bool] = Field(
        True,
        description="Perform warm-up inference after loading"
    )
    
    config_path: Optional[str] = Field(
        None,
        description="Optional path to custom configuration file"
    )


class UNetInitializeResponse(BaseModel):
    """Response model for U-Net service initialization."""
    
    success: bool = Field(description="Whether initialization was successful")
    message: str = Field(description="Initialization status message")
    initialization_time_ms: int = Field(description="Time taken for initialization")
    service_ready: bool = Field(description="Whether service is ready")
    request_id: str = Field(description="Request identifier")
    
    # Error information
    error_details: Optional[str] = Field(None, description="Error details if failed")


class UNetModelInfoResponse(BaseModel):
    """Response model for U-Net model information."""
    
    model_config = ConfigDict(protected_namespaces=())
    
    model_name: str = Field(description="Model identifier")
    model_variant: str = Field(description="Model variant")
    checkpoint_path: str = Field(description="Path to model checkpoint")
    model_parameters: int = Field(description="Total model parameters")
    input_shape: List[int] = Field(description="Model input shape")
    output_shape: List[int] = Field(description="Model output shape")
    device: str = Field(description="Device being used")
    
    # Optional detailed information
    architecture_details: Optional[Dict[str, Any]] = Field(None, description="Architecture details")
    training_metadata: Optional[Dict[str, Any]] = Field(None, description="Training metadata")
    preprocessing_config: Optional[Dict[str, Any]] = Field(None, description="Preprocessing config")
    supported_formats: List[str] = Field(default_factory=lambda: ["JPEG", "PNG", "BMP"])


# Utility functions for model conversion

def numpy_to_base64_png(array: np.ndarray) -> str:
    """
    Convert numpy array to base64 encoded PNG.
    
    Args:
        array: Numpy array (grayscale or RGB)
        
    Returns:
        Base64 encoded PNG data
    """
    from PIL import Image
    import io
    
    # Ensure proper data type
    if array.dtype != np.uint8:
        array = (array * 255).astype(np.uint8)
    
    # Create PIL image
    if array.ndim == 2:
        image = Image.fromarray(array, mode='L')
    elif array.ndim == 3:
        image = Image.fromarray(array, mode='RGB')
    else:
        raise ValueError(f"Unsupported array dimensions: {array.ndim}")
    
    # Convert to base64
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def base64_to_numpy(base64_data: str) -> np.ndarray:
    """
    Convert base64 encoded image to numpy array.
    
    Args:
        base64_data: Base64 encoded image data
        
    Returns:
        Numpy array in RGB format
    """
    from PIL import Image
    import io
    
    # Decode base64
    image_data = base64.b64decode(base64_data)
    
    # Load as PIL image
    image = Image.open(io.BytesIO(image_data))
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array
    return np.array(image) 