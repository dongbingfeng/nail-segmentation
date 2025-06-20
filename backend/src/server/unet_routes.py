from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Optional
import logging
import asyncio
import uuid
from datetime import datetime

from src.unet.models import (
    UNetSegmentationRequest,
    UNetSegmentationResponse,
    UNetBatchRequest,
    UNetBatchResponse,
    UNetHealthResponse,
    UNetModelInfoResponse,
    UNetInitializeRequest,
    UNetInitializeResponse
)
from src.unet.unet_service import UNetModelService
from src.config.unet_config import get_unet_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/unet", tags=["unet"])

# Global service instance
_unet_service: Optional[UNetModelService] = None

def get_unet_service() -> UNetModelService:
    """Dependency to get the U-Net service instance."""
    global _unet_service
    if _unet_service is None:
        raise HTTPException(
            status_code=503,
            detail="U-Net service not initialized. Please check service health."
        )
    return _unet_service

async def initialize_unet_service(config_path: Optional[str] = None):
    """Initialize the U-Net service during startup."""
    global _unet_service
    try:
        print(f"Initializing U-Net service with config path: {config_path}")
        config = get_unet_config(config_path)
        _unet_service = UNetModelService(config)
        logger.info("U-Net service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize U-Net service: {e}")
        _unet_service = None
        raise

@router.post("/segment", response_model=UNetSegmentationResponse)
async def segment_image(
    request: UNetSegmentationRequest,
    service: UNetModelService = Depends(get_unet_service)
):
    """
    Perform nail segmentation on a single image.
    
    Args:
        request: Segmentation request with base64 encoded image
        service: U-Net service instance
        
    Returns:
        Segmentation response with mask, confidence scores, and metadata
    """
    request_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    try:
        logger.info(f"Processing segmentation request {request_id}")
        
        # Perform segmentation
        result = await service.segment_async(
            image_data=request.image_data,
            threshold=request.threshold,
            return_confidence=request.return_confidence,
            return_contours=request.return_contours,
            return_visualizations=request.return_visualizations
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        mask_data = result["mask_data"]
        # Convert binary mask array to base64 string
        if mask_data is not None:
            import base64
            import io
            from PIL import Image
            import numpy as np
            
            # Convert numpy array to PIL Image
            mask_array = mask_data
            if isinstance(mask_array, np.ndarray):
                # Ensure array is in correct format (0-255, uint8)
                if mask_array.dtype != np.uint8:
                    mask_array = (mask_array * 255).astype(np.uint8)
                
                # Convert to PIL Image
                mask_image = Image.fromarray(mask_array)
                
                # Save to bytes buffer as PNG
                buffer = io.BytesIO()
                mask_image.save(buffer, format='PNG')
                buffer.seek(0)
                
                # Convert to base64 string
                mask_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Create response
        response = UNetSegmentationResponse(
            success=True,
            mask_data=mask_base64,
            mask_area=result.get("mask_area"),
            mask_area_ratio=result.get("mask_area_ratio"),
            threshold_used=result.get("threshold_used"),
            largest_contour_area=result.get("largest_contour_area"),
            confidence_scores=result.get("confidence_scores"),
            contours=result.get("contours"),
            visualizations=result.get("visualizations"),
            processing_time_ms=int(processing_time * 1000),
            model_info=result["model_info"],
            request_id=request_id
        )
        print("2. segment_image result::")
        logger.info(f"Segmentation completed for request {request_id} in {processing_time:.3f}s")
        return response
        
    except Exception as e:
        logger.error(f"Segmentation failed for request {request_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "SEGMENTATION_FAILED",
                "message": "Failed to process segmentation request",
                "details": str(e),
                "request_id": request_id
            }
        )

@router.post("/segment-batch", response_model=UNetBatchResponse)
async def segment_batch(
    request: UNetBatchRequest,
    service: UNetModelService = Depends(get_unet_service)
):
    """
    Perform nail segmentation on multiple images in batch.
    
    Args:
        request: Batch segmentation request with multiple images
        service: U-Net service instance
        
    Returns:
        Batch response with results for each image
    """
    request_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    try:
        logger.info(f"Processing batch segmentation request {request_id} with {len(request.images)} images")
        
        # Validate batch size
        config = get_unet_config()
        if len(request.images) > config.inference.batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size {len(request.images)} exceeds maximum {config.inference.batch_size}"
            )
        
        # Perform batch segmentation
        results = await service.segment_batch_async(
            images_data=[img.image_data for img in request.images],
            threshold=request.threshold,
            return_confidence=request.return_confidence,
            return_contours=request.return_contours,
            return_visualizations=request.return_visualizations
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Create individual responses
        segmentation_results = []
        for i, result in enumerate(results):
            seg_response = UNetSegmentationResponse(
                success=True,
                mask_data=result["mask_data"],
                confidence_scores=result.get("confidence_scores"),
                contours=result.get("contours"),
                visualizations=result.get("visualizations"),
                processing_time_ms=int(processing_time * 1000 / len(results)),
                model_info=result["model_info"],
                request_id=f"{request_id}_{i}"
            )
            segmentation_results.append(seg_response)
        
        # Create batch response
        response = UNetBatchResponse(
            success=True,
            results=segmentation_results,
            total_processing_time_ms=int(processing_time * 1000),
            batch_size=len(request.images),
            request_id=request_id
        )
        
        logger.info(f"Batch segmentation completed for request {request_id} in {processing_time:.3f}s")
        return response
        
    except Exception as e:
        logger.error(f"Batch segmentation failed for request {request_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "BATCH_SEGMENTATION_FAILED",
                "message": "Failed to process batch segmentation request",
                "details": str(e),
                "request_id": request_id
            }
        )

@router.get("/health", response_model=UNetHealthResponse)
async def get_health_status(service: UNetModelService = Depends(get_unet_service)):
    """
    Get U-Net service health status and capabilities.
    
    Returns:
        Health response with service status and model information
    """
    try:
        health_status = await service.get_health_status()
        
        return UNetHealthResponse(
            service_healthy=health_status["service_healthy"],
            model_loaded=health_status["model_loaded"],
            memory_pools_ready=health_status["memory_pools_ready"],
            gpu_available=health_status["gpu_available"],
            last_health_check=health_status["last_health_check"],
            model_info=health_status.get("model_info"),
            memory_usage=health_status.get("memory_usage"),
            capabilities=health_status.get("capabilities", [])
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return UNetHealthResponse(
            service_healthy=False,
            model_loaded=False,
            memory_pools_ready=False,
            gpu_available=False,
            last_health_check=datetime.now(),
            error_message=str(e)
        )

@router.get("/model-info", response_model=UNetModelInfoResponse)
async def get_model_info(service: UNetModelService = Depends(get_unet_service)):
    """
    Get detailed information about the loaded U-Net model.
    
    Returns:
        Model information including architecture, parameters, and capabilities
    """
    try:
        model_info = await service.get_model_info()
        
        return UNetModelInfoResponse(
            model_name=model_info["model_name"],
            model_variant=model_info["model_variant"],
            checkpoint_path=model_info["checkpoint_path"],
            model_parameters=model_info["model_parameters"],
            input_shape=model_info["input_shape"],
            output_shape=model_info["output_shape"],
            device=model_info["device"],
            architecture_details=model_info.get("architecture_details"),
            training_metadata=model_info.get("training_metadata"),
            preprocessing_config=model_info.get("preprocessing_config"),
            supported_formats=model_info.get("supported_formats", ["JPEG", "PNG", "BMP"])
        )
        
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "MODEL_INFO_FAILED",
                "message": "Failed to retrieve model information",
                "details": str(e)
            }
        )

@router.post("/initialize", response_model=UNetInitializeResponse)
async def initialize_service(
    request: UNetInitializeRequest,
    background_tasks: BackgroundTasks
):
    """
    Manually initialize or reinitialize the U-Net service.
    
    Args:
        request: Initialization request with optional configuration overrides
        background_tasks: FastAPI background tasks for async initialization
        
    Returns:
        Initialization response with status and timing information
    """
    request_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    try:
        logger.info(f"Manual initialization requested {request_id}")
        
        # Perform initialization
        if request.force_reload:
            logger.info("Force reload requested, reinitializing service")
            await initialize_unet_service()
        else:
            # Check if service is already healthy
            global _unet_service
            if _unet_service is not None:
                health = await _unet_service.get_health_status()
                if health["service_healthy"]:
                    logger.info("Service already healthy, skipping initialization")
                    return UNetInitializeResponse(
                        success=True,
                        message="Service already initialized and healthy",
                        initialization_time_ms=0,
                        service_ready=True,
                        request_id=request_id
                    )
            
            # Initialize if not healthy
            await initialize_unet_service()
        
        # Perform warm-up if requested
        if request.warm_up and _unet_service is not None:
            logger.info("Performing service warm-up")
            await _unet_service.warm_up()
        
        # Calculate initialization time
        initialization_time = (datetime.now() - start_time).total_seconds()
        
        return UNetInitializeResponse(
            success=True,
            message="U-Net service initialized successfully",
            initialization_time_ms=int(initialization_time * 1000),
            service_ready=True,
            request_id=request_id
        )
        
    except Exception as e:
        initialization_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Initialization failed for request {request_id}: {e}")
        
        return UNetInitializeResponse(
            success=False,
            message=f"Failed to initialize U-Net service: {str(e)}",
            initialization_time_ms=int(initialization_time * 1000),
            service_ready=False,
            error_details=str(e),
            request_id=request_id
        )

# Note: Exception handlers are handled at the app level in main.py
# These functions can be imported and used as app-level exception handlers if needed

async def unet_http_exception_handler(request, exc):
    """Custom error handler for U-Net specific HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail if isinstance(exc.detail, dict) else {
                "code": "HTTP_ERROR",
                "message": str(exc.detail),
                "details": None
            },
            "timestamp": datetime.now().isoformat()
        }
    )

async def unet_general_exception_handler(request, exc):
    """General error handler for unexpected exceptions in U-Net routes."""
    logger.error(f"Unexpected error in U-Net route: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "details": str(exc)
            },
            "timestamp": datetime.now().isoformat()
        }
    ) 