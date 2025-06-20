from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any
import asyncio

from src.sam.sam_service import SAMService
from src.sam.models import Point, SAMResult
from src.config.sam_config import SAMConfig

router = APIRouter()

# Global SAM service instance
sam_service: SAMService = None

async def get_sam_service() -> SAMService:
    """Get or create SAM service instance"""
    global sam_service
    if sam_service is None:
        config = SAMConfig.from_env()
        sam_service = SAMService(config)
    return sam_service

@router.post("/api/sam/segment")
async def process_sam_segmentation(request: dict):
    """Process SAM segmentation request
    
    Expected request format:
    {
        "imageId": "string",
        "boundingBox": {
            "topLeft": {"x": float, "y": float},
            "bottomRight": {"x": float, "y": float}
        },
        "points": [{"x": float, "y": float}],
        "labels": [1, 0]  // 1 for positive, 0 for negative
    }
    """
    try:
        # Validate request structure
        if "imageId" not in request:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing required field: imageId"
            )
        
        # Handle both old input_box format and new boundingBox format
        if "boundingBox" in request:
            # New format from frontend
            bbox = request["boundingBox"]
            input_box = [
                bbox["topLeft"]["x"], 
                bbox["topLeft"]["y"], 
                bbox["bottomRight"]["x"], 
                bbox["bottomRight"]["y"]
            ]
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing required field: boundingBox"
            )
        
        print(f"SAM Request: {request}")
        
        # Extract and validate data
        image_id = request["imageId"]

        if "points" not in request or not request["points"]:
            # Use center of bounding box if no points provided
            points_data = [{"x": (input_box[0] + input_box[2]) // 2, "y": (input_box[1] + input_box[3]) // 2}]
        else:
            points_data = request["points"]

        if "labels" not in request or not request["labels"]:
            labels = [1] * len(points_data)
        else:
            labels = request["labels"]
        
        # Convert to Point objects
        try:
            points = [Point(x=p["x"], y=p["y"]) for p in points_data] if len(points_data) > 0 else []
        except (KeyError, TypeError) as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid points format: {str(e)}"
            )
        
        # Validate labels
        if not all(label in [0, 1] for label in labels):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Labels must be 0 (negative) or 1 (positive)"
            )
        if len(points) != len(labels):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Points and labels must have the same length"
            )
        
        # Get SAM service
        service = await get_sam_service()
        
        # Construct image path (assuming images are in frontend/public/images/)
        image_path = f"images/{image_id}.bmp"  # Adjust extension as needed
        
        # Process segmentation
        result = await service.process_segmentation(image_path, input_box, points, labels)
        
        # Convert result to response format matching frontend SAMResponse interface
        if result.success:
            masks_response = []
            mask_points = []
            
            for mask in result.masks:
                # Extract all True points from the mask image 
                if hasattr(mask, 'mask_image') and mask.mask_image:
                    for y in range(len(mask.mask_image)):
                        for x in range(len(mask.mask_image[y])):
                            if mask.mask_image[y][x]:
                                mask_points.append({"x": int(x), "y": int(y)})
                
                mask_data = {
                    "points": [{"x": float(p.x), "y": float(p.y)} for p in mask.points],
                    "confidence": float(mask.confidence)
                }
                masks_response.append(mask_data)
            
            print(f"Generated {len(mask_points)} mask points")
            
            response = {
                "masks": masks_response,
                "mask_points": mask_points,
                "success": True,
                "processing_time": float(result.processing_time)
            }
        else:
            response = {
                "masks": [],
                "mask_points": [],
                "success": False,
                "error": result.error_message
            }
        
        return response
        
    except HTTPException:
        raise
    #except Exception as e:
    #    raise HTTPException(
    #        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    #        detail=f"Internal server error: {str(e)}"
    #    )


@router.get("/api/sam/health")
async def sam_health_check() -> Dict[str, Any]:
    """Check SAM service health status"""
    try:
        service = await get_sam_service()
        health_info = await service.health_check()
        return health_info
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "model_loaded": False
        }


@router.get("/api/sam/model-info")
async def get_sam_model_info() -> Dict[str, Any]:
    """Get SAM model information"""
    try:
        service = await get_sam_service()
        model_info = service.get_model_info()
        return model_info
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )


@router.post("/api/sam/initialize")
async def initialize_sam_model():
    """Initialize SAM model (useful for warming up)"""
    try:
        service = await get_sam_service()
        success = await service.initialize_model()
        
        if success:
            return {
                "message": "SAM model initialized successfully",
                "success": True
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to initialize SAM model"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Initialization failed: {str(e)}"
        ) 