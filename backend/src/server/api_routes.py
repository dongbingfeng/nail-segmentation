from fastapi import APIRouter, HTTPException, status
from typing import List
from .image_service import image_service, ImageMetadata, Annotation, ProgressStatus

router = APIRouter()


@router.get("/images", response_model=List[ImageMetadata])
async def get_images():
    """Get list of all available images for labeling."""
    try:
        images = await image_service.get_image_list()
        return images
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve image list: {str(e)}"
        )


@router.get("/images/{image_id}/annotations", response_model=List[Annotation])
async def get_image_annotations(image_id: str):
    """Get annotations for a specific image."""
    try:
        annotations = await image_service.get_image_annotations(image_id)
        return annotations
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve annotations for image {image_id}: {str(e)}"
        )


@router.post("/images/{image_id}/annotations")
async def save_image_annotations(image_id: str, annotations: List[Annotation]):
    """Save annotations for a specific image."""
    try:
        success = await image_service.save_image_annotations(image_id, annotations)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save annotations for image {image_id}"
            )
        
        return {"message": "Annotations saved successfully", "imageId": image_id, "count": len(annotations)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save annotations for image {image_id}: {str(e)}"
        )


@router.get("/labeling/progress", response_model=ProgressStatus)
async def get_labeling_progress():
    """Get overall labeling progress status."""
    try:
        progress = await image_service.get_labeling_progress()
        return progress
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve labeling progress: {str(e)}"
        )


@router.post("/labeling/progress/{current_index}")
async def update_progress_index(current_index: int):
    """Update the current image index in progress tracking."""
    try:
        success = await image_service.update_progress_index(current_index)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update progress index"
            )
        
        return {"message": "Progress updated successfully", "currentIndex": current_index}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update progress index: {str(e)}"
        ) 