import os
import json
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel


class ImageMetadata(BaseModel):
    id: str
    filename: str
    url: str
    width: int
    height: int
    isCompleted: bool
    annotationCount: int
    lastModified: str


class Annotation(BaseModel):
    id: str
    imageId: str
    type: str
    coordinates: Dict
    label: str
    confidence: Optional[float] = None
    createdAt: str
    updatedAt: str


class ProgressStatus(BaseModel):
    totalImages: int
    completedImages: int
    currentImageIndex: int
    percentComplete: float


class ImageService:
    def __init__(self):
        self.images_dir = Path("../frontend/public/images")
        self.annotations_dir = Path("../backend/data/annotations")
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        
        # Supported image extensions
        self.supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    
    async def get_image_list(self) -> List[ImageMetadata]:
        """Scan filesystem and return list of available images with metadata."""
        images = []
        
        if not self.images_dir.exists():
            return images
        
        # Scan directory for image files
        for file_path in self.images_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                # Generate image ID from filename
                image_id = file_path.stem
                
                # Get file stats
                stat = file_path.stat()
                last_modified = datetime.fromtimestamp(stat.st_mtime).isoformat()
                
                # Check if annotations exist
                annotation_count = await self._get_annotation_count(image_id)
                is_completed = annotation_count > 0
                
                # For now, set default dimensions (will be updated by frontend)
                image_metadata = ImageMetadata(
                    id=image_id,
                    filename=file_path.name,
                    url=f"/images/{file_path.name}",
                    width=800,  # Default, will be updated by frontend
                    height=600,  # Default, will be updated by frontend
                    isCompleted=is_completed,
                    annotationCount=annotation_count,
                    lastModified=last_modified
                )
                
                images.append(image_metadata)
        
        # Sort by filename for consistent ordering
        images.sort(key=lambda x: x.filename)
        
        return images
    
    async def get_image_annotations(self, image_id: str) -> List[Annotation]:
        """Retrieve saved annotations for a specific image."""
        annotation_file = self.annotations_dir / f"{image_id}_annotations.json"
        
        if not annotation_file.exists():
            return []
        
        try:
            with open(annotation_file, 'r') as f:
                data = json.load(f)
            
            annotations = []
            for annotation_data in data.get('annotations', []):
                annotation = Annotation(**annotation_data)
                annotations.append(annotation)
            
            return annotations
        
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Error loading annotations for {image_id}: {e}")
            return []
    
    async def save_image_annotations(self, image_id: str, annotations: List[Annotation]) -> bool:
        """Save annotations for a specific image."""
        annotation_file = self.annotations_dir / f"{image_id}_annotations.json"
        
        try:
            # Convert annotations to dict format
            annotations_data = {
                "imageId": image_id,
                "annotations": [annotation.dict() for annotation in annotations],
                "savedAt": datetime.now().isoformat(),
                "count": len(annotations)
            }
            
            with open(annotation_file, 'w') as f:
                json.dump(annotations_data, f, indent=2)
            
            return True
        
        except Exception as e:
            print(f"Error saving annotations for {image_id}: {e}")
            return False
    
    async def get_labeling_progress(self) -> ProgressStatus:
        """Get overall labeling progress status."""
        images = await self.get_image_list()
        total_images = len(images)
        completed_images = sum(1 for img in images if img.isCompleted)
        
        # Load current progress from file if exists
        progress_file = self.annotations_dir / "progress.json"
        current_index = 0
        
        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                    current_index = progress_data.get('currentImageIndex', 0)
            except (json.JSONDecodeError, KeyError):
                current_index = 0
        
        # Ensure current index is within bounds
        current_index = min(current_index, max(0, total_images - 1))
        
        percent_complete = (completed_images / total_images * 100) if total_images > 0 else 0
        
        return ProgressStatus(
            totalImages=total_images,
            completedImages=completed_images,
            currentImageIndex=current_index,
            percentComplete=round(percent_complete, 1)
        )
    
    async def update_progress_index(self, current_index: int) -> bool:
        """Update the current image index in progress tracking."""
        progress_file = self.annotations_dir / "progress.json"
        
        try:
            progress_data = {
                "currentImageIndex": current_index,
                "updatedAt": datetime.now().isoformat()
            }
            
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
            
            return True
        
        except Exception as e:
            print(f"Error updating progress index: {e}")
            return False
    
    async def _get_annotation_count(self, image_id: str) -> int:
        """Get the number of annotations for a specific image."""
        annotation_file = self.annotations_dir / f"{image_id}_annotations.json"
        
        if not annotation_file.exists():
            return 0
        
        try:
            with open(annotation_file, 'r') as f:
                data = json.load(f)
            return data.get('count', 0)
        
        except (json.JSONDecodeError, KeyError):
            return 0


# Global service instance
image_service = ImageService() 