import time
import numpy as np
import cv2
from typing import List, Dict, Any, Optional
from pathlib import Path
import torch
from segment_anything import SamPredictor

from .models import Point, SegmentationMask, SAMResult
from .model_manager import ModelManager
from src.config.sam_config import SAMConfig


class SAMService:
    """Main service for SAM segmentation operations"""
    
    def __init__(self, config: Optional[SAMConfig] = None):
        self.config = config or SAMConfig.from_env()
        self.model_manager = ModelManager(self.config.models_dir)
        self.predictor: Optional[SamPredictor] = None
        self._initialized = False
        
    async def initialize_model(self) -> bool:
        """Initialize the SAM model (download if needed, load into memory)"""
        try:
            # Check if model exists locally
            model_path = self.model_manager.get_model_path(self.config.model_type, self.config)
            print("model_path:", model_path)
            if not model_path:
                # Download model if not present
                model_path = await self.model_manager.download_model(self.config.model_type, self.config)
            
            # Load model
            self.predictor = self.model_manager.load_model(
                model_path, 
                self.config.model_type, 
                self.config.device
            )
            
            self._initialized = True
            print(f"SAM service initialized with {self.config.model_type} model")
            return True
            
        except Exception as e:
            print(f"Failed to initialize SAM service: {e}")
            self._initialized = False
            return False
    
    async def process_segmentation(self, image_path: str, input_box: List[int], points: List[Point], labels: List[int]) -> SAMResult:
        """Process segmentation request using SAM"""
        start_time = time.time()
        
        try:
            # Ensure model is initialized
            if not self._initialized:
                init_success = await self.initialize_model()
                if not init_success:
                    return SAMResult(
                        masks=[],
                        success=False,
                        error_message="Failed to initialize SAM model"
                    )
            
            # Validate input
            if len(points) != len(labels):
                return SAMResult(
                    masks=[],
                    success=False,
                    error_message="Points and labels must have the same length"
                )
            
            if len(points) > self.config.max_points:
                return SAMResult(
                    masks=[],
                    success=False,
                    error_message=f"Too many points (max: {self.config.max_points})"
                )
            
            # Load and preprocess image
            image = self._load_image(image_path)
            if image is None:
                return SAMResult(
                    masks=[],
                    success=False,
                    error_message="Failed to load image"
                )
            
            # Set image for SAM predictor
            self.predictor.set_image(image)
            
            # Convert points to numpy arrays
            point_coords = np.array([[p.x, p.y] for p in points])
            point_labels = np.array(labels)
            np_input_box = np.array(input_box)
            # Run SAM prediction
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
                box = np_input_box
            )
            # Post-process masks
            segmentation_masks = self._postprocess_masks(masks, scores)
            
            # Filter by confidence threshold
            #filtered_masks = [
            #    mask for mask in segmentation_masks 
            #    if mask.confidence >= self.config.confidence_threshold
            #]
            # best_mask = masks[np.argmax(scores)]
            processing_time = time.time() - start_time
            
            return SAMResult(
                masks=segmentation_masks,
                success=True,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return SAMResult(
                masks=[],
                success=False,
                error_message=f"Segmentation failed: {str(e)}",
                processing_time=processing_time
            )
    
    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load and preprocess image for SAM"""
        try:
            # Handle both absolute and relative paths
            if not Path(image_path).is_absolute():
                # Find the project root (nail-segmentation directory)
                current_path = Path(__file__).resolve()
                project_root = None
                for parent in current_path.parents:
                    if parent.name == "nail-segmentation":
                        project_root = parent
                        break
                
                if project_root is None:
                    print(f"Could not find project root directory")
                    return None
                
                # Construct path to frontend public directory
                frontend_public = project_root / "frontend" / "public"
                full_path = frontend_public / image_path.lstrip('/')
            else:
                full_path = Path(image_path)
            
            print(f"Loading image from: {full_path}")
            
            if not full_path.exists():
                print(f"Image not found: {full_path}")
                return None
            
            # Load image using OpenCV
            image = cv2.imread(str(full_path))
            if image is None:
                print(f"Failed to load image: {full_path}")
                return None
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            print(f"Successfully loaded image: {image.shape}")
            return image
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for SAM (currently just returns the image)"""
        # SAM handles preprocessing internally, but we could add custom preprocessing here
        return image
    
    def _postprocess_masks(self, masks: np.ndarray, scores: np.ndarray) -> List[SegmentationMask]:
        """Convert SAM masks to SegmentationMask objects"""
        segmentation_masks = []
        best_mask = masks[np.argmax(scores)]
        best_score = scores[np.argmax(scores)]
        # for i, (mask, score) in enumerate(zip(masks, scores)):
        #    try:
        # Convert boolean mask to contours
        mask_uint8 = (best_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return segmentation_masks
                
        # Use the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
                
        # Simplify contour to reduce point count
        epsilon = 0.0005 * cv2.arcLength(largest_contour, True)
        simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                
        # Convert contour to points
        points = [Point(x=float(point[0][0]), y=float(point[0][1])) for point in simplified_contour]
                
        if len(points) < 3:  # Need at least 3 points for a polygon
            return segmentation_masks
                
        # Calculate area
        area = int(cv2.contourArea(largest_contour))
        mask_image = best_mask.tolist()

        segmentation_mask = SegmentationMask(
            points=points,
            mask_image=mask_image,
            confidence= best_score, #float(scores[np.argmax(scores)]),
            area=area
        )
                
        segmentation_masks.append(segmentation_mask)
                
        #    except Exception as e:
        #        print(f"Error processing mask {i}: {e}")
        #        continue
        
        return segmentation_masks
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current SAM model"""
        info = {
            "initialized": self._initialized,
            "model_type": self.config.model_type,
            "device": self.config.device,
            "confidence_threshold": self.config.confidence_threshold,
            "max_points": self.config.max_points
        }
        
        if self._initialized:
            info.update(self.model_manager.get_loaded_model_info())
        
        return info
    
    async def health_check(self) -> Dict[str, Any]:
        """Check SAM service health"""
        health_info = {
            "status": "healthy" if self._initialized else "unhealthy",
            "model_loaded": self._initialized,
            "model_type": self.config.model_type,
            "device": self.config.device
        }
        
        if not self._initialized:
            health_info["status"] = "loading"
            health_info["message"] = "Model not initialized"
        
        # Add GPU info if available
        if torch.cuda.is_available():
            health_info["gpu_available"] = True
            health_info["gpu_memory_allocated"] = torch.cuda.memory_allocated()
            health_info["gpu_memory_cached"] = torch.cuda.memory_reserved()
        else:
            health_info["gpu_available"] = False
        
        return health_info 