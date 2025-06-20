from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime


@dataclass
class Point:
    """Represents a 2D point coordinate for SAM input"""
    x: float
    y: float


@dataclass  
class SegmentationMask:
    """Represents a segmentation mask result from SAM"""
    points: List[Point]
    mask_image: List[List[bool]]
    confidence: float
    area: int


@dataclass
class SAMResult:
    """Complete result from SAM segmentation processing"""
    masks: List[SegmentationMask]
    success: bool
    error_message: Optional[str] = None
    processing_time: float = 0.0
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat() 