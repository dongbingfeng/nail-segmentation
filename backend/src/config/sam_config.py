from dataclasses import dataclass, field
from typing import Dict
import os


@dataclass
class SAMConfig:
    """Configuration settings for SAM integration"""
    model_type: str = "vit_h"
    device: str = "auto"
    models_dir: str = "./models"
    max_points: int = 10
    confidence_threshold: float = 0.5
    enable_postprocessing: bool = True
    checkpoint_urls: Dict[str, str] = field(default_factory=lambda: {
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth", 
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    })
    
    @classmethod
    def from_env(cls) -> 'SAMConfig':
        """Create configuration from environment variables"""
        return cls(
            model_type=os.getenv("SAM_MODEL_TYPE", "vit_h"),
            device=os.getenv("SAM_DEVICE", "auto"),
            models_dir=os.getenv("SAM_MODELS_DIR", "./models"),
            max_points=int(os.getenv("SAM_MAX_POINTS", "10")),
            confidence_threshold=float(os.getenv("SAM_CONFIDENCE_THRESHOLD", "0.5")),
            enable_postprocessing=os.getenv("SAM_ENABLE_POSTPROCESSING", "true").lower() == "true"
        )
    
    def get_checkpoint_path(self) -> str:
        """Get the local path for the model checkpoint"""
        filename = {
            "vit_b": "sam_vit_b_01ec64.pth",
            "vit_l": "sam_vit_l_0b3195.pth",
            "vit_h": "sam_vit_h_4b8939.pth"
        }[self.model_type]
        return os.path.join(self.models_dir, filename) 