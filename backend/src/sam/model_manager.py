import os
import hashlib
import aiohttp
import asyncio
from pathlib import Path
from typing import Optional, Any, Dict
import torch
from segment_anything import sam_model_registry, SamPredictor

from src.config.sam_config import SAMConfig


class ModelManager:
    """Manages SAM model downloads, loading, and caching"""
    
    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._loaded_models: Dict[str, Any] = {}
        
        # Expected checksums for model verification
        self._checksums = {
            "sam_vit_b_01ec64.pth": "375195639b5b30b3c213b73bb80d7bc8",
            "sam_vit_l_0b3195.pth": "0b3195507c81ec31b7cbc8ba5be80a23", 
            "sam_vit_h_4b8939.pth": "4b8939a88964f0f4ff5f5b2642c598a6"
        }
    
    async def download_model(self, model_type: str, config: SAMConfig) -> str:
        """Download SAM model checkpoint if not already present"""
        if model_type not in config.checkpoint_urls:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        checkpoint_path = config.get_checkpoint_path()
        
        # Check if model already exists and is valid
        if os.path.exists(checkpoint_path):
            if self._verify_model_integrity(checkpoint_path):
                return checkpoint_path
            else:
                # Remove corrupted file
                print(f"Removing corrupted model file: {checkpoint_path}")
                os.remove(checkpoint_path)
        
        # Download model with retry logic
        url = config.checkpoint_urls[model_type]
        print(f"Downloading SAM {model_type} model from {url}...")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            total_size = int(response.headers.get('content-length', 0))
                            downloaded_size = 0
                            
                            with open(checkpoint_path, 'wb') as f:
                                async for chunk in response.content.iter_chunked(1024 * 1024):  # 1MB chunks
                                    f.write(chunk)
                                    downloaded_size += len(chunk)
                                    if total_size > 0:
                                        progress = (downloaded_size / total_size) * 100
                                        print(f"Download progress: {progress:.1f}% ({downloaded_size / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB)")
                            
                            # Verify the download completed
                            if total_size > 0 and downloaded_size != total_size:
                                raise Exception(f"Download incomplete: {downloaded_size}/{total_size} bytes")
                            
                            print(f"Download completed: {downloaded_size / (1024*1024):.1f}MB")
                            break
                        else:
                            raise Exception(f"Failed to download model: HTTP {response.status}")
                            
            except Exception as e:
                print(f"Download attempt {attempt + 1} failed: {e}")
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
                
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to download model after {max_retries} attempts: {e}")
                
                print(f"Retrying in 5 seconds...")
                await asyncio.sleep(5)
        
        # Verify downloaded model
        if not self._verify_model_integrity(checkpoint_path):
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            raise Exception("Downloaded model failed integrity check")
        
        print(f"Successfully downloaded and verified {model_type} model")
        return checkpoint_path
    
    def get_model_path(self, model_type: str, config: SAMConfig) -> Optional[str]:
        """Get path to model checkpoint if it exists locally"""
        checkpoint_path = config.get_checkpoint_path()
        if os.path.exists(checkpoint_path) and self._verify_model_integrity(checkpoint_path):
            return checkpoint_path
        return None
    
    def load_model(self, model_path: str, model_type: str, device: str) -> SamPredictor:
        """Load SAM model and return predictor"""
        # Check if model is already loaded
        cache_key = f"{model_type}_{device}"
        if cache_key in self._loaded_models:
            return self._loaded_models[cache_key]
        
        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            # Load SAM model
            sam = sam_model_registry[model_type](checkpoint=model_path)
            sam.to(device=device)
            
            # Create predictor
            predictor = SamPredictor(sam)
            
            # Cache the loaded model
            self._loaded_models[cache_key] = predictor
            
            print(f"Successfully loaded SAM {model_type} model on {device}")
            return predictor
            
        except Exception as e:
            raise Exception(f"Failed to load SAM model: {str(e)}")
    
    def _verify_model_integrity(self, model_path: str) -> bool:
        """Verify model file integrity using file size check and basic format validation"""
        try:
            if not os.path.exists(model_path):
                return False
            
            file_size = os.path.getsize(model_path)
            
            # Expected approximate sizes for SAM models (in bytes)
            expected_sizes = {
                "sam_vit_b_01ec64.pth": 358 * 1024 * 1024,  # ~358MB
                "sam_vit_l_0b3195.pth": 1249 * 1024 * 1024,  # ~1.25GB 
                "sam_vit_h_4b8939.pth": 2564 * 1024 * 1024   # ~2.56GB
            }
            
            filename = os.path.basename(model_path)
            if filename in expected_sizes:
                expected_size = expected_sizes[filename]
                # Allow 5% tolerance in file size
                min_size = int(expected_size * 0.95)
                max_size = int(expected_size * 1.05)
                
                if not (min_size <= file_size <= max_size):
                    print(f"Model file size check failed: {file_size} bytes, expected ~{expected_size} bytes")
                    return False
            else:
                # Fallback to general size check
                min_size = 100 * 1024 * 1024  # 100MB minimum
                max_size = 3 * 1024 * 1024 * 1024  # 3GB maximum
                
                if not (min_size <= file_size <= max_size):
                    print(f"Model file size out of expected range: {file_size} bytes")
                    return False
            
            # Basic file format check - PyTorch models start with specific bytes
            try:
                with open(model_path, 'rb') as f:
                    header = f.read(8)
                    # PyTorch model files are ZIP archives, check for ZIP signature
                    if not header.startswith(b'PK\x03\x04') and not header.startswith(b'PK\x05\x06') and not header.startswith(b'PK\x07\x08'):
                        print(f"Model file format check failed: invalid ZIP header")
                        return False
            except Exception as e:
                print(f"Error reading model file header: {e}")
                return False
            
            print(f"Model integrity check passed: {file_size / (1024*1024):.1f}MB")
            return True
            
        except Exception as e:
            print(f"Error during model integrity check: {e}")
            return False
    
    def get_loaded_model_info(self) -> Dict[str, Any]:
        """Get information about currently loaded models"""
        info = {
            "loaded_models": list(self._loaded_models.keys()),
            "model_count": len(self._loaded_models)
        }
        
        if torch.cuda.is_available():
            info["gpu_memory_allocated"] = torch.cuda.memory_allocated()
            info["gpu_memory_cached"] = torch.cuda.memory_reserved()
        
        return info
    
    def clear_cache(self):
        """Clear loaded model cache to free memory"""
        self._loaded_models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 