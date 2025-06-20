#!/usr/bin/env python3
"""Test SAM backend imports"""

import sys
import os

# Add backend src to path
backend_src = os.path.join(os.path.dirname(__file__), '..', 'src')
backend_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, backend_src)
sys.path.insert(0, backend_root)

def test_sam_imports():
    print("Testing SAM backend imports...")
    
    try:
        from sam.sam_service import SAMService
        from config.sam_config import SAMConfig
        from sam.models import Point, SegmentationMask, SAMResult
        print("✓ SAM imports successful")
        
        # Test config
        config = SAMConfig()
        print(f"✓ Default config: {config.model_type}, {config.device}")
        
        # Test service creation (without model loading)
        service = SAMService(config)
        print("✓ SAM service creation successful")
        
        print("All SAM backend components working!")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_sam_imports()
    sys.exit(0 if success else 1) 