#!/usr/bin/env python3
"""
End-to-End Pipeline Test Script

Tests the complete training pipeline with dummy data to ensure all components work together.
"""

import os
import sys
import tempfile
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import yaml
import logging

# Add the training module to the path
training_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, training_root)

from data.dataset import NailSegmentationDataset
from data.transforms import get_train_transforms, get_val_transforms
from models.unet import AttentionUNet
from training.trainer import NailSegmentationTrainer
from utils.config import TrainingConfig
from utils.logging import setup_logging

def create_dummy_data(data_dir: Path, num_samples: int = 10):
    """Create dummy training data for testing."""
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Create annotations structure
    annotations = {}
    
    # Create dummy images and corresponding annotations
    for i in range(num_samples):
        # Create dummy RGB image (256x256)
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        image_pil = Image.fromarray(image, mode='RGB')
        image_filename = f"image_{i:03d}.jpg"
        image_pil.save(images_dir / image_filename, quality=95)
        
        # Create annotation for this image
        # Generate some random points for segmentation
        num_points = 8
        center_x, center_y = 128, 128
        radius = 60
        
        points = []
        for j in range(num_points):
            angle = 2 * np.pi * j / num_points
            x = int(center_x + radius * np.cos(angle) + np.random.randint(-10, 10))
            y = int(center_y + radius * np.sin(angle) + np.random.randint(-10, 10))
            points.append({"x": x, "y": y})
        
        annotations[f"image_{i:03d}"] = {
            "path": f"images/{image_filename}",
            "annotations": [
                {
                    "type": "segmentation",
                    "points": points
                }
            ]
        }
    
    # Save annotations file
    annotations_file = data_dir / "annotations.json"
    with open(annotations_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"Created {num_samples} dummy samples in {data_dir}")
    return str(annotations_file)

def test_data_loading(data_dir: Path, annotations_file: str, config: TrainingConfig):
    """Test data loading and augmentation."""
    print("Testing data loading...")
    
    try:
        # Test dataset creation
        train_transforms = get_train_transforms(config)
        val_transforms = get_val_transforms(config)
        
        dataset = NailSegmentationDataset(
            data_dir=str(data_dir),
            annotations_file=annotations_file,
            transform=train_transforms,
            split='train',
            config=config
        )
        
        print(f"Dataset created successfully with {len(dataset)} samples")
        
        # Test data loading
        sample = dataset[0]
        image, mask = sample['image'], sample['mask']
        
        print(f"Sample loaded - Image shape: {image.shape}, Mask shape: {mask.shape}")
        print(f"Image dtype: {image.dtype}, Mask dtype: {mask.dtype}")
        print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"Mask range: [{mask.min():.3f}, {mask.max():.3f}]")
        
        # Test data loader
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        batch = next(iter(dataloader))
        
        print(f"Batch loaded - Images: {batch['image'].shape}, Masks: {batch['mask'].shape}")
        
        return True
        
    except Exception as e:
        print(f"Data loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_creation(config: TrainingConfig):
    """Test model creation and forward pass."""
    print("Testing model creation...")
    
    try:
        # Create model using the create_model function
        from models.unet import create_model
        model = create_model(config)
        
        print(f"Model created successfully")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, config.image_size[1], config.image_size[0])
        
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"Forward pass successful - Input: {dummy_input.shape}, Output: {output.shape}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"Model creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_loop(data_dir: Path, annotations_file: str, config: TrainingConfig, output_dir: Path):
    """Test the training loop with minimal epochs."""
    print("Testing training loop...")
    
    try:
        # Modify config for quick testing
        config.epochs = 2
        config.batch_size = 2
        config.validation_split = 0.3
        
        # Copy annotations file to expected location (since trainer expects it there)
        import shutil
        expected_annotations = data_dir / "annotations.json"
        if not expected_annotations.exists():
            shutil.copy2(annotations_file, expected_annotations)
        
        # Create trainer
        trainer = NailSegmentationTrainer(config, output_dir=str(output_dir))
        
        # Train using just the data directory
        results = trainer.train(str(data_dir))
        
        print("Training completed successfully")
        print(f"Results: {results}")
        
        # Check if checkpoints were saved
        checkpoint_dir = output_dir / "checkpoints"
        if checkpoint_dir.exists() and list(checkpoint_dir.glob("*.pth")):
            print("Checkpoints saved successfully")
            return True
        else:
            print("Warning: No checkpoints found")
            return False
            
    except Exception as e:
        print(f"Training loop test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_saving_loading(output_dir: Path, config: TrainingConfig):
    """Test model saving and loading."""
    print("Testing model saving and loading...")
    
    try:
        # Create and save a model
        from models.unet import create_model
        model = create_model(config)
        
        # Save model
        model_path = output_dir / "test_model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config.__dict__
        }, model_path)
        
        print(f"Model saved to {model_path}")
        
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        
        new_model = create_model(config)
        new_model.load_state_dict(checkpoint['model_state_dict'])
        
        print("Model loaded successfully")
        
        # Test that models produce same output
        dummy_input = torch.randn(1, 3, config.image_size[1], config.image_size[0])
        
        model.eval()
        new_model.eval()
        
        with torch.no_grad():
            output1 = model(dummy_input)
            output2 = new_model(dummy_input)
        
        if torch.allclose(output1, output2, atol=1e-6):
            print("Model saving/loading test passed")
            return True
        else:
            print("Model outputs don't match after loading")
            return False
            
    except Exception as e:
        print(f"Model saving/loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_end_to_end_test():
    """Run complete end-to-end pipeline test."""
    print("=" * 60)
    print("NAIL SEGMENTATION TRAINING PIPELINE TEST")
    print("=" * 60)
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        data_dir = temp_path / "data"
        output_dir = temp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        log_file = output_dir / "test.log"
        setup_logging(log_dir=str(log_file.parent), log_level="INFO")
        
        # Load configuration
        config_path = Path(__file__).parent / "config.yaml"
        config = TrainingConfig.from_yaml(str(config_path))
        
        # Optimize config for testing
        config.image_size = [128, 128]  # Smaller for faster testing
        config.epochs = 2
        config.batch_size = 2
        
        print(f"Configuration loaded from {config_path}")
        print(f"Test configuration: {config.image_size[0]}x{config.image_size[1]}, {config.epochs} epochs")
        
        # Test 1: Create dummy data
        print("\n1. Creating dummy dataset...")
        annotations_file = create_dummy_data(data_dir, num_samples=8)
        
        # Test 2: Data loading
        print("\n2. Testing data loading...")
        if not test_data_loading(data_dir, annotations_file, config):
            print("❌ Data loading test failed")
            return False
        print("✅ Data loading test passed")
        
        # Test 3: Model creation
        print("\n3. Testing model creation...")
        if not test_model_creation(config):
            print("❌ Model creation test failed")
            return False
        print("✅ Model creation test passed")
        
        # Test 4: Model saving/loading
        print("\n4. Testing model saving/loading...")
        if not test_model_saving_loading(output_dir, config):
            print("❌ Model saving/loading test failed")
            return False
        print("✅ Model saving/loading test passed")
        
        # Test 5: Training loop
        print("\n5. Testing training loop...")
        if not test_training_loop(data_dir, annotations_file, config, output_dir):
            print("❌ Training loop test failed")
            return False
        print("✅ Training loop test passed")
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! Training pipeline is working correctly.")
        print("=" * 60)
        
        return True

if __name__ == "__main__":
    success = run_end_to_end_test()
    sys.exit(0 if success else 1) 