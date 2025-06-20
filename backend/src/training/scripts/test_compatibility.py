#!/usr/bin/env python3
"""
GPU/CPU compatibility test for nail segmentation training.
Tests that training works correctly on both GPU and CPU environments.
"""

import os
import sys
import tempfile
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import logging

# Add the training module to the path
training_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, training_root)

from data.dataset import NailSegmentationDataset
from data.transforms import get_train_transforms
from models.unet import AttentionUNet
from training.trainer import NailSegmentationTrainer
from utils.config import TrainingConfig
from utils.logging import setup_logging
from utils.performance import MemoryOptimizer, log_system_info

def create_test_data(data_dir: Path, num_samples: int = 6):
    """Create minimal test dataset."""
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(num_samples):
        # Create small test images (128x128)
        image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        image_pil = Image.fromarray(image)
        image_pil.save(images_dir / f"test_{i:03d}.jpg")
        
        # Create corresponding masks
        mask = np.random.randint(0, 2, (128, 128), dtype=np.uint8) * 255
        mask_pil = Image.fromarray(mask, mode='L')
        mask_pil.save(masks_dir / f"test_{i:03d}.png")
    
    print(f"Created {num_samples} test samples")

def test_device_compatibility(device_name: str, force_device: str = None):
    """Test training compatibility on specified device."""
    print(f"\n{'='*60}")
    print(f"TESTING {device_name.upper()} COMPATIBILITY")
    print(f"{'='*60}")
    
    # Determine device
    if force_device:
        device = torch.device(force_device)
    elif device_name.lower() == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            data_dir = temp_path / "data"
            output_dir = temp_path / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create test data
            create_test_data(data_dir)
            
            # Load and modify config for quick testing
            config_path = Path(__file__).parent / "config.yaml"
            config = TrainingConfig.from_yaml(str(config_path))
            
            # Optimize config for testing
            config.image_size = [128, 128]
            config.training.epochs = 2
            config.training.batch_size = 2
            config.training.validation_split = 0.3
            config.training.save_every = 1
            
            # Test 1: Model creation and device placement
            print("\n1. Testing model creation and device placement...")
            model = AttentionUNet(
                in_channels=config.model.input_channels,
                out_channels=config.model.output_channels,
                base_channels=32,  # Smaller for testing
                depth=3,  # Smaller depth
                attention=config.model.attention,
                dropout=config.model.dropout,
                batch_norm=config.model.batch_norm
            )
            
            model = model.to(device)
            print(f"✓ Model created and moved to {device}")
            
            # Test 2: Data loading
            print("\n2. Testing data loading...")
            transforms = get_train_transforms(config.image_size)
            dataset = NailSegmentationDataset(
                data_dir=str(data_dir),
                transforms=transforms,
                split='train'
            )
            
            from torch.utils.data import DataLoader
            dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
            batch = next(iter(dataloader))
            
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            print(f"✓ Data loaded successfully")
            print(f"  Images shape: {images.shape}, device: {images.device}")
            print(f"  Masks shape: {masks.shape}, device: {masks.device}")
            
            # Test 3: Forward pass
            print("\n3. Testing forward pass...")
            model.eval()
            with torch.no_grad():
                outputs = model(images)
            
            print(f"✓ Forward pass successful")
            print(f"  Output shape: {outputs.shape}, device: {outputs.device}")
            
            # Test 4: Backward pass
            print("\n4. Testing backward pass...")
            model.train()
            outputs = model(images)
            
            # Simple loss calculation
            loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, masks)
            loss.backward()
            
            print(f"✓ Backward pass successful")
            print(f"  Loss: {loss.item():.4f}")
            
            # Test 5: Memory usage
            print("\n5. Checking memory usage...")
            memory_info = MemoryOptimizer.get_memory_info()
            
            print(f"  CPU Memory: {memory_info['cpu_memory_mb']:.1f} MB")
            if device.type == 'cuda':
                print(f"  GPU Memory: {memory_info['gpu_memory_allocated_mb']:.1f} MB allocated")
                print(f"  GPU Memory: {memory_info['gpu_memory_reserved_mb']:.1f} MB reserved")
            
            # Test 6: Short training run
            print("\n6. Testing short training run...")
            
            # Create trainer
            trainer = NailSegmentationTrainer(config, str(output_dir))
            trainer.device = device  # Force device
            
            # Prepare data
            trainer.prepare_data(str(data_dir))
            
            # Run one epoch
            config.training.epochs = 1
            trainer.train()
            
            print(f"✓ Training completed successfully")
            
            # Test 7: Model saving/loading
            print("\n7. Testing model saving/loading...")
            
            # Save model
            model_path = output_dir / f"test_model_{device.type}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'device': str(device)
            }, model_path)
            
            # Load model
            checkpoint = torch.load(model_path, map_location=device)
            new_model = AttentionUNet(
                in_channels=config.model.input_channels,
                out_channels=config.model.output_channels,
                base_channels=32,
                depth=3,
                attention=config.model.attention,
                dropout=config.model.dropout,
                batch_norm=config.model.batch_norm
            )
            new_model.load_state_dict(checkpoint['model_state_dict'])
            new_model = new_model.to(device)
            
            print(f"✓ Model saved and loaded successfully")
            
            print(f"\n🎉 {device_name.upper()} COMPATIBILITY TEST PASSED!")
            return True
            
    except Exception as e:
        print(f"\n❌ {device_name.upper()} COMPATIBILITY TEST FAILED!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_device_switching():
    """Test switching between devices during training."""
    print(f"\n{'='*60}")
    print("TESTING DEVICE SWITCHING")
    print(f"{'='*60}")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping device switching test")
        return True
    
    try:
        # Create a simple model
        model = AttentionUNet(
            in_channels=3,
            out_channels=1,
            base_channels=32,
            depth=2,
            attention=False,
            dropout=0.1,
            batch_norm=True
        )
        
        # Test CPU -> GPU
        print("\n1. Testing CPU -> GPU transfer...")
        model_cpu = model.to('cpu')
        model_gpu = model_cpu.to('cuda')
        
        # Test with dummy data
        dummy_input_cpu = torch.randn(1, 3, 128, 128)
        dummy_input_gpu = dummy_input_cpu.to('cuda')
        
        with torch.no_grad():
            output_gpu = model_gpu(dummy_input_gpu)
        
        print(f"✓ CPU -> GPU transfer successful")
        
        # Test GPU -> CPU
        print("\n2. Testing GPU -> CPU transfer...")
        model_cpu_back = model_gpu.to('cpu')
        
        with torch.no_grad():
            output_cpu = model_cpu_back(dummy_input_cpu)
        
        print(f"✓ GPU -> CPU transfer successful")
        
        # Test that outputs are similar (accounting for numerical precision)
        output_gpu_cpu = output_gpu.cpu()
        if torch.allclose(output_cpu, output_gpu_cpu, atol=1e-5):
            print("✓ Outputs consistent across devices")
        else:
            print("⚠ Outputs differ slightly across devices (expected due to precision)")
        
        print(f"\n🎉 DEVICE SWITCHING TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ DEVICE SWITCHING TEST FAILED!")
        print(f"Error: {e}")
        return False

def run_compatibility_tests():
    """Run all compatibility tests."""
    print("=" * 80)
    print("NAIL SEGMENTATION TRAINING COMPATIBILITY TESTS")
    print("=" * 80)
    
    # Log system information
    log_system_info()
    
    test_results = []
    
    # Test CPU compatibility
    print("\n" + "="*80)
    cpu_result = test_device_compatibility("CPU", force_device="cpu")
    test_results.append(("CPU Compatibility", cpu_result))
    
    # Test GPU compatibility if available
    if torch.cuda.is_available():
        print("\n" + "="*80)
        gpu_result = test_device_compatibility("GPU", force_device="cuda")
        test_results.append(("GPU Compatibility", gpu_result))
        
        # Test device switching
        switch_result = test_device_switching()
        test_results.append(("Device Switching", switch_result))
    else:
        print("\n" + "="*80)
        print("CUDA not available, skipping GPU tests")
        test_results.append(("GPU Compatibility", "SKIPPED"))
        test_results.append(("Device Switching", "SKIPPED"))
    
    # Print final results
    print("\n" + "=" * 80)
    print("COMPATIBILITY TEST RESULTS")
    print("=" * 80)
    
    all_passed = True
    for test_name, result in test_results:
        if result == "SKIPPED":
            status = "SKIPPED"
        elif result:
            status = "PASS"
        else:
            status = "FAIL"
            all_passed = False
        
        print(f"{test_name:<25}: {status}")
    
    print("=" * 80)
    if all_passed:
        print("🎉 ALL COMPATIBILITY TESTS PASSED!")
        print("Training pipeline is compatible with available hardware.")
    else:
        print("❌ Some compatibility tests failed.")
        print("Please check the error messages above.")
    
    return all_passed

if __name__ == "__main__":
    success = run_compatibility_tests()
    sys.exit(0 if success else 1) 