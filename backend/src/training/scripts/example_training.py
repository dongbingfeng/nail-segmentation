#!/usr/bin/env python3
"""
Example training script for nail segmentation.
Demonstrates how to use the training pipeline with sample data.
"""

import os
import sys
import argparse
import tempfile
import shutil
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import yaml

# Add the training module to the path
training_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, training_root)

from data.dataset import NailSegmentationDataset
from models.unet import AttentionUNet
from training.trainer import NailSegmentationTrainer
from utils.config import TrainingConfig
from utils.logging import setup_logging
from utils.performance import log_system_info

def create_sample_dataset(output_dir: Path, num_samples: int = 20):
    """Create a sample dataset for demonstration."""
    print(f"Creating sample dataset with {num_samples} images...")
    
    images_dir = output_dir / "sample_data" / "images"
    masks_dir = output_dir / "sample_data" / "masks"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Create realistic-looking nail images and masks
    for i in range(num_samples):
        # Create a more realistic nail-like image
        # Background (skin tone)
        image = np.full((256, 256, 3), [220, 180, 140], dtype=np.uint8)
        
        # Add some noise
        noise = np.random.randint(-20, 20, (256, 256, 3))
        image = np.clip(image.astype(int) + noise, 0, 255).astype(np.uint8)
        
        # Create nail region (oval shape in center)
        y, x = np.ogrid[:256, :256]
        center_y, center_x = 128, 128
        
        # Create oval mask for nail
        nail_mask = ((x - center_x) / 60) ** 2 + ((y - center_y) / 80) ** 2 <= 1
        
        # Make nail region slightly different color (more pink/white)
        nail_color = [240, 200, 200]
        for c in range(3):
            image[nail_mask, c] = nail_color[c]
        
        # Add some variation to nail color
        nail_noise = np.random.randint(-15, 15, nail_mask.sum())
        for c in range(3):
            image[nail_mask, c] = np.clip(
                image[nail_mask, c].astype(int) + nail_noise, 0, 255
            ).astype(np.uint8)
        
        # Save image
        image_pil = Image.fromarray(image)
        image_pil.save(images_dir / f"nail_{i:03d}.jpg")
        
        # Create corresponding binary mask
        mask = (nail_mask * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask, mode='L')
        mask_pil.save(masks_dir / f"nail_{i:03d}.png")
    
    print(f"✓ Sample dataset created in {output_dir / 'sample_data'}")
    return output_dir / "sample_data"

def run_example_training(data_dir: Path, output_dir: Path, config_path: Path):
    """Run an example training session."""
    print("\n" + "="*60)
    print("RUNNING EXAMPLE TRAINING")
    print("="*60)
    
    # Load configuration
    config = TrainingConfig.from_yaml(str(config_path))
    
    # Modify config for demonstration (shorter training)
    config.training.epochs = 5
    config.training.batch_size = 4
    config.training.validation_split = 0.2
    config.training.save_every = 2
    config.training.log_every = 5
    
    print(f"Configuration loaded from {config_path}")
    print(f"Training for {config.training.epochs} epochs")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Image size: {config.image_size[0]}x{config.image_size[1]}")
    
    # Setup logging
    log_file = output_dir / "training.log"
    setup_logging(str(log_file))
    
    # Log system information
    log_system_info()
    
    # Create trainer
    print(f"\nInitializing trainer...")
    trainer = NailSegmentationTrainer(config, str(output_dir))
    
    # Prepare data
    print(f"Preparing data from {data_dir}...")
    trainer.prepare_data(str(data_dir))
    
    print(f"Dataset prepared:")
    print(f"  Training samples: {len(trainer.train_dataset)}")
    print(f"  Validation samples: {len(trainer.val_dataset)}")
    
    # Start training
    print(f"\nStarting training...")
    print("Expected output:")
    print("  - Training progress with loss values")
    print("  - Validation metrics every epoch")
    print("  - Model checkpoints saved periodically")
    print("  - TensorBoard logs for visualization")
    
    try:
        trainer.train()
        print(f"\n✅ Training completed successfully!")
        
        # Show results
        checkpoint_dir = output_dir / "checkpoints"
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("*.pth"))
            print(f"✓ {len(checkpoints)} checkpoints saved")
            
            # Show best model info
            best_model_path = checkpoint_dir / "best_model.pth"
            if best_model_path.exists():
                checkpoint = torch.load(best_model_path, map_location='cpu')
                print(f"✓ Best model saved with validation IoU: {checkpoint.get('best_val_iou', 'N/A'):.4f}")
        
        # Show log file
        if log_file.exists():
            print(f"✓ Training log saved to: {log_file}")
        
        # Show TensorBoard logs
        tensorboard_dir = output_dir / "tensorboard"
        if tensorboard_dir.exists():
            print(f"✓ TensorBoard logs saved to: {tensorboard_dir}")
            print(f"  View with: tensorboard --logdir {tensorboard_dir}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demonstrate_model_usage(output_dir: Path, data_dir: Path):
    """Demonstrate how to use the trained model."""
    print("\n" + "="*60)
    print("DEMONSTRATING MODEL USAGE")
    print("="*60)
    
    # Load the best model
    checkpoint_dir = output_dir / "checkpoints"
    best_model_path = checkpoint_dir / "best_model.pth"
    
    if not best_model_path.exists():
        print("❌ No trained model found. Please run training first.")
        return False
    
    try:
        # Load model
        print("Loading trained model...")
        checkpoint = torch.load(best_model_path, map_location='cpu')
        
        # Create model architecture
        model = AttentionUNet(
            in_channels=3,
            out_channels=1,
            base_channels=64,
            depth=4,
            attention=True,
            dropout=0.2,
            batch_norm=True
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✓ Model loaded successfully")
        print(f"  Training epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Validation IoU: {checkpoint.get('best_val_iou', 'N/A'):.4f}")
        
        # Test on a sample image
        print("\nTesting model on sample image...")
        
        # Load a test image
        images_dir = data_dir / "images"
        test_images = list(images_dir.glob("*.jpg"))
        
        if test_images:
            test_image_path = test_images[0]
            
            # Load and preprocess image
            from data.transforms import get_val_transforms
            transforms = get_val_transforms(256)
            
            image = Image.open(test_image_path).convert('RGB')
            original_size = image.size
            
            # Apply transforms
            transformed = transforms(image=np.array(image))
            input_tensor = transformed['image'].unsqueeze(0)
            
            # Run inference
            with torch.no_grad():
                output = model(input_tensor)
                prediction = torch.sigmoid(output)
                binary_mask = (prediction > 0.5).float()
            
            # Save results
            results_dir = output_dir / "inference_results"
            results_dir.mkdir(exist_ok=True)
            
            # Save original image
            shutil.copy(test_image_path, results_dir / "original.jpg")
            
            # Save prediction
            pred_np = prediction.squeeze().cpu().numpy()
            pred_image = Image.fromarray((pred_np * 255).astype(np.uint8), mode='L')
            pred_image = pred_image.resize(original_size)
            pred_image.save(results_dir / "prediction.png")
            
            # Save binary mask
            binary_np = binary_mask.squeeze().cpu().numpy()
            binary_image = Image.fromarray((binary_np * 255).astype(np.uint8), mode='L')
            binary_image = binary_image.resize(original_size)
            binary_image.save(results_dir / "binary_mask.png")
            
            print(f"✓ Inference results saved to: {results_dir}")
            print(f"  - original.jpg: Input image")
            print(f"  - prediction.png: Model prediction (grayscale)")
            print(f"  - binary_mask.png: Binary segmentation mask")
            
            return True
        else:
            print("❌ No test images found")
            return False
            
    except Exception as e:
        print(f"❌ Model usage demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to run the example training."""
    parser = argparse.ArgumentParser(description="Example nail segmentation training")
    parser.add_argument("--output-dir", type=str, default="./example_output",
                       help="Output directory for training results")
    parser.add_argument("--data-dir", type=str, default=None,
                       help="Data directory (if not provided, sample data will be created)")
    parser.add_argument("--config", type=str, default=None,
                       help="Config file path (defaults to config.yaml in scripts directory)")
    parser.add_argument("--create-sample-data", action="store_true",
                       help="Create sample dataset for demonstration")
    parser.add_argument("--num-samples", type=int, default=20,
                       help="Number of sample images to create")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip training and only demonstrate model usage")
    
    args = parser.parse_args()
    
    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = Path(args.config) if args.config else Path(__file__).parent / "config.yaml"
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return 1
    
    print("🚀 NAIL SEGMENTATION TRAINING EXAMPLE")
    print("="*60)
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Config file: {config_path.absolute()}")
    
    # Determine data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            print(f"❌ Data directory not found: {data_dir}")
            return 1
    else:
        # Create sample data
        print("\nNo data directory provided, creating sample dataset...")
        data_dir = create_sample_dataset(output_dir, args.num_samples)
    
    print(f"Data directory: {data_dir.absolute()}")
    
    # Run training
    if not args.skip_training:
        success = run_example_training(data_dir, output_dir, config_path)
        if not success:
            print("\n❌ Example training failed")
            return 1
    
    # Demonstrate model usage
    success = demonstrate_model_usage(output_dir, data_dir)
    if not success:
        print("\n❌ Model usage demonstration failed")
        return 1
    
    print("\n" + "="*60)
    print("🎉 EXAMPLE TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Results saved to: {output_dir.absolute()}")
    print("\nNext steps:")
    print("1. Check the training logs and TensorBoard visualizations")
    print("2. Examine the inference results")
    print("3. Try training with your own dataset")
    print("4. Experiment with different model configurations")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 