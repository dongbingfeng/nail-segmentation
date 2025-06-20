# Nail Segmentation Training Module

A comprehensive training system for nail segmentation using U-Net architecture with attention mechanisms.

## Overview

This module provides a complete training pipeline for nail segmentation models, featuring:

- **U-Net Architecture**: Attention-enhanced U-Net optimized for small datasets
- **Robust Training**: Mixed precision, gradient clipping, early stopping
- **Comprehensive Metrics**: IoU, Dice coefficient, pixel accuracy
- **Hardware Flexibility**: Automatic CPU/GPU detection and optimization
- **Extensive Monitoring**: TensorBoard integration, detailed logging
- **Error Handling**: Comprehensive error handling and recovery mechanisms

## Quick Start

### 1. Basic Training

```bash
# Train with default configuration
python scripts/train.py --data-dir /path/to/your/data --output-dir ./results

# Train with custom configuration
python scripts/train.py --config custom_config.yaml --data-dir /path/to/data
```

### 2. Example Training with Sample Data

```bash
# Run example training with automatically generated sample data
python scripts/example_training.py --output-dir ./example_results

# Create sample data only
python scripts/example_training.py --create-sample-data --num-samples 50
```

### 3. Test Pipeline

```bash
# Test end-to-end pipeline
python scripts/test_pipeline.py

# Test GPU/CPU compatibility
python scripts/test_compatibility.py
```

## Installation

### Dependencies

```bash
pip install torch>=2.0.0 torchvision>=0.15.0
pip install albumentations>=1.3.0 tensorboard>=2.12.0
pip install tqdm>=4.65.0 pyyaml>=6.0 scikit-learn>=1.3.0
pip install psutil  # For performance monitoring
```

### Hardware Requirements

- **Minimum**: CPU with 8GB RAM
- **Recommended**: GPU with 6GB+ VRAM
- **Storage**: 2GB for model checkpoints and logs

## Data Format

### Directory Structure

```
data/
├── images/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
└── masks/
    ├── image_001.png
    ├── image_002.png
    └── ...
```

### Requirements

- **Images**: RGB format (JPG, PNG)
- **Masks**: Grayscale binary masks (PNG)
- **Naming**: Corresponding images and masks must have the same filename
- **Size**: Any size (will be resized during training)

## Configuration

### Basic Configuration (`config.yaml`)

```yaml
# Model Configuration
model:
  input_channels: 3
  output_channels: 1
  base_channels: 64
  depth: 4
  attention: true
  dropout: 0.2
  batch_norm: true

# Training Configuration
training:
  epochs: 100
  batch_size: 8
  learning_rate: 0.0001
  validation_split: 0.2
  early_stopping_patience: 15
  gradient_clip_val: 1.0
  mixed_precision: true

# Data Configuration
data:
  image_size: [256,256]
  augmentation:
    rotation: 15
    horizontal_flip: 0.5
    brightness: 0.2
    contrast: 0.2

# Optimization
optimizer:
  type: "adam"
  weight_decay: 0.00001
  lr_scheduler: "cosine"

# Loss Function
loss:
  bce_weight: 1.0
  dice_weight: 1.0
  focal_alpha: 0.25
  focal_gamma: 2.0
```

### Advanced Configuration Options

#### Model Architecture

```yaml
model:
  base_channels: 32    # Smaller for limited data/memory
  depth: 3            # Reduce for faster training
  attention: false    # Disable for speed
  dropout: 0.3        # Increase for regularization
```

#### Training Optimization

```yaml
training:
  batch_size: 4       # Reduce for limited memory
  mixed_precision: false  # Disable if causing issues
  gradient_clip_val: 0.5  # Reduce for stability
  accumulate_grad_batches: 2  # Simulate larger batch size
```

#### Hardware-Specific Settings

```yaml
# For CPU training
device: "cpu"
training:
  batch_size: 2
  num_workers: 2
  mixed_precision: false

# For GPU training
device: "cuda"
training:
  batch_size: 16
  num_workers: 8
  mixed_precision: true
```

## Usage Examples

### 1. Standard Training

```python
from training.utils.config import TrainingConfig
from training.training.trainer import NailSegmentationTrainer

# Load configuration
config = TrainingConfig.from_yaml("config.yaml")

# Create trainer
trainer = NailSegmentationTrainer(config, output_dir="./results")

# Train model
trainer.train(data_dir="./data")
```

### 2. Custom Training Loop

```python
# Setup model and data
trainer.setup_model()
train_loader, val_loader = trainer.setup_data("./data")

# Custom training loop
for epoch in range(config.training.epochs):
    train_metrics = trainer.train_epoch(train_loader)
    val_metrics = trainer.validate_epoch(val_loader)
    
    print(f"Epoch {epoch}: Train IoU: {train_metrics['iou']:.4f}, "
          f"Val IoU: {val_metrics['iou']:.4f}")
```

### 3. Model Inference

```python
import torch
from training.models.unet import AttentionUNet
from training.data.transforms import get_val_transforms

# Load trained model
checkpoint = torch.load("results/checkpoints/best_model.pth")
model = AttentionUNet(**checkpoint['model_config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare image
transforms = get_val_transforms(256)
image_tensor = transforms(image=image_array)['image'].unsqueeze(0)

# Run inference
with torch.no_grad():
    output = model(image_tensor)
    prediction = torch.sigmoid(output)
    binary_mask = (prediction > 0.5).float()
```

## Monitoring and Visualization

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir results/tensorboard

# View at http://localhost:6006
```

### Available Metrics

- **Loss**: Training and validation loss curves
- **IoU**: Intersection over Union scores
- **Dice**: Dice coefficient scores
- **Learning Rate**: Learning rate schedule
- **Images**: Sample predictions and ground truth

### Log Files

- `training.log`: Detailed training logs
- `config.yaml`: Training configuration
- `metrics.json`: Final training metrics

## Performance Optimization

### Memory Optimization

```python
from training.utils.performance import MemoryOptimizer

# Clear GPU cache
MemoryOptimizer.clear_cache()

# Get memory info
memory_info = MemoryOptimizer.get_memory_info()

# Suggest optimal batch size
optimal_batch_size = MemoryOptimizer.suggest_batch_size(
    model, input_shape=(1, 3, 256, 256), device=device
)
```

### Training Speed

```python
from training.utils.performance import TrainingOptimizer

# Setup mixed precision
scaler = TrainingOptimizer.setup_mixed_precision()

# Compile model (PyTorch 2.0+)
model = TrainingOptimizer.setup_compile_optimization(model)

# Optimize for inference
model = TrainingOptimizer.optimize_model_for_inference(model)
```

## Troubleshooting

### Common Issues

#### 1. GPU Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Reduce batch size: `training.batch_size: 4`
- Reduce image size: `data.image_size: [128,128]`
- Enable gradient accumulation: `training.accumulate_grad_batches: 2`
- Disable mixed precision: `training.mixed_precision: false`

#### 2. Training Not Converging

**Symptoms**: Loss not decreasing, poor validation metrics

**Solutions**:
- Check data quality and labels
- Reduce learning rate: `training.learning_rate: 0.00001`
- Increase regularization: `model.dropout: 0.3`
- Add more data augmentation
- Reduce model complexity: `model.depth: 3`

#### 3. Slow Training

**Solutions**:
- Increase batch size if memory allows
- Use mixed precision: `training.mixed_precision: true`
- Optimize data loading: `training.num_workers: 8`
- Use GPU if available
- Enable model compilation (PyTorch 2.0+)

#### 4. Data Loading Errors

**Error**: `Dataset is empty` or `FileNotFoundError`

**Solutions**:
- Check data directory structure
- Verify image and mask file extensions
- Ensure matching filenames for images and masks
- Check file permissions

#### 5. Model Architecture Errors

**Error**: `RuntimeError: size mismatch`

**Solutions**:
- Check input/output channel configuration
- Verify image size compatibility
- Ensure model depth is appropriate for image size

### Debug Mode

```bash
# Enable debug logging
python scripts/train.py --log-level DEBUG

# Run with error handling test
python scripts/test_compatibility.py
```

### Performance Profiling

```python
from training.utils.performance import benchmark_training_step

# Benchmark training performance
results = benchmark_training_step(
    model, dataloader, device, num_steps=10
)
print(f"Steps per second: {results['steps_per_second']:.2f}")
```

## API Reference

### Core Classes

#### `NailSegmentationTrainer`

Main training class with comprehensive training loop.

**Methods**:
- `train(data_dir)`: Complete training pipeline
- `setup_model()`: Initialize model, optimizer, loss
- `setup_data(data_dir)`: Create data loaders
- `train_epoch(train_loader)`: Single training epoch
- `validate_epoch(val_loader)`: Single validation epoch

#### `TrainingConfig`

Configuration management with validation.

**Methods**:
- `from_yaml(path)`: Load from YAML file
- `save(path)`: Save configuration
- `get_device()`: Get optimal device
- `validate()`: Validate configuration

#### `AttentionUNet`

U-Net model with attention mechanisms.

**Parameters**:
- `in_channels`: Input channels (default: 3)
- `out_channels`: Output channels (default: 1)
- `base_channels`: Base feature channels (default: 64)
- `depth`: Network depth (default: 4)
- `attention`: Enable attention gates (default: True)

### Utility Functions

#### Data Processing

```python
from training.data.transforms import get_train_transforms, get_val_transforms
from training.data.dataset import NailSegmentationDataset
```

#### Performance Monitoring

```python
from training.utils.performance import (
    PerformanceMonitor, MemoryOptimizer, TrainingOptimizer
)
```

#### Checkpointing

```python
from training.utils.checkpoint import CheckpointManager
```

## Contributing

### Development Setup

```bash
# Install development dependencies
pip install -e .[dev]

# Run tests
python -m pytest tests/

# Run linting
flake8 training/
black training/
```

### Adding New Features

1. Follow the existing code structure
2. Add comprehensive error handling
3. Include unit tests
4. Update documentation
5. Test on both CPU and GPU

## License

This training module is part of the nail segmentation project. See the main project LICENSE file for details.

## Support

For issues and questions:

1. Check this documentation
2. Run the compatibility tests
3. Check the troubleshooting section
4. Review the example scripts
5. Open an issue with detailed error logs 