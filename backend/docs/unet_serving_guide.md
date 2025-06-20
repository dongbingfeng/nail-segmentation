# U-Net Model Serving Guide

## Overview

This guide covers the deployment and usage of the U-Net nail segmentation serving system. The system provides high-performance real-time segmentation through REST API endpoints with comprehensive monitoring and error handling.

## Architecture

### Components

- **UNetModelService**: Core service managing model loading, inference, and memory pools
- **API Endpoints**: FastAPI routes for segmentation requests and health monitoring  
- **Memory Pool Manager**: Pre-allocated tensor pools for efficient inference
- **Health Check System**: Monitoring and validation of service components
- **Configuration System**: YAML-based configuration with environment variable overrides

### Performance Features

- **Eager Loading**: Models loaded at startup for consistent response times
- **Memory Pools**: Pre-allocated tensors eliminate allocation overhead
- **Batch Processing**: Efficient processing of multiple images
- **Device Management**: Automatic GPU/CPU fallback and optimization
- **Thread Safety**: Concurrent request handling with proper synchronization

## API Reference

### Base URL
```
http://localhost:8001/api/unet
```

### Endpoints

#### POST /segment
Perform single image segmentation.

**Request Body:**
```json
{
  "image_data": "base64_encoded_image_data",
  "threshold": 0.5,
  "return_confidence": false,
  "return_contours": false,
  "return_visualizations": false
}
```

**Response:**
```json
{
  "success": true,
  "mask_data": "base64_encoded_mask",
  "confidence_scores": {
    "overall_mean": 0.85,
    "mask_mean": 0.92,
    "background_mean": 0.15
  },
  "contours": [
    {
      "points": [[x1, y1], [x2, y2], ...],
      "area": 1234.5,
      "bounding_box": {"x": 10, "y": 20, "width": 100, "height": 80}
    }
  ],
  "processing_time_ms": 150,
  "model_info": {
    "model_name": "unet",
    "model_variant": "standard",
    "device": "cuda"
  },
  "request_id": "uuid-string"
}
```

#### POST /segment-batch
Process multiple images in batch.

**Request Body:**
```json
{
  "images": [
    {"image_data": "base64_encoded_image_1"},
    {"image_data": "base64_encoded_image_2"}
  ],
  "threshold": 0.5,
  "return_confidence": false,
  "return_contours": false,
  "return_visualizations": false
}
```

**Response:**
```json
{
  "success": true,
  "results": [
    // Array of UNetSegmentationResponse objects
  ],
  "total_processing_time_ms": 250,
  "batch_size": 2,
  "request_id": "uuid-string"
}
```

#### GET /health
Get service health status.

**Response:**
```json
{
  "service_healthy": true,
  "model_loaded": true,
  "memory_pools_ready": true,
  "gpu_available": true,
  "last_health_check": "2025-06-18T11:45:30",
  "model_info": {
    "model_name": "unet",
    "model_variant": "standard",
    "device": "cuda"
  },
  "memory_usage": {
    "input_pool_utilization": 0.3,
    "output_pool_utilization": 0.2
  },
  "capabilities": ["segmentation", "batch_processing", "confidence_scoring"]
}
```

#### GET /model-info
Get detailed model information.

**Response:**
```json
{
  "model_name": "unet",
  "model_variant": "standard",
  "checkpoint_path": "/path/to/model.pth",
  "model_parameters": 7850000,
  "input_shape": [1, 3, 256, 256],
  "output_shape": [1, 1, 256, 256],
  "device": "cuda",
  "architecture_details": {
    "model_type": "AttentionUNet",
    "base_channels": 64,
    "depth": 4,
    "use_attention": true
  },
  "preprocessing_config": {
    "image_size": [256, 256],
    "normalize": true,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
  },
  "supported_formats": ["JPEG", "PNG", "BMP"]
}
```

#### POST /initialize
Manually initialize or reinitialize the service.

**Request Body:**
```json
{
  "force_reload": false,
  "warm_up": true
}
```

**Response:**
```json
{
  "success": true,
  "message": "U-Net service initialized successfully",
  "initialization_time_ms": 5000,
  "service_ready": true,
  "request_id": "uuid-string"
}
```

## Configuration

### Configuration File: `config.yaml`

```yaml
unet:
  model:
    checkpoint_dir: "../models/unet/checkpoints/"
    model_variant: "standard"  # standard, lightweight, deep
    device: "auto"  # auto, cuda, cpu
    fallback_to_cpu: true
    
  preprocessing:
    image_size: [256, 256]
    normalize: true
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
    
  inference:
    batch_size: 8
    threshold: 0.5
    enable_confidence: true
    enable_batch_processing: true
    
  memory:
    pool_size_multiplier: 10
    enable_gpu_pools: true
    max_pool_size_gb: 4
    
  health:
    startup_validation: true
    dummy_inference_test: true
    health_check_interval: 30
```

### Environment Variables

Override configuration with environment variables:

```bash
export UNET_MODEL_DEVICE=cuda
export UNET_INFERENCE_BATCH_SIZE=16
export UNET_MEMORY_MAX_POOL_SIZE_GB=8
export UNET_HEALTH_DUMMY_INFERENCE_TEST=false
```

## Deployment

### Prerequisites

1. **Python Dependencies**
   ```bash
   pip install torch torchvision fastapi uvicorn pillow numpy opencv-python
   ```

2. **Model Checkpoints**
   - Place trained model files in `models/unet/checkpoints/`
   - Supported formats: `.pth`, `.pt`, `.ckpt`
   - Naming convention: `best.pth`, `final.pth`, or timestamp-based

3. **Hardware Requirements**
   - **GPU**: NVIDIA GPU with CUDA support (recommended)
   - **RAM**: Minimum 8GB, recommended 16GB+
   - **VRAM**: Minimum 4GB for GPU inference
   - **Storage**: 2GB for model and dependencies

### Local Development

1. **Start the server**
   ```bash
   cd nail-segmentation/backend
   python -m uvicorn src.server.main:app --host 0.0.0.0 --port 8001 --reload
   ```

2. **Test the endpoints**
   ```bash
   curl http://localhost:8001/api/unet/health
   ```

### Production Deployment

1. **Docker Deployment**
   ```dockerfile
   FROM python:3.10-slim
   
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   
   COPY src/ ./src/
   COPY config.yaml .
   COPY models/ ./models/
   
   EXPOSE 8001
   CMD ["uvicorn", "src.server.main:app", "--host", "0.0.0.0", "--port", "8001"]
   ```

2. **Kubernetes Deployment**
   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: unet-serving
   spec:
     replicas: 3
     selector:
       matchLabels:
         app: unet-serving
     template:
       metadata:
         labels:
           app: unet-serving
       spec:
         containers:
         - name: unet-serving
           image: unet-serving:latest
           ports:
           - containerPort: 8001
           env:
           - name: UNET_MODEL_DEVICE
             value: "cuda"
           resources:
             requests:
               memory: "4Gi"
               nvidia.com/gpu: 1
             limits:
               memory: "8Gi"
               nvidia.com/gpu: 1
   ```

## Performance Optimization

### Memory Optimization

1. **Pool Sizing**
   - Monitor memory usage: `GET /api/unet/health`
   - Adjust `pool_size_multiplier` based on concurrent load
   - Set `max_pool_size_gb` to prevent OOM

2. **Batch Processing**
   - Use batch endpoints for multiple images
   - Optimal batch size: 4-8 images for most GPUs
   - Monitor GPU memory utilization

### Inference Optimization

1. **Model Variants**
   - `lightweight`: Faster inference, lower accuracy
   - `standard`: Balanced performance and accuracy
   - `deep`: Highest accuracy, slower inference

2. **Device Selection**
   - Use GPU for production workloads
   - CPU fallback for development/testing
   - Monitor device utilization

### Monitoring

1. **Health Checks**
   ```bash
   # Basic health check
   curl http://localhost:8001/api/unet/health
   
   # Detailed model info
   curl http://localhost:8001/api/unet/model-info
   ```

2. **Performance Metrics**
   - Processing time per request
   - Memory pool utilization
   - GPU/CPU usage
   - Error rates

## Troubleshooting

### Common Issues

1. **Service Not Starting**
   ```
   Error: No valid model checkpoint found
   ```
   - **Solution**: Ensure model files exist in `checkpoint_dir`
   - **Check**: File permissions and naming convention

2. **CUDA Out of Memory**
   ```
   Error: CUDA out of memory
   ```
   - **Solution**: Reduce `batch_size` or `max_pool_size_gb`
   - **Alternative**: Set `device: cpu` for CPU inference

3. **Slow Inference**
   ```
   Warning: Inference time exceeds threshold
   ```
   - **Solution**: Check GPU utilization and model variant
   - **Optimization**: Use `lightweight` variant for speed

4. **Import Errors**
   ```
   Error: No module named 'training'
   ```
   - **Solution**: Ensure all dependencies are installed
   - **Check**: Python path and module structure

### Debug Mode

Enable debug logging and outputs:

```yaml
development:
  enable_debug_endpoints: true
  save_debug_outputs: true
  debug_output_dir: "./debug_outputs/"
  
unet:
  logging:
    level: "DEBUG"
    log_requests: true
    enable_performance_logging: true
```

### Performance Benchmarking

Run performance tests:

```python
import requests
import time
import base64
from PIL import Image
import io

# Create test image
image = Image.new('RGB', (256, 256), color='red')
buffer = io.BytesIO()
image.save(buffer, format='PNG')
image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

# Benchmark single inference
start_time = time.time()
response = requests.post('http://localhost:8001/api/unet/segment', json={
    'image_data': image_data,
    'threshold': 0.5
})
end_time = time.time()

print(f"Inference time: {(end_time - start_time) * 1000:.2f}ms")
print(f"Service time: {response.json().get('processing_time_ms', 0)}ms")
```

## Security Considerations

1. **Input Validation**
   - All image data is validated before processing
   - Base64 encoding prevents binary injection
   - Request size limits prevent DoS attacks

2. **Error Handling**
   - Structured error responses prevent information leakage
   - Request IDs for secure debugging
   - Rate limiting for production deployments

3. **Authentication**
   - Add API key authentication for production
   - Use HTTPS for encrypted communication
   - Implement request logging for audit trails

## Support

For issues and questions:
- Check logs in the service startup output
- Use health endpoints for service status
- Enable debug mode for detailed diagnostics
- Monitor GPU/CPU utilization and memory usage 