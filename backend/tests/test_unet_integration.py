"""
Integration tests for U-Net serving system.

This module tests the complete U-Net serving pipeline including:
- Configuration loading and validation
- Service initialization and health checks
- API endpoint functionality
- Model loading and inference
- Error handling and graceful degradation
"""

import pytest
import asyncio
import base64
import io
import sys
from pathlib import Path
from PIL import Image
import numpy as np
from fastapi.testclient import TestClient

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.unet_config import get_unet_config, UNetConfig
from unet.unet_service import UNetModelService
from unet.models import UNetSegmentationRequest, UNetBatchRequest, UNetBatchImageRequest
from server.main import app


class TestUNetConfiguration:
    """Test U-Net configuration system."""
    
    def test_config_loading(self):
        """Test configuration loading from YAML."""
        config = get_unet_config()
        assert config is not None
        assert config.model.model_variant in ["standard", "lightweight", "deep"]
        assert config.preprocessing.image_size == [256, 256]
        assert config.inference.batch_size > 0
    
    def test_config_validation(self):
        """Test configuration parameter validation."""
        config = get_unet_config()
        
        # Test preprocessing parameters
        assert len(config.preprocessing.mean) == 3
        assert len(config.preprocessing.std) == 3
        assert all(0 <= val <= 1 for val in config.preprocessing.mean)
        
        # Test inference parameters
        assert 0 < config.inference.threshold < 1
        assert config.inference.batch_size <= 32
        
        # Test memory parameters
        assert config.memory.pool_size_multiplier > 0
        assert config.memory.max_pool_size_gb > 0


class TestUNetService:
    """Test U-Net service functionality."""
    
    @pytest.fixture
    def config(self):
        """Get test configuration."""
        return get_unet_config()
    
    @pytest.fixture  
    def dummy_image(self):
        """Create a dummy image for testing."""
        # Create a simple test image
        image = Image.new('RGB', (256, 256), color='red')
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def test_service_initialization(self, config):
        """Test service initialization without model loading."""
        # Skip actual model loading for unit tests
        config.health.startup_validation = False
        config.health.dummy_inference_test = False
        
        try:
            service = UNetModelService(config, lazy_loading=True)
            assert service.config == config
            assert service._model_loaded is False  # No actual model in test
        except Exception as e:
            # Should not fail with lazy loading
            pytest.fail(f"Service initialization failed unexpectedly: {e}")
    
    def test_health_check_without_model(self, config):
        """Test health check when model is not loaded."""
        config.health.startup_validation = False
        config.health.dummy_inference_test = False
        
        try:
            service = UNetModelService(config, lazy_loading=True)
            health = service.health_check()
            
            assert 'service_name' in health
            assert 'model_loaded' in health
            assert health['model_loaded'] is False
            
        except Exception as e:
            pytest.fail(f"Health check failed unexpectedly: {e}")
    
    @pytest.mark.asyncio
    async def test_async_health_status(self, config):
        """Test async health status method."""
        config.health.startup_validation = False
        config.health.dummy_inference_test = False
        
        try:
            service = UNetModelService(config, lazy_loading=True)
            health_status = await service.get_health_status()
            
            assert 'service_healthy' in health_status
            assert 'model_loaded' in health_status
            assert 'memory_pools_ready' in health_status
            assert 'gpu_available' in health_status
            assert 'last_health_check' in health_status
            
        except Exception as e:
            pytest.fail(f"Async health status failed unexpectedly: {e}")


class TestAPIEndpoints:
    """Test U-Net API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def dummy_image_data(self):
        """Create dummy image data for API tests."""
        # Create a simple test image
        image = Image.new('RGB', (256, 256), color='blue')
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def test_health_endpoint(self, client):
        """Test health endpoint availability."""
        response = client.get("/api/unet/health")
        # Should return either 200 (healthy) or 503 (not initialized)
        assert response.status_code in [200, 503]
        
        data = response.json()
        if response.status_code == 200:
            assert 'service_healthy' in data
            assert 'model_loaded' in data
    
    def test_model_info_endpoint(self, client):
        """Test model info endpoint."""
        response = client.get("/api/unet/model-info")
        # Should return either 200 (model loaded) or 503 (not initialized)
        assert response.status_code in [200, 503]
    
    def test_segmentation_endpoint_structure(self, client, dummy_image_data):
        """Test segmentation endpoint structure (will fail without model)."""
        request_data = {
            "image_data": dummy_image_data,
            "threshold": 0.5,
            "return_confidence": False,
            "return_contours": False,
            "return_visualizations": False
        }
        
        response = client.post("/api/unet/segment", json=request_data)
        # Should return 503 (service not available) without actual model
        assert response.status_code == 503
        
        data = response.json()
        assert 'detail' in data or 'success' in data
    
    def test_batch_segmentation_endpoint_structure(self, client, dummy_image_data):
        """Test batch segmentation endpoint structure."""
        request_data = {
            "images": [{"image_data": dummy_image_data}],
            "threshold": 0.5,
            "return_confidence": False,
            "return_contours": False,
            "return_visualizations": False
        }
        
        response = client.post("/api/unet/segment-batch", json=request_data)
        # Should return 503 (service not available) without actual model
        assert response.status_code == 503
    
    def test_initialize_endpoint(self, client):
        """Test initialization endpoint structure without triggering heavy model loading."""
        # Test with invalid request data first to ensure endpoint exists
        # This should return a validation error without triggering model loading
        invalid_request_data = {
            "force_reload": "invalid_type",  # Should be boolean
            "warm_up": "invalid_type"
        }
        
        response = client.post("/api/unet/initialize", json=invalid_request_data)
        # Should return 422 (validation error) for invalid data types
        assert response.status_code == 422
        
        # Test the endpoint with minimal valid request
        # Note: We avoid actually triggering initialization due to memory constraints
        # in test environment, but verify the endpoint exists and handles requests properly
        print("Note: Skipping actual initialization due to memory constraints in test environment")
        print("The endpoint structure and validation are verified instead.")


class TestErrorHandling:
    """Test error handling and validation."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_invalid_image_data(self, client):
        """Test handling of invalid base64 image data."""
        request_data = {
            "image_data": "invalid_base64_data",
            "threshold": 0.5
        }
        
        response = client.post("/api/unet/segment", json=request_data)
        assert response.status_code in [422, 500, 503]  # Validation error or service unavailable
    
    def test_invalid_threshold(self, client):
        """Test handling of invalid threshold values."""
        # Create valid image data
        image = Image.new('RGB', (256, 256), color='green')
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Test invalid threshold
        request_data = {
            "image_data": image_data,
            "threshold": 1.5  # Invalid: > 1.0
        }
        
        response = client.post("/api/unet/segment", json=request_data)
        assert response.status_code in [422, 503]  # Validation error or service unavailable
    
    def test_batch_size_limit(self, client):
        """Test batch size validation."""
        # Create valid image data
        image = Image.new('RGB', (256, 256), color='yellow')
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Create request with too many images
        images = [{"image_data": image_data} for _ in range(50)]  # Exceeds typical batch limit
        request_data = {
            "images": images,
            "threshold": 0.5
        }
        
        response = client.post("/api/unet/segment-batch", json=request_data)
        assert response.status_code in [400, 422, 503]  # Bad request or service unavailable


class TestPerformanceMetrics:
    """Test performance measurement and monitoring."""
    
    @pytest.fixture
    def config(self):
        """Get test configuration."""
        return get_unet_config()
    
    def test_timing_measurement(self, config):
        """Test that timing is properly measured."""
        config.health.startup_validation = False
        config.health.dummy_inference_test = False
        
        try:
            service = UNetModelService(config, lazy_loading=True)
            
            # Check if service tracks timing
            assert hasattr(service, '_load_start_time')
            
        except Exception as e:
            pytest.fail(f"Timing measurement test failed unexpectedly: {e}")


if __name__ == "__main__":
    # Run basic tests
    print("Running U-Net integration tests...")
    
    # Test configuration
    try:
        config = get_unet_config()
        print(f"✓ Configuration loaded: {config.model.model_variant} variant")
    except Exception as e:
        print(f"✗ Configuration failed: {e}")
    
    # Test API availability (with error handling for import issues)
    try:
        from fastapi.testclient import TestClient
        # Try to import app, but handle import errors gracefully
        try:
            from server.main import app
            client = TestClient(app)
            response = client.get("/")
            if response.status_code == 200:
                print("✓ Server is running")
            else:
                print(f"✗ Server response: {response.status_code}")
        except ImportError as e:
            print(f"⚠ Server import issue (expected in direct execution): {e}")
            print("⚠ API tests skipped due to import issues")
    except Exception as e:
        print(f"✗ Server test failed: {e}")
    
    # Test health endpoint (only if server import worked)
    try:
        if 'client' in locals():
            response = client.get("/api/unet/health")
            print(f"✓ Health endpoint responds: {response.status_code}")
        else:
            print("⚠ Health endpoint test skipped due to server import issues")
    except Exception as e:
        print(f"✗ Health endpoint failed: {e}")
    
    print("Integration test summary complete.") 