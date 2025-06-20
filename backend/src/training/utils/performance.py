"""
Performance monitoring and optimization utilities for training.
"""

import time
import psutil
import torch
import gc
from typing import Dict, Any, Optional
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor training performance metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = {
            'epoch_times': [],
            'batch_times': [],
            'memory_usage': [],
            'gpu_memory_usage': [],
            'cpu_usage': [],
            'data_loading_times': [],
            'forward_pass_times': [],
            'backward_pass_times': []
        }
        self.start_time = None
        self.epoch_start_time = None
        self.batch_start_time = None
    
    def start_epoch(self):
        """Mark the start of an epoch."""
        self.epoch_start_time = time.time()
    
    def end_epoch(self):
        """Mark the end of an epoch and record time."""
        if self.epoch_start_time is not None:
            epoch_time = time.time() - self.epoch_start_time
            self.metrics['epoch_times'].append(epoch_time)
            self.epoch_start_time = None
            return epoch_time
        return 0
    
    def start_batch(self):
        """Mark the start of a batch."""
        self.batch_start_time = time.time()
    
    def end_batch(self):
        """Mark the end of a batch and record time."""
        if self.batch_start_time is not None:
            batch_time = time.time() - self.batch_start_time
            self.metrics['batch_times'].append(batch_time)
            self.batch_start_time = None
            return batch_time
        return 0
    
    def record_memory_usage(self):
        """Record current memory usage."""
        # CPU memory
        process = psutil.Process()
        cpu_memory = process.memory_info().rss / 1024 / 1024  # MB
        self.metrics['memory_usage'].append(cpu_memory)
        
        # GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            self.metrics['gpu_memory_usage'].append(gpu_memory)
        
        # CPU usage
        cpu_percent = psutil.cpu_percent()
        self.metrics['cpu_usage'].append(cpu_percent)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        summary = {}
        
        if self.metrics['epoch_times']:
            summary['avg_epoch_time'] = sum(self.metrics['epoch_times']) / len(self.metrics['epoch_times'])
            summary['total_epoch_time'] = sum(self.metrics['epoch_times'])
        
        if self.metrics['batch_times']:
            summary['avg_batch_time'] = sum(self.metrics['batch_times']) / len(self.metrics['batch_times'])
            summary['batches_per_second'] = len(self.metrics['batch_times']) / sum(self.metrics['batch_times'])
        
        if self.metrics['memory_usage']:
            summary['avg_memory_usage_mb'] = sum(self.metrics['memory_usage']) / len(self.metrics['memory_usage'])
            summary['peak_memory_usage_mb'] = max(self.metrics['memory_usage'])
        
        if self.metrics['gpu_memory_usage']:
            summary['avg_gpu_memory_usage_mb'] = sum(self.metrics['gpu_memory_usage']) / len(self.metrics['gpu_memory_usage'])
            summary['peak_gpu_memory_usage_mb'] = max(self.metrics['gpu_memory_usage'])
        
        if self.metrics['cpu_usage']:
            summary['avg_cpu_usage_percent'] = sum(self.metrics['cpu_usage']) / len(self.metrics['cpu_usage'])
        
        return summary

@contextmanager
def timer(name: str, monitor: Optional[PerformanceMonitor] = None):
    """Context manager for timing operations."""
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        logger.debug(f"{name} took {elapsed:.4f} seconds")
        if monitor and hasattr(monitor.metrics, f'{name.lower()}_times'):
            monitor.metrics[f'{name.lower()}_times'].append(elapsed)

class MemoryOptimizer:
    """Utilities for memory optimization during training."""
    
    @staticmethod
    def clear_cache():
        """Clear GPU cache and run garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    @staticmethod
    def get_memory_info() -> Dict[str, float]:
        """Get current memory information."""
        info = {}
        
        # CPU memory
        process = psutil.Process()
        info['cpu_memory_mb'] = process.memory_info().rss / 1024 / 1024
        info['cpu_memory_percent'] = process.memory_percent()
        
        # System memory
        system_memory = psutil.virtual_memory()
        info['system_memory_total_gb'] = system_memory.total / 1024 / 1024 / 1024
        info['system_memory_available_gb'] = system_memory.available / 1024 / 1024 / 1024
        info['system_memory_percent'] = system_memory.percent
        
        # GPU memory
        if torch.cuda.is_available():
            info['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
            info['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
            info['gpu_memory_max_allocated_mb'] = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        return info
    
    @staticmethod
    def optimize_dataloader_workers(batch_size: int, dataset_size: int) -> int:
        """Determine optimal number of DataLoader workers."""
        # Get number of CPU cores
        cpu_count = psutil.cpu_count(logical=False)
        
        # Conservative approach: use fewer workers to avoid memory issues
        if dataset_size < 100:
            return min(2, cpu_count)
        elif dataset_size < 1000:
            return min(4, cpu_count)
        else:
            return min(8, cpu_count)
    
    @staticmethod
    def suggest_batch_size(model: torch.nn.Module, input_shape: tuple, device: torch.device) -> int:
        """Suggest optimal batch size based on available memory."""
        if not torch.cuda.is_available() or device.type == 'cpu':
            # For CPU, use smaller batch sizes
            return 4
        
        # Start with a small batch size and increase until we hit memory limits
        model.eval()
        batch_sizes = [1, 2, 4, 8, 16, 32]
        optimal_batch_size = 1
        
        for batch_size in batch_sizes:
            try:
                # Clear cache before testing
                torch.cuda.empty_cache()
                
                # Create dummy input
                dummy_input = torch.randn(batch_size, *input_shape[1:]).to(device)
                
                # Test forward pass
                with torch.no_grad():
                    _ = model(dummy_input)
                
                # Test backward pass (approximate)
                dummy_input.requires_grad_(True)
                output = model(dummy_input)
                loss = output.sum()
                loss.backward()
                
                optimal_batch_size = batch_size
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    break
                else:
                    raise e
            finally:
                # Clean up
                torch.cuda.empty_cache()
                gc.collect()
        
        return optimal_batch_size

class TrainingOptimizer:
    """Optimization utilities for training speed."""
    
    @staticmethod
    def setup_mixed_precision() -> Optional[torch.cuda.amp.GradScaler]:
        """Setup mixed precision training if available."""
        if torch.cuda.is_available() and hasattr(torch.cuda.amp, 'autocast'):
            return torch.cuda.amp.GradScaler()
        return None
    
    @staticmethod
    def optimize_model_for_inference(model: torch.nn.Module) -> torch.nn.Module:
        """Optimize model for faster inference."""
        # Set to evaluation mode
        model.eval()
        
        # Disable gradient computation
        for param in model.parameters():
            param.requires_grad = False
        
        # Try to use TorchScript if possible
        try:
            # Create dummy input for tracing
            dummy_input = torch.randn(1, 3, 256, 256)
            if next(model.parameters()).is_cuda:
                dummy_input = dummy_input.cuda()
            
            # Trace the model
            traced_model = torch.jit.trace(model, dummy_input)
            logger.info("Model successfully traced with TorchScript")
            return traced_model
            
        except Exception as e:
            logger.warning(f"Could not trace model with TorchScript: {e}")
            return model
    
    @staticmethod
    def setup_compile_optimization(model: torch.nn.Module) -> torch.nn.Module:
        """Setup torch.compile optimization if available (PyTorch 2.0+)."""
        if hasattr(torch, 'compile'):
            try:
                compiled_model = torch.compile(model)
                logger.info("Model compiled with torch.compile")
                return compiled_model
            except Exception as e:
                logger.warning(f"Could not compile model: {e}")
        
        return model

def log_system_info():
    """Log system information for debugging."""
    logger.info("=== SYSTEM INFORMATION ===")
    
    # Python and PyTorch versions
    import sys
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    
    # CPU information
    logger.info(f"CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    logger.info(f"CPU frequency: {psutil.cpu_freq().current:.2f} MHz")
    
    # Memory information
    memory = psutil.virtual_memory()
    logger.info(f"System memory: {memory.total / 1024**3:.2f} GB total, {memory.available / 1024**3:.2f} GB available")
    
    # GPU information
    if torch.cuda.is_available():
        logger.info(f"CUDA available: True")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {gpu_props.name}, {gpu_props.total_memory / 1024**3:.2f} GB")
    else:
        logger.info("CUDA available: False")
    
    logger.info("=== END SYSTEM INFORMATION ===")

def benchmark_training_step(model: torch.nn.Module, 
                          dataloader: torch.utils.data.DataLoader,
                          device: torch.device,
                          num_steps: int = 10) -> Dict[str, float]:
    """Benchmark training step performance."""
    model.train()
    
    # Warmup
    for i, batch in enumerate(dataloader):
        if i >= 3:  # 3 warmup steps
            break
        
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        with torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.no_grad():
            outputs = model(images)
    
    # Actual benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    for i, batch in enumerate(dataloader):
        if i >= num_steps:
            break
        
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        with torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.no_grad():
            outputs = model(images)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_step_time = total_time / num_steps
    steps_per_second = num_steps / total_time
    
    return {
        'total_time': total_time,
        'avg_step_time': avg_step_time,
        'steps_per_second': steps_per_second
    } 