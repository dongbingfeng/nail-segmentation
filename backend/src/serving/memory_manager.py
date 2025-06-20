"""
Memory Pool Manager for Tensor Pre-allocation

Provides efficient memory management for machine learning inference by 
pre-allocating tensor pools and managing memory usage patterns.
"""

import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict, deque
import threading

import torch
import numpy as np


class TensorPool:
    """
    Pool for managing tensors of a specific shape and dtype
    """
    
    def __init__(self, shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device, pool_size: int = 10):
        """
        Initialize tensor pool
        
        Args:
            shape: Tensor shape to pre-allocate
            dtype: Data type of tensors
            device: Device to allocate tensors on
            pool_size: Number of tensors to pre-allocate
        """
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.pool_size = pool_size
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Pool storage
        self._available: deque = deque()
        self._in_use: set = set()
        
        # Statistics
        self._total_allocated = 0
        self._peak_usage = 0
        self._allocation_count = 0
        self._cache_hits = 0
        self._cache_misses = 0
        
    def initialize(self) -> bool:
        """
        Initialize the pool by pre-allocating tensors
        
        Returns:
            bool: True if initialization successful
        """
        try:
            with self._lock:
                for _ in range(self.pool_size):
                    tensor = torch.zeros(self.shape, dtype=self.dtype, device=self.device)
                    self._available.append(tensor)
                    self._total_allocated += 1
                
                return True
                
        except Exception as e:
            logging.error(f"Failed to initialize tensor pool {self.shape}: {e}")
            return False
    
    def get_tensor(self) -> torch.Tensor:
        """
        Get a tensor from the pool or allocate a new one
        
        Returns:
            torch.Tensor: Available tensor
        """
        with self._lock:
            if self._available:
                # Reuse from pool
                tensor = self._available.popleft()
                self._in_use.add(id(tensor))
                self._cache_hits += 1
                
                # Update peak usage
                current_usage = len(self._in_use)
                self._peak_usage = max(self._peak_usage, current_usage)
                
                return tensor
            else:
                # Allocate new tensor
                tensor = torch.zeros(self.shape, dtype=self.dtype, device=self.device)
                self._in_use.add(id(tensor))
                self._total_allocated += 1
                self._cache_misses += 1
                self._allocation_count += 1
                
                # Update peak usage
                current_usage = len(self._in_use)
                self._peak_usage = max(self._peak_usage, current_usage)
                
                return tensor
    
    def return_tensor(self, tensor: torch.Tensor) -> None:
        """
        Return a tensor to the pool
        
        Args:
            tensor: Tensor to return to pool
        """
        with self._lock:
            tensor_id = id(tensor)
            if tensor_id in self._in_use:
                self._in_use.remove(tensor_id)
                
                # Clear tensor data and return to pool
                tensor.zero_()
                self._available.append(tensor)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        with self._lock:
            return {
                "shape": list(self.shape),
                "dtype": str(self.dtype),
                "device": str(self.device),
                "pool_size": self.pool_size,
                "available": len(self._available),
                "in_use": len(self._in_use),
                "total_allocated": self._total_allocated,
                "peak_usage": self._peak_usage,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "hit_ratio": self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0.0
            }
    
    def cleanup(self) -> None:
        """Cleanup pool resources"""
        with self._lock:
            self._available.clear()
            self._in_use.clear()
            
            # Force garbage collection for GPU memory
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()


class MemoryPoolManager:
    """
    Manager for multiple tensor pools with automatic allocation and monitoring
    """
    
    def __init__(self, device: torch.device, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize memory pool manager
        
        Args:
            device: Device to allocate tensors on
            config: Pool configuration dictionary
            logger: Optional logger instance
        """
        self.device = device
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Pool storage
        self.pools: Dict[str, TensorPool] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self._start_time = time.time()
        self._total_memory_allocated = 0
        
    async def initialize_pools(self) -> bool:
        """
        Initialize all configured tensor pools
        
        Returns:
            bool: True if all pools initialized successfully
        """
        try:
            with self._lock:
                pool_configs = self.config.get('pools', {})
                
                for pool_name, pool_config in pool_configs.items():
                    shape = tuple(pool_config['shape'])
                    dtype = getattr(torch, pool_config.get('dtype', 'float32'))
                    pool_size = pool_config.get('size', 10)
                    
                    # Create pool
                    pool = TensorPool(
                        shape=shape,
                        dtype=dtype,
                        device=self.device,
                        pool_size=pool_size
                    )
                    
                    # Initialize pool
                    if pool.initialize():
                        self.pools[pool_name] = pool
                        
                        # Calculate memory usage
                        tensor_size_bytes = np.prod(shape) * torch.tensor([], dtype=dtype).element_size()
                        pool_memory_mb = (tensor_size_bytes * pool_size) / (1024 * 1024)
                        self._total_memory_allocated += pool_memory_mb
                        
                        self.logger.info(f"Initialized pool '{pool_name}': {shape} x{pool_size} ({pool_memory_mb:.1f}MB)")
                    else:
                        self.logger.error(f"Failed to initialize pool '{pool_name}'")
                        return False
                
                self.logger.info(f"All memory pools initialized. Total memory: {self._total_memory_allocated:.1f}MB")
                return True
                
        except Exception as e:
            self.logger.error(f"Memory pool initialization failed: {e}")
            return False
    
    def get_tensor(self, pool_name: str) -> Optional[torch.Tensor]:
        """
        Get a tensor from a specific pool
        
        Args:
            pool_name: Name of the pool to get tensor from
            
        Returns:
            Optional[torch.Tensor]: Tensor from pool or None if pool doesn't exist
        """
        with self._lock:
            if pool_name in self.pools:
                return self.pools[pool_name].get_tensor()
            else:
                self.logger.warning(f"Pool '{pool_name}' not found")
                return None
    
    def return_tensor(self, pool_name: str, tensor: torch.Tensor) -> None:
        """
        Return a tensor to a specific pool
        
        Args:
            pool_name: Name of the pool to return tensor to
            tensor: Tensor to return
        """
        with self._lock:
            if pool_name in self.pools:
                self.pools[pool_name].return_tensor(tensor)
            else:
                self.logger.warning(f"Pool '{pool_name}' not found for tensor return")
    
    def get_tensor_by_shape(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Get a tensor by shape, finding the best matching pool or allocating new
        
        Args:
            shape: Required tensor shape
            dtype: Required tensor dtype
            
        Returns:
            torch.Tensor: Tensor matching requirements
        """
        with self._lock:
            # Try to find exact match in pools
            for pool_name, pool in self.pools.items():
                if pool.shape == shape and pool.dtype == dtype:
                    return pool.get_tensor()
            
            # No matching pool found, allocate directly
            self.logger.debug(f"No pool found for shape {shape}, allocating directly")
            return torch.zeros(shape, dtype=dtype, device=self.device)
    
    def create_dynamic_pool(self, name: str, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32, size: int = 5) -> bool:
        """
        Create a new pool dynamically
        
        Args:
            name: Pool name
            shape: Tensor shape
            dtype: Tensor dtype
            size: Pool size
            
        Returns:
            bool: True if pool created successfully
        """
        with self._lock:
            if name in self.pools:
                self.logger.warning(f"Pool '{name}' already exists")
                return False
            
            try:
                pool = TensorPool(
                    shape=shape,
                    dtype=dtype,
                    device=self.device,
                    pool_size=size
                )
                
                if pool.initialize():
                    self.pools[name] = pool
                    
                    # Update memory tracking
                    tensor_size_bytes = np.prod(shape) * torch.tensor([], dtype=dtype).element_size()
                    pool_memory_mb = (tensor_size_bytes * size) / (1024 * 1024)
                    self._total_memory_allocated += pool_memory_mb
                    
                    self.logger.info(f"Created dynamic pool '{name}': {shape} x{size} ({pool_memory_mb:.1f}MB)")
                    return True
                else:
                    return False
                    
            except Exception as e:
                self.logger.error(f"Failed to create dynamic pool '{name}': {e}")
                return False
    
    def get_pool_status(self) -> Dict[str, Any]:
        """
        Get status of all pools
        
        Returns:
            Dict[str, Any]: Pool status information
        """
        with self._lock:
            pool_stats = {}
            total_available = 0
            total_in_use = 0
            total_allocated = 0
            
            for pool_name, pool in self.pools.items():
                stats = pool.get_stats()
                pool_stats[pool_name] = stats
                total_available += stats['available']
                total_in_use += stats['in_use']
                total_allocated += stats['total_allocated']
            
            # Calculate memory usage
            memory_usage = {}
            if self.device.type == 'cuda':
                try:
                    memory_usage = {
                        "gpu_memory_allocated": torch.cuda.memory_allocated(self.device),
                        "gpu_memory_reserved": torch.cuda.memory_reserved(self.device),
                        "gpu_available": True
                    }
                except Exception:
                    memory_usage = {"gpu_available": False}
            
            return {
                "pools": pool_stats,
                "summary": {
                    "total_pools": len(self.pools),
                    "total_available": total_available,
                    "total_in_use": total_in_use,
                    "total_allocated": total_allocated,
                    "estimated_memory_mb": self._total_memory_allocated,
                    "uptime_seconds": time.time() - self._start_time
                },
                "memory_usage": memory_usage
            }
    
    def optimize_pools(self) -> None:
        """
        Optimize pool sizes based on usage patterns
        """
        with self._lock:
            for pool_name, pool in self.pools.items():
                stats = pool.get_stats()
                
                # If hit ratio is low and we have many misses, consider increasing pool size
                if stats['hit_ratio'] < 0.8 and stats['cache_misses'] > 10:
                    self.logger.info(f"Pool '{pool_name}' has low hit ratio ({stats['hit_ratio']:.2f}), consider increasing size")
                
                # If pool is underutilized, consider reducing size
                elif stats['peak_usage'] < pool.pool_size * 0.3 and stats['total_allocated'] > 20:
                    self.logger.info(f"Pool '{pool_name}' appears underutilized (peak: {stats['peak_usage']}/{pool.pool_size})")
    
    def cleanup(self) -> None:
        """
        Cleanup all pools and free memory
        """
        with self._lock:
            self.logger.info("Cleaning up memory pools")
            
            for pool_name, pool in self.pools.items():
                pool.cleanup()
                self.logger.debug(f"Cleaned up pool '{pool_name}'")
            
            self.pools.clear()
            
            # Force garbage collection
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                self.logger.info("GPU memory cache cleared")
            
            self._total_memory_allocated = 0 