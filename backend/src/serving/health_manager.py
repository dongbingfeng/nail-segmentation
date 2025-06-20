"""
Health Check Manager for Binary Health Status Monitoring

Provides simple binary health indicators for services with validation
of model loading, memory pools, and inference capability.
"""

import time
import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
import threading

import torch


class HealthStatus(Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    INITIALIZING = "initializing"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check definition"""
    name: str
    check_function: Callable[[], bool]
    description: str
    critical: bool = True  # Whether failure of this check makes service unhealthy
    timeout_seconds: float = 5.0


class HealthCheckManager:
    """
    Manager for binary health status monitoring of services
    """
    
    def __init__(self, service_name: str, logger: Optional[logging.Logger] = None):
        """
        Initialize health check manager
        
        Args:
            service_name: Name of the service being monitored
            logger: Optional logger instance
        """
        self.service_name = service_name
        self.logger = logger or logging.getLogger(__name__)
        
        # Health checks registry
        self._health_checks: List[HealthCheck] = []
        self._last_check_results: Dict[str, bool] = {}
        self._last_check_time: Optional[float] = None
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Status tracking
        self._overall_status = HealthStatus.UNKNOWN
        self._status_message = "Not initialized"
        self._initialization_complete = False
        
        # Performance tracking
        self._check_count = 0
        self._total_check_time = 0.0
        
    def add_health_check(self, health_check: HealthCheck) -> None:
        """
        Add a health check to the manager
        
        Args:
            health_check: Health check to add
        """
        with self._lock:
            self._health_checks.append(health_check)
            self.logger.debug(f"Added health check '{health_check.name}' for {self.service_name}")
    
    def add_standard_checks(self, 
                          model_loaded_check: Callable[[], bool],
                          memory_pools_check: Callable[[], bool],
                          device_check: Callable[[], bool]) -> None:
        """
        Add standard health checks for model services
        
        Args:
            model_loaded_check: Function to check if model is loaded
            memory_pools_check: Function to check if memory pools are ready
            device_check: Function to check if device is available
        """
        standard_checks = [
            HealthCheck(
                name="model_loaded",
                check_function=model_loaded_check,
                description="Verify model is loaded and ready",
                critical=True,
                timeout_seconds=2.0
            ),
            HealthCheck(
                name="memory_pools_ready",
                check_function=memory_pools_check,
                description="Verify memory pools are initialized",
                critical=True,
                timeout_seconds=1.0
            ),
            HealthCheck(
                name="device_available",
                check_function=device_check,
                description="Verify compute device is available",
                critical=True,
                timeout_seconds=1.0
            )
        ]
        
        for check in standard_checks:
            self.add_health_check(check)
    
    def add_inference_capability_check(self, inference_test_function: Callable[[], bool]) -> None:
        """
        Add inference capability test
        
        Args:
            inference_test_function: Function to test inference capability
        """
        inference_check = HealthCheck(
            name="inference_capability",
            check_function=inference_test_function,
            description="Verify inference capability with dummy data",
            critical=True,
            timeout_seconds=10.0
        )
        
        self.add_health_check(inference_check)
    
    async def run_health_checks(self) -> Dict[str, Any]:
        """
        Run all health checks and return status
        
        Returns:
            Dict[str, Any]: Health check results
        """
        start_time = time.time()
        
        with self._lock:
            check_results = {}
            failed_checks = []
            critical_failures = []
            
            # Run each health check with timeout
            for health_check in self._health_checks:
                try:
                    # Run check with timeout
                    check_start = time.time()
                    
                    # Simple timeout implementation
                    result = await asyncio.wait_for(
                        self._run_single_check(health_check),
                        timeout=health_check.timeout_seconds
                    )
                    
                    check_duration = time.time() - check_start
                    
                    check_results[health_check.name] = {
                        "passed": result,
                        "description": health_check.description,
                        "critical": health_check.critical,
                        "duration_seconds": check_duration
                    }
                    
                    # Track failures
                    if not result:
                        failed_checks.append(health_check.name)
                        if health_check.critical:
                            critical_failures.append(health_check.name)
                    
                    self._last_check_results[health_check.name] = result
                    
                except asyncio.TimeoutError:
                    self.logger.warning(f"Health check '{health_check.name}' timed out after {health_check.timeout_seconds}s")
                    check_results[health_check.name] = {
                        "passed": False,
                        "description": health_check.description,
                        "critical": health_check.critical,
                        "duration_seconds": health_check.timeout_seconds,
                        "error": "Timeout"
                    }
                    failed_checks.append(health_check.name)
                    if health_check.critical:
                        critical_failures.append(health_check.name)
                    
                    self._last_check_results[health_check.name] = False
                    
                except Exception as e:
                    self.logger.error(f"Health check '{health_check.name}' failed with exception: {e}")
                    check_results[health_check.name] = {
                        "passed": False,
                        "description": health_check.description,
                        "critical": health_check.critical,
                        "duration_seconds": time.time() - check_start,
                        "error": str(e)
                    }
                    failed_checks.append(health_check.name)
                    if health_check.critical:
                        critical_failures.append(health_check.name)
                    
                    self._last_check_results[health_check.name] = False
            
            # Determine overall status
            if critical_failures:
                self._overall_status = HealthStatus.UNHEALTHY
                self._status_message = f"Critical health checks failed: {', '.join(critical_failures)}"
            elif failed_checks:
                self._overall_status = HealthStatus.HEALTHY  # Non-critical failures don't mark as unhealthy
                self._status_message = f"Non-critical health checks failed: {', '.join(failed_checks)}"
            else:
                self._overall_status = HealthStatus.HEALTHY
                self._status_message = "All health checks passed"
            
            # Update tracking
            total_duration = time.time() - start_time
            self._check_count += 1
            self._total_check_time += total_duration
            self._last_check_time = time.time()
            
            return {
                "service_name": self.service_name,
                "status": self._overall_status.value,
                "message": self._status_message,
                "checks": check_results,
                "summary": {
                    "total_checks": len(self._health_checks),
                    "passed_checks": len(self._health_checks) - len(failed_checks),
                    "failed_checks": len(failed_checks),
                    "critical_failures": len(critical_failures),
                    "check_duration_seconds": total_duration
                },
                "metadata": {
                    "last_check_time": self._last_check_time,
                    "check_count": self._check_count,
                    "average_check_time": self._total_check_time / self._check_count if self._check_count > 0 else 0
                }
            }
    
    async def _run_single_check(self, health_check: HealthCheck) -> bool:
        """
        Run a single health check
        
        Args:
            health_check: Health check to run
            
        Returns:
            bool: True if check passed, False otherwise
        """
        try:
            # Handle both sync and async check functions
            result = health_check.check_function()
            if asyncio.iscoroutine(result):
                result = await result
            
            return bool(result)
            
        except Exception as e:
            self.logger.error(f"Health check '{health_check.name}' raised exception: {e}")
            return False
    
    def get_quick_status(self) -> Dict[str, Any]:
        """
        Get quick health status without running checks
        
        Returns:
            Dict[str, Any]: Current status information
        """
        with self._lock:
            return {
                "service_name": self.service_name,
                "status": self._overall_status.value,
                "message": self._status_message,
                "last_check_time": self._last_check_time,
                "initialization_complete": self._initialization_complete,
                "total_checks_configured": len(self._health_checks)
            }
    
    def is_healthy(self) -> bool:
        """
        Check if service is currently healthy
        
        Returns:
            bool: True if healthy, False otherwise
        """
        return self._overall_status == HealthStatus.HEALTHY
    
    def mark_initialization_complete(self) -> None:
        """Mark service initialization as complete"""
        with self._lock:
            self._initialization_complete = True
            self.logger.info(f"Service '{self.service_name}' initialization marked complete")
    
    def mark_initialization_failed(self, reason: str) -> None:
        """
        Mark service initialization as failed
        
        Args:
            reason: Reason for initialization failure
        """
        with self._lock:
            self._overall_status = HealthStatus.UNHEALTHY
            self._status_message = f"Initialization failed: {reason}"
            self._initialization_complete = False
            self.logger.error(f"Service '{self.service_name}' initialization failed: {reason}")
    
    def create_gpu_memory_check(self, device: torch.device) -> Callable[[], bool]:
        """
        Create a GPU memory availability check
        
        Args:
            device: PyTorch device to check
            
        Returns:
            Callable[[], bool]: Check function
        """
        def check_gpu_memory() -> bool:
            if device.type != 'cuda':
                return True  # Non-GPU devices always pass
            
            try:
                # Try to allocate a small tensor to verify GPU is accessible
                test_tensor = torch.zeros(1, device=device)
                del test_tensor
                return True
            except Exception as e:
                self.logger.warning(f"GPU memory check failed: {e}")
                return False
        
        return check_gpu_memory
    
    def create_model_inference_check(self, model: torch.nn.Module, 
                                   device: torch.device,
                                   input_shape: tuple) -> Callable[[], bool]:
        """
        Create a model inference capability check
        
        Args:
            model: Model to test
            device: Device to run test on
            input_shape: Shape of dummy input tensor
            
        Returns:
            Callable[[], bool]: Check function
        """
        def check_inference() -> bool:
            try:
                # Create dummy input
                dummy_input = torch.randn(input_shape, device=device)
                
                # Run inference
                model.eval()
                with torch.no_grad():
                    output = model(dummy_input)
                
                # Verify output is reasonable
                if output is None or torch.isnan(output).any() or torch.isinf(output).any():
                    self.logger.warning("Model inference check failed: invalid output")
                    return False
                
                return True
                
            except Exception as e:
                self.logger.warning(f"Model inference check failed: {e}")
                return False
        
        return check_inference
    
    def create_memory_pool_check(self, memory_manager) -> Callable[[], bool]:
        """
        Create a memory pool status check
        
        Args:
            memory_manager: Memory pool manager to check
            
        Returns:
            Callable[[], bool]: Check function
        """
        def check_memory_pools() -> bool:
            try:
                if memory_manager is None:
                    return True  # No memory manager is acceptable
                
                status = memory_manager.get_pool_status()
                
                # Check if any pools exist and are functional
                pools = status.get('pools', {})
                if not pools:
                    return True  # No pools configured is acceptable
                
                # Verify pools are initialized
                for pool_name, pool_stats in pools.items():
                    if pool_stats.get('total_allocated', 0) == 0:
                        self.logger.warning(f"Memory pool '{pool_name}' has no allocated tensors")
                        return False
                
                return True
                
            except Exception as e:
                self.logger.warning(f"Memory pool check failed: {e}")
                return False
        
        return check_memory_pools
    
    def get_health_statistics(self) -> Dict[str, Any]:
        """
        Get health check performance statistics
        
        Returns:
            Dict[str, Any]: Health check statistics
        """
        with self._lock:
            return {
                "service_name": self.service_name,
                "total_health_checks": self._check_count,
                "average_check_duration": self._total_check_time / self._check_count if self._check_count > 0 else 0,
                "last_check_results": self._last_check_results.copy(),
                "current_status": self._overall_status.value,
                "initialization_complete": self._initialization_complete
            } 