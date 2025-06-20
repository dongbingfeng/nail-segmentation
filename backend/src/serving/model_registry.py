"""
Model Registry for Discovery and Capability Management

Provides a centralized registry for managing multiple model services,
their capabilities, and health status aggregation.
"""

import logging
from typing import Dict, Any, Optional, List, Type
from abc import ABC, abstractmethod
import threading
import asyncio

from .base_service import BaseModelService


class ModelCapability:
    """
    Represents a model's capabilities and metadata
    """
    
    def __init__(self, 
                 model_type: str,
                 input_format: str,
                 output_format: str,
                 batch_support: bool = False,
                 max_batch_size: int = 1,
                 supported_devices: List[str] = None,
                 metadata: Dict[str, Any] = None):
        """
        Initialize model capability description
        
        Args:
            model_type: Type of model (e.g., 'unet', 'sam')
            input_format: Expected input format description
            output_format: Output format description
            batch_support: Whether model supports batch processing
            max_batch_size: Maximum supported batch size
            supported_devices: List of supported devices
            metadata: Additional metadata
        """
        self.model_type = model_type
        self.input_format = input_format
        self.output_format = output_format
        self.batch_support = batch_support
        self.max_batch_size = max_batch_size
        self.supported_devices = supported_devices or ['cpu', 'cuda']
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert capability to dictionary"""
        return {
            "model_type": self.model_type,
            "input_format": self.input_format,
            "output_format": self.output_format,
            "batch_support": self.batch_support,
            "max_batch_size": self.max_batch_size,
            "supported_devices": self.supported_devices,
            "metadata": self.metadata
        }


class ModelRegistry:
    """
    Registry for managing multiple model services and their capabilities
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize model registry
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Registry storage
        self._services: Dict[str, BaseModelService] = {}
        self._capabilities: Dict[str, ModelCapability] = {}
        self._service_classes: Dict[str, Type[BaseModelService]] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Status tracking
        self._registry_initialized = False
        
    def register_service_class(self, 
                             service_name: str, 
                             service_class: Type[BaseModelService],
                             capability: ModelCapability) -> None:
        """
        Register a service class for later instantiation
        
        Args:
            service_name: Unique name for the service
            service_class: Service class to register
            capability: Service capability description
        """
        with self._lock:
            self._service_classes[service_name] = service_class
            self._capabilities[service_name] = capability
            
            self.logger.info(f"Registered service class '{service_name}': {capability.model_type}")
    
    async def create_service(self, service_name: str, config: Any) -> bool:
        """
        Create and initialize a service instance
        
        Args:
            service_name: Name of service to create
            config: Configuration for the service
            
        Returns:
            bool: True if service created and initialized successfully
        """
        with self._lock:
            if service_name in self._services:
                self.logger.warning(f"Service '{service_name}' already exists")
                return True
            
            if service_name not in self._service_classes:
                self.logger.error(f"No service class registered for '{service_name}'")
                return False
            
            try:
                # Create service instance
                service_class = self._service_classes[service_name]
                service = service_class(config, service_name)
                
                # Initialize service
                if await service.initialize_service():
                    self._services[service_name] = service
                    self.logger.info(f"Service '{service_name}' created and initialized successfully")
                    return True
                else:
                    self.logger.error(f"Failed to initialize service '{service_name}'")
                    return False
                    
            except Exception as e:
                self.logger.error(f"Failed to create service '{service_name}': {e}")
                return False
    
    def get_service(self, service_name: str) -> Optional[BaseModelService]:
        """
        Get a service instance by name
        
        Args:
            service_name: Name of service to retrieve
            
        Returns:
            Optional[BaseModelService]: Service instance or None if not found
        """
        with self._lock:
            return self._services.get(service_name)
    
    def get_capability(self, service_name: str) -> Optional[ModelCapability]:
        """
        Get capability information for a service
        
        Args:
            service_name: Name of service
            
        Returns:
            Optional[ModelCapability]: Capability information or None if not found
        """
        with self._lock:
            return self._capabilities.get(service_name)
    
    def list_services(self) -> List[str]:
        """
        List all registered service names
        
        Returns:
            List[str]: List of service names
        """
        with self._lock:
            return list(self._services.keys())
    
    def list_available_service_classes(self) -> List[str]:
        """
        List all available service classes that can be instantiated
        
        Returns:
            List[str]: List of available service class names
        """
        with self._lock:
            return list(self._service_classes.keys())
    
    async def initialize_all_services(self, service_configs: Dict[str, Any]) -> Dict[str, bool]:
        """
        Initialize all configured services
        
        Args:
            service_configs: Dictionary mapping service names to their configurations
            
        Returns:
            Dict[str, bool]: Results of initialization for each service
        """
        results = {}
        
        for service_name, config in service_configs.items():
            if service_name in self._service_classes:
                results[service_name] = await self.create_service(service_name, config)
            else:
                self.logger.warning(f"No service class registered for '{service_name}', skipping")
                results[service_name] = False
        
        # Update registry status
        with self._lock:
            self._registry_initialized = True
            successful_services = sum(1 for success in results.values() if success)
            self.logger.info(f"Registry initialization complete: {successful_services}/{len(results)} services initialized")
        
        return results
    
    def get_aggregate_health(self) -> Dict[str, Any]:
        """
        Get aggregated health status of all services
        
        Returns:
            Dict[str, Any]: Aggregate health information
        """
        with self._lock:
            service_health = {}
            healthy_count = 0
            total_count = len(self._services)
            
            for service_name, service in self._services.items():
                health = service.health_check()
                service_health[service_name] = health
                
                if health.get('status') == 'healthy':
                    healthy_count += 1
            
            overall_status = 'healthy' if healthy_count == total_count and total_count > 0 else 'unhealthy'
            
            return {
                "overall_status": overall_status,
                "services": service_health,
                "summary": {
                    "total_services": total_count,
                    "healthy_services": healthy_count,
                    "unhealthy_services": total_count - healthy_count,
                    "registry_initialized": self._registry_initialized
                }
            }
    
    def get_aggregate_model_info(self) -> Dict[str, Any]:
        """
        Get aggregated model information from all services
        
        Returns:
            Dict[str, Any]: Aggregate model information
        """
        with self._lock:
            service_info = {}
            capabilities_info = {}
            
            for service_name, service in self._services.items():
                # Get runtime model info
                service_info[service_name] = service.get_model_info()
                
                # Get capability info
                if service_name in self._capabilities:
                    capabilities_info[service_name] = self._capabilities[service_name].to_dict()
            
            return {
                "services": service_info,
                "capabilities": capabilities_info,
                "available_service_classes": list(self._service_classes.keys())
            }
    
    async def process_request(self, service_name: str, input_data: Any, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a request through a specific service
        
        Args:
            service_name: Name of service to use
            input_data: Input data for processing
            metadata: Additional metadata
            
        Returns:
            Dict[str, Any]: Processing result
        """
        service = self.get_service(service_name)
        
        if not service:
            return {
                "success": False,
                "error": {
                    "code": "SERVICE_NOT_FOUND",
                    "message": f"Service '{service_name}' not found",
                    "details": f"Available services: {self.list_services()}"
                }
            }
        
        return await service.process_request(input_data, metadata)
    
    def route_request_by_capability(self, 
                                  required_capability: str, 
                                  input_data: Any, 
                                  metadata: Dict[str, Any] = None) -> Optional[str]:
        """
        Find a service that supports the required capability
        
        Args:
            required_capability: Required model type or capability
            input_data: Input data (for batch size detection)
            metadata: Additional requirements
            
        Returns:
            Optional[str]: Service name that can handle the request, or None
        """
        with self._lock:
            # Simple capability matching - can be extended for more complex routing
            for service_name, capability in self._capabilities.items():
                if capability.model_type == required_capability:
                    # Check if service is healthy and available
                    service = self._services.get(service_name)
                    if service:
                        health = service.health_check()
                        if health.get('status') == 'healthy':
                            return service_name
            
            return None
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """
        Get performance and usage statistics for all services
        
        Returns:
            Dict[str, Any]: Service statistics
        """
        with self._lock:
            statistics = {}
            
            for service_name, service in self._services.items():
                health = service.health_check()
                statistics[service_name] = {
                    "inference_count": health.get("inference_count", 0),
                    "average_inference_time": health.get("average_inference_time"),
                    "last_inference_time": health.get("last_inference_time"),
                    "status": health.get("status"),
                    "memory_usage": health.get("memory_pools", {})
                }
            
            return statistics
    
    async def shutdown_all_services(self) -> None:
        """
        Shutdown all services and cleanup resources
        """
        with self._lock:
            self.logger.info("Shutting down all services")
            
            for service_name, service in self._services.items():
                try:
                    service.shutdown()
                    self.logger.info(f"Service '{service_name}' shutdown complete")
                except Exception as e:
                    self.logger.error(f"Error shutting down service '{service_name}': {e}")
            
            self._services.clear()
            self._registry_initialized = False
            
            self.logger.info("All services shutdown complete")
    
    def remove_service(self, service_name: str) -> bool:
        """
        Remove a service from the registry
        
        Args:
            service_name: Name of service to remove
            
        Returns:
            bool: True if service was removed successfully
        """
        with self._lock:
            if service_name in self._services:
                try:
                    service = self._services[service_name]
                    service.shutdown()
                    del self._services[service_name]
                    
                    self.logger.info(f"Service '{service_name}' removed from registry")
                    return True
                    
                except Exception as e:
                    self.logger.error(f"Error removing service '{service_name}': {e}")
                    return False
            else:
                self.logger.warning(f"Service '{service_name}' not found in registry")
                return False 