"""
Dependency Injection Container

Provides IoC (Inversion of Control) container for managing dependencies.

Author: Professional ML Engineering Team
Date: 2025
"""

import logging
from typing import Any, Callable, Dict, Optional, Type, TypeVar, get_type_hints
import inspect

logger = logging.getLogger(__name__)

T = TypeVar('T')


class Container:
    """
    Dependency Injection Container.
    
    Features:
    - Singleton and transient lifetimes
    - Auto-wiring based on type hints
    - Factory functions
    - Lazy resolution
    - Circular dependency detection
    """
    
    SINGLETON = 'singleton'
    TRANSIENT = 'transient'
    
    def __init__(self):
        """Initialize container."""
        self._bindings: Dict[Type, Dict] = {}
        self._singletons: Dict[Type, Any] = {}
        self._resolving: set = set()  # For circular dependency detection
        
        logger.info("DI Container initialized")
    
    def bind(
        self,
        interface: Type[T],
        implementation: Optional[Type[T]] = None,
        factory: Optional[Callable[[], T]] = None,
        lifetime: str = TRANSIENT,
        **kwargs
    ) -> None:
        """
        Bind an interface to an implementation.
        
        Args:
            interface: Interface type
            implementation: Implementation class
            factory: Factory function (alternative to implementation)
            lifetime: 'singleton' or 'transient'
            **kwargs: Additional configuration
        """
        if implementation is None and factory is None:
            # Self-binding
            implementation = interface
        
        if factory and implementation:
            raise ValueError("Cannot specify both implementation and factory")
        
        self._bindings[interface] = {
            'implementation': implementation,
            'factory': factory,
            'lifetime': lifetime,
            'kwargs': kwargs
        }
        
        logger.debug(f"Bound {interface.__name__} -> {implementation.__name__ if implementation else 'factory'} ({lifetime})")
    
    def bind_instance(self, interface: Type[T], instance: T) -> None:
        """
        Bind an interface to an existing instance (singleton).
        
        Args:
            interface: Interface type
            instance: Instance to bind
        """
        self._bindings[interface] = {
            'implementation': None,
            'factory': None,
            'lifetime': self.SINGLETON,
            'kwargs': {}
        }
        self._singletons[interface] = instance
        
        logger.debug(f"Bound instance for {interface.__name__}")
    
    def resolve(self, interface: Type[T]) -> T:
        """
        Resolve an instance of the interface.
        
        Args:
            interface: Interface type to resolve
            
        Returns:
            Instance of the interface
        """
        # Check for circular dependency
        if interface in self._resolving:
            raise RuntimeError(f"Circular dependency detected for {interface.__name__}")
        
        # Check if already in singleton cache
        if interface in self._singletons:
            return self._singletons[interface]
        
        # Check if binding exists
        if interface not in self._bindings:
            # Try auto-wiring if it's a concrete class
            if inspect.isclass(interface):
                logger.debug(f"Auto-wiring {interface.__name__}")
                return self._auto_wire(interface)
            raise KeyError(f"No binding found for {interface.__name__}")
        
        binding = self._bindings[interface]
        
        # Mark as resolving
        self._resolving.add(interface)
        
        try:
            # Create instance
            if binding['factory']:
                instance = binding['factory']()
            else:
                implementation = binding['implementation']
                instance = self._auto_wire(implementation, **binding['kwargs'])
            
            # Cache if singleton
            if binding['lifetime'] == self.SINGLETON:
                self._singletons[interface] = instance
            
            return instance
        
        finally:
            # Remove from resolving set
            self._resolving.discard(interface)
    
    def _auto_wire(self, cls: Type[T], **kwargs) -> T:
        """
        Auto-wire dependencies based on type hints.
        
        Args:
            cls: Class to instantiate
            **kwargs: Additional keyword arguments
            
        Returns:
            Instance with dependencies injected
        """
        # Get constructor signature
        sig = inspect.signature(cls.__init__)
        params = sig.parameters
        
        # Get type hints
        try:
            hints = get_type_hints(cls.__init__)
        except:
            hints = {}
        
        # Resolve dependencies
        resolved_kwargs = {}
        
        for param_name, param in params.items():
            if param_name == 'self':
                continue
            
            # Skip if already provided
            if param_name in kwargs:
                continue
            
            # Try to resolve from type hint
            if param_name in hints:
                param_type = hints[param_name]
                
                # Skip Optional types for now
                if hasattr(param_type, '__origin__'):
                    continue
                
                try:
                    resolved_kwargs[param_name] = self.resolve(param_type)
                except (KeyError, RuntimeError):
                    # Can't resolve, use default if available
                    if param.default != inspect.Parameter.empty:
                        resolved_kwargs[param_name] = param.default
        
        # Merge with provided kwargs
        resolved_kwargs.update(kwargs)
        
        return cls(**resolved_kwargs)
    
    def has(self, interface: Type) -> bool:
        """Check if interface is registered."""
        return interface in self._bindings or interface in self._singletons
    
    def clear(self) -> None:
        """Clear all bindings and singletons."""
        self._bindings.clear()
        self._singletons.clear()
        self._resolving.clear()
        logger.debug("Container cleared")


class ServiceProvider:
    """
    Service provider for accessing common services.
    
    Provides easy access to framework services like:
    - Plugin manager
    - Event manager
    - Hook manager
    - Registry
    - Configuration
    """
    
    def __init__(self, container: Container):
        """
        Initialize service provider.
        
        Args:
            container: DI container
        """
        self.container = container
        self._services: Dict[str, Any] = {}
    
    def add_service(self, name: str, service: Any) -> None:
        """
        Add a named service.
        
        Args:
            name: Service name
            service: Service instance
        """
        self._services[name] = service
        logger.debug(f"Service added: {name}")
    
    def get_service(self, name: str) -> Any:
        """
        Get a named service.
        
        Args:
            name: Service name
            
        Returns:
            Service instance
        """
        if name not in self._services:
            raise KeyError(f"Service not found: {name}")
        return self._services[name]
    
    def has_service(self, name: str) -> bool:
        """Check if service exists."""
        return name in self._services
    
    def list_services(self) -> list:
        """List all registered services."""
        return list(self._services.keys())


# Global container instance
_global_container: Optional[Container] = None


def get_container() -> Container:
    """Get the global container instance."""
    global _global_container
    if _global_container is None:
        _global_container = Container()
    return _global_container


def resolve(interface: Type[T]) -> T:
    """Convenience function to resolve from global container."""
    return get_container().resolve(interface)

