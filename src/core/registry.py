"""
Registry System for component registration and discovery.

Provides centralized registration for models, optimizers, losses, metrics, etc.

Author: Professional ML Engineering Team
Date: 2025
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar
import inspect

logger = logging.getLogger(__name__)

T = TypeVar('T')


class Registry:
    """
    Generic registry for components.
    
    Supports:
    - Registration with aliases
    - Factory functions
    - Lazy loading
    - Type validation
    - Auto-discovery
    """
    
    def __init__(self, name: str, base_class: Optional[Type] = None):
        """
        Initialize registry.
        
        Args:
            name: Registry name
            base_class: Base class for type validation
        """
        self.name = name
        self.base_class = base_class
        self._registry: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._aliases: Dict[str, str] = {}
        self._metadata: Dict[str, Dict] = {}
        
        logger.info(f"Registry created: {name}")
    
    def register(
        self,
        name: str,
        component: Any,
        aliases: Optional[List[str]] = None,
        override: bool = False,
        **metadata
    ) -> None:
        """
        Register a component.
        
        Args:
            name: Component name
            component: Component class or instance
            aliases: Alternative names
            override: Whether to override existing registration
            **metadata: Additional metadata
        """
        # Check if already registered
        if name in self._registry and not override:
            raise ValueError(f"Component '{name}' already registered in {self.name}")
        
        # Type validation
        if self.base_class is not None:
            if inspect.isclass(component):
                if not issubclass(component, self.base_class):
                    raise TypeError(
                        f"Component must be subclass of {self.base_class.__name__}"
                    )
            elif not isinstance(component, self.base_class):
                logger.warning(
                    f"Component instance is not of type {self.base_class.__name__}"
                )
        
        # Register component
        self._registry[name] = component
        self._metadata[name] = metadata
        
        # Register aliases
        if aliases:
            for alias in aliases:
                self._aliases[alias] = name
        
        logger.debug(f"Registered '{name}' in {self.name} registry")
    
    def register_factory(
        self,
        name: str,
        factory: Callable,
        aliases: Optional[List[str]] = None,
        **metadata
    ) -> None:
        """
        Register a factory function.
        
        Factory will be called when component is requested.
        
        Args:
            name: Component name
            factory: Factory function
            aliases: Alternative names
            **metadata: Additional metadata
        """
        self._factories[name] = factory
        self._metadata[name] = metadata
        
        if aliases:
            for alias in aliases:
                self._aliases[alias] = name
        
        logger.debug(f"Registered factory '{name}' in {self.name} registry")
    
    def get(self, name: str, *args, **kwargs) -> Any:
        """
        Get a component.
        
        Args:
            name: Component name or alias
            *args: Arguments for factory/constructor
            **kwargs: Keyword arguments for factory/constructor
            
        Returns:
            Component instance
        """
        # Resolve alias
        if name in self._aliases:
            name = self._aliases[name]
        
        # Check factory first
        if name in self._factories:
            factory = self._factories[name]
            return factory(*args, **kwargs)
        
        # Check registry
        if name in self._registry:
            component = self._registry[name]
            
            # If it's a class, instantiate it
            if inspect.isclass(component):
                return component(*args, **kwargs)
            
            # If it's already an instance, return it
            return component
        
        raise KeyError(f"Component '{name}' not found in {self.name} registry")
    
    def has(self, name: str) -> bool:
        """Check if component is registered."""
        return (
            name in self._registry or
            name in self._factories or
            name in self._aliases
        )
    
    def list(self) -> List[str]:
        """List all registered components."""
        return list(set(self._registry.keys()) | set(self._factories.keys()))
    
    def get_metadata(self, name: str) -> Dict:
        """Get component metadata."""
        if name in self._aliases:
            name = self._aliases[name]
        return self._metadata.get(name, {})
    
    def unregister(self, name: str) -> None:
        """Unregister a component."""
        # Remove from registry
        if name in self._registry:
            del self._registry[name]
        
        # Remove from factories
        if name in self._factories:
            del self._factories[name]
        
        # Remove metadata
        if name in self._metadata:
            del self._metadata[name]
        
        # Remove aliases
        aliases_to_remove = [k for k, v in self._aliases.items() if v == name]
        for alias in aliases_to_remove:
            del self._aliases[alias]
        
        logger.debug(f"Unregistered '{name}' from {self.name} registry")
    
    def clear(self) -> None:
        """Clear all registrations."""
        self._registry.clear()
        self._factories.clear()
        self._aliases.clear()
        self._metadata.clear()
        logger.debug(f"Cleared {self.name} registry")
    
    def __contains__(self, name: str) -> bool:
        return self.has(name)
    
    def __len__(self) -> int:
        return len(set(self._registry.keys()) | set(self._factories.keys()))


class ComponentRegistry:
    """
    Central registry for all ML components.
    
    Manages multiple registries for different component types:
    - Models
    - Optimizers
    - Schedulers
    - Loss functions
    - Metrics
    - Callbacks
    - Data transforms
    """
    
    def __init__(self):
        """Initialize component registry."""
        self._registries: Dict[str, Registry] = {}
        
        # Create standard registries
        self.models = self.create_registry('models')
        self.optimizers = self.create_registry('optimizers')
        self.schedulers = self.create_registry('schedulers')
        self.losses = self.create_registry('losses')
        self.metrics = self.create_registry('metrics')
        self.callbacks = self.create_registry('callbacks')
        self.transforms = self.create_registry('transforms')
        self.plugins = self.create_registry('plugins')
        
        logger.info("ComponentRegistry initialized")
    
    def create_registry(self, name: str, base_class: Optional[Type] = None) -> Registry:
        """
        Create a new registry.
        
        Args:
            name: Registry name
            base_class: Base class for type validation
            
        Returns:
            Created Registry
        """
        if name in self._registries:
            raise ValueError(f"Registry '{name}' already exists")
        
        registry = Registry(name, base_class)
        self._registries[name] = registry
        
        return registry
    
    def get_registry(self, name: str) -> Registry:
        """Get a registry by name."""
        if name not in self._registries:
            raise KeyError(f"Registry '{name}' not found")
        return self._registries[name]
    
    def list_registries(self) -> List[str]:
        """List all registries."""
        return list(self._registries.keys())
    
    def __getitem__(self, name: str) -> Registry:
        return self.get_registry(name)


# Global component registry instance
_global_component_registry: Optional[ComponentRegistry] = None


def get_component_registry() -> ComponentRegistry:
    """Get the global component registry instance."""
    global _global_component_registry
    if _global_component_registry is None:
        _global_component_registry = ComponentRegistry()
    return _global_component_registry


def register_model(name: str, model_class: Type, **kwargs) -> None:
    """Convenience function to register a model."""
    get_component_registry().models.register(name, model_class, **kwargs)


def register_optimizer(name: str, optimizer_class: Type, **kwargs) -> None:
    """Convenience function to register an optimizer."""
    get_component_registry().optimizers.register(name, optimizer_class, **kwargs)


def register_metric(name: str, metric_class: Type, **kwargs) -> None:
    """Convenience function to register a metric."""
    get_component_registry().metrics.register(name, metric_class, **kwargs)

