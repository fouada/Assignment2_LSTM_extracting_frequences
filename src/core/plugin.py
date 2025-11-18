"""
Plugin Architecture System
Provides base classes and management for extensible plugin system.

Author: Professional ML Engineering Team
Date: 2025
"""

import importlib
import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Callable
import pkgutil

logger = logging.getLogger(__name__)


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str
    author: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    enabled: bool = True
    priority: int = 100  # Lower number = higher priority


class Plugin(ABC):
    """
    Abstract base class for all plugins.
    
    Plugins can extend functionality at various points in the system:
    - Data preprocessing
    - Model architectures
    - Training strategies
    - Evaluation metrics
    - Visualization
    - Callbacks
    """
    
    def __init__(self):
        self._metadata: Optional[PluginMetadata] = None
        self._hooks: Dict[str, List[Callable]] = {}
        self._initialized = False
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass
    
    def initialize(self, **kwargs) -> None:
        """
        Initialize the plugin with configuration.
        
        Args:
            **kwargs: Configuration parameters
        """
        if self._initialized:
            logger.warning(f"Plugin {self.name} already initialized")
            return
        
        self._metadata = self.get_metadata()
        self.setup(**kwargs)
        self._initialized = True
        logger.info(f"Plugin initialized: {self.name} v{self.version}")
    
    def setup(self, **kwargs) -> None:
        """
        Plugin-specific setup logic.
        
        Override this method to perform custom initialization.
        
        Args:
            **kwargs: Configuration parameters
        """
        pass
    
    def teardown(self) -> None:
        """
        Cleanup resources when plugin is disabled.
        
        Override this method to perform custom cleanup.
        """
        pass
    
    def register_hook(self, hook_name: str, callback: Callable) -> None:
        """
        Register a hook callback.
        
        Args:
            hook_name: Name of the hook point
            callback: Function to call at hook point
        """
        if hook_name not in self._hooks:
            self._hooks[hook_name] = []
        self._hooks[hook_name].append(callback)
        logger.debug(f"Plugin {self.name} registered hook: {hook_name}")
    
    def get_hooks(self, hook_name: Optional[str] = None) -> Dict[str, List[Callable]]:
        """
        Get registered hooks.
        
        Args:
            hook_name: Specific hook name, or None for all hooks
            
        Returns:
            Dictionary of hook names to callbacks
        """
        if hook_name:
            return {hook_name: self._hooks.get(hook_name, [])}
        return self._hooks
    
    @property
    def name(self) -> str:
        """Get plugin name."""
        return self._metadata.name if self._metadata else self.__class__.__name__
    
    @property
    def version(self) -> str:
        """Get plugin version."""
        return self._metadata.version if self._metadata else "0.0.0"
    
    @property
    def enabled(self) -> bool:
        """Check if plugin is enabled."""
        return self._metadata.enabled if self._metadata else False
    
    @property
    def is_initialized(self) -> bool:
        """Check if plugin is initialized."""
        return self._initialized
    
    def __str__(self) -> str:
        return f"Plugin({self.name} v{self.version})"
    
    def __repr__(self) -> str:
        return self.__str__()


class PluginManager:
    """
    Manages plugin lifecycle: discovery, loading, initialization, execution.
    
    Features:
    - Auto-discovery of plugins
    - Dependency resolution
    - Priority-based execution
    - Hot reload support
    - Plugin isolation
    """
    
    def __init__(self, plugin_dirs: Optional[List[Path]] = None):
        """
        Initialize plugin manager.
        
        Args:
            plugin_dirs: List of directories to search for plugins
        """
        self._plugins: Dict[str, Plugin] = {}
        self._plugin_dirs = plugin_dirs or []
        self._load_order: List[str] = []
        
        logger.info("PluginManager initialized")
    
    def discover_plugins(self, package_name: str = "plugins") -> List[Type[Plugin]]:
        """
        Auto-discover plugins in specified package.
        
        Args:
            package_name: Package name to search for plugins
            
        Returns:
            List of discovered plugin classes
        """
        discovered = []
        
        try:
            # Import the plugins package
            plugins_package = importlib.import_module(package_name)
            package_path = Path(plugins_package.__file__).parent
            
            # Iterate through all modules in package
            for importer, modname, ispkg in pkgutil.iter_modules([str(package_path)]):
                try:
                    module_name = f"{package_name}.{modname}"
                    module = importlib.import_module(module_name)
                    
                    # Find all Plugin subclasses in module
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if issubclass(obj, Plugin) and obj != Plugin:
                            discovered.append(obj)
                            logger.debug(f"Discovered plugin: {name} in {module_name}")
                
                except Exception as e:
                    logger.error(f"Error loading module {modname}: {e}")
        
        except ImportError as e:
            logger.warning(f"Plugin package {package_name} not found: {e}")
        
        logger.info(f"Discovered {len(discovered)} plugin(s)")
        return discovered
    
    def register_plugin(self, plugin: Plugin) -> None:
        """
        Register a plugin instance.
        
        Args:
            plugin: Plugin instance to register
        """
        if not isinstance(plugin, Plugin):
            raise TypeError(f"Expected Plugin instance, got {type(plugin)}")
        
        if not plugin.is_initialized:
            plugin.initialize()
        
        name = plugin.name
        
        if name in self._plugins:
            logger.warning(f"Plugin {name} already registered, replacing")
        
        self._plugins[name] = plugin
        self._resolve_load_order()
        
        logger.info(f"Plugin registered: {name}")
    
    def unregister_plugin(self, name: str) -> None:
        """
        Unregister a plugin.
        
        Args:
            name: Plugin name
        """
        if name in self._plugins:
            plugin = self._plugins[name]
            plugin.teardown()
            del self._plugins[name]
            self._resolve_load_order()
            logger.info(f"Plugin unregistered: {name}")
    
    def get_plugin(self, name: str) -> Optional[Plugin]:
        """
        Get plugin by name.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin instance or None
        """
        return self._plugins.get(name)
    
    def list_plugins(self, enabled_only: bool = False) -> List[Plugin]:
        """
        List all registered plugins.
        
        Args:
            enabled_only: Only return enabled plugins
            
        Returns:
            List of plugin instances
        """
        plugins = list(self._plugins.values())
        
        if enabled_only:
            plugins = [p for p in plugins if p.enabled]
        
        return sorted(plugins, key=lambda p: p._metadata.priority)
    
    def enable_plugin(self, name: str) -> None:
        """Enable a plugin."""
        plugin = self.get_plugin(name)
        if plugin:
            plugin._metadata.enabled = True
            logger.info(f"Plugin enabled: {name}")
    
    def disable_plugin(self, name: str) -> None:
        """Disable a plugin."""
        plugin = self.get_plugin(name)
        if plugin:
            plugin._metadata.enabled = False
            logger.info(f"Plugin disabled: {name}")
    
    def _resolve_load_order(self) -> None:
        """
        Resolve plugin load order based on dependencies and priorities.
        
        Uses topological sort for dependency resolution.
        """
        # Simple priority-based ordering for now
        # TODO: Implement full dependency resolution
        plugins = sorted(
            self._plugins.values(),
            key=lambda p: p._metadata.priority
        )
        self._load_order = [p.name for p in plugins]
        logger.debug(f"Plugin load order: {self._load_order}")
    
    def execute_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """
        Execute all registered hooks for a given hook point.
        
        Args:
            hook_name: Name of the hook point
            *args: Positional arguments to pass to hooks
            **kwargs: Keyword arguments to pass to hooks
            
        Returns:
            List of results from all hook executions
        """
        results = []
        
        for plugin_name in self._load_order:
            plugin = self._plugins[plugin_name]
            
            if not plugin.enabled:
                continue
            
            hooks = plugin.get_hooks(hook_name).get(hook_name, [])
            
            for hook in hooks:
                try:
                    result = hook(*args, **kwargs)
                    results.append(result)
                    logger.debug(f"Executed hook {hook_name} from plugin {plugin_name}")
                except Exception as e:
                    logger.error(f"Error executing hook {hook_name} from {plugin_name}: {e}")
        
        return results
    
    def get_plugins_by_tag(self, tag: str) -> List[Plugin]:
        """
        Get all plugins with a specific tag.
        
        Args:
            tag: Tag to filter by
            
        Returns:
            List of matching plugins
        """
        return [
            p for p in self._plugins.values()
            if tag in p._metadata.tags and p.enabled
        ]
    
    def reload_plugin(self, name: str) -> None:
        """
        Hot reload a plugin.
        
        Args:
            name: Plugin name to reload
        """
        plugin = self.get_plugin(name)
        if plugin:
            # Store configuration
            config = plugin.__dict__.copy()
            
            # Unregister
            self.unregister_plugin(name)
            
            # Re-import module
            module = importlib.import_module(plugin.__class__.__module__)
            importlib.reload(module)
            
            # Find plugin class
            plugin_class = getattr(module, plugin.__class__.__name__)
            
            # Create new instance
            new_plugin = plugin_class()
            new_plugin.initialize(**config)
            
            # Register
            self.register_plugin(new_plugin)
            
            logger.info(f"Plugin reloaded: {name}")
    
    def __len__(self) -> int:
        return len(self._plugins)
    
    def __contains__(self, name: str) -> bool:
        return name in self._plugins
    
    def __iter__(self):
        return iter(self._plugins.values())

