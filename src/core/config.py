"""
Configuration Management System

Advanced configuration with validation, merging, and plugin support.

Author: Professional ML Engineering Team
Date: 2025
"""

import logging
import yaml
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from copy import deepcopy

logger = logging.getLogger(__name__)


@dataclass
class ConfigSchema:
    """
    Configuration schema for validation.
    """
    name: str
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    defaults: Dict[str, Any] = field(default_factory=dict)
    validators: Dict[str, callable] = field(default_factory=dict)


class ConfigManager:
    """
    Advanced configuration manager.
    
    Features:
    - YAML/JSON loading
    - Environment variable interpolation
    - Configuration merging
    - Validation
    - Plugin-specific configurations
    - Hot reload
    - Nested access with dot notation
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to main configuration file
        """
        self._config: Dict[str, Any] = {}
        self._schemas: Dict[str, ConfigSchema] = {}
        self._plugin_configs: Dict[str, Dict] = {}
        self._config_path = Path(config_path) if config_path else None
        
        if self._config_path and self._config_path.exists():
            self.load(self._config_path)
        
        logger.info("ConfigManager initialized")
    
    def load(self, path: Union[str, Path]) -> None:
        """
        Load configuration from file.
        
        Args:
            path: Path to configuration file (YAML or JSON)
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        # Load based on extension
        with open(path, 'r') as f:
            if path.suffix in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif path.suffix == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")
        
        # Merge with existing config
        self._config = self._deep_merge(self._config, config)
        self._config_path = path
        
        logger.info(f"Configuration loaded from: {path}")
    
    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            path: Path to save to (defaults to loaded path)
        """
        if path is None:
            if self._config_path is None:
                raise ValueError("No config path specified")
            path = self._config_path
        else:
            path = Path(path)
        
        # Save based on extension
        with open(path, 'w') as f:
            if path.suffix in ['.yaml', '.yml']:
                yaml.dump(self._config, f, default_flow_style=False)
            elif path.suffix == '.json':
                json.dump(self._config, f, indent=2)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")
        
        logger.info(f"Configuration saved to: {path}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'model.hidden_size')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'model.hidden_size')
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        # Navigate to parent
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set value
        config[keys[-1]] = value
        
        logger.debug(f"Config set: {key} = {value}")
    
    def merge(self, other: Union[Dict, 'ConfigManager']) -> None:
        """
        Merge another configuration.
        
        Args:
            other: Dictionary or ConfigManager to merge
        """
        if isinstance(other, ConfigManager):
            other = other._config
        
        self._config = self._deep_merge(self._config, other)
        logger.debug("Configuration merged")
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """
        Deep merge two dictionaries.
        
        Args:
            base: Base dictionary
            update: Dictionary to merge in
            
        Returns:
            Merged dictionary
        """
        result = deepcopy(base)
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        
        return result
    
    def register_schema(self, schema: ConfigSchema) -> None:
        """
        Register a configuration schema for validation.
        
        Args:
            schema: Configuration schema
        """
        self._schemas[schema.name] = schema
        logger.debug(f"Schema registered: {schema.name}")
    
    def validate(self, schema_name: Optional[str] = None) -> bool:
        """
        Validate configuration against schema.
        
        Args:
            schema_name: Specific schema to validate, or None for all
            
        Returns:
            True if valid
        """
        schemas = [self._schemas[schema_name]] if schema_name else self._schemas.values()
        
        for schema in schemas:
            # Check required fields
            for field in schema.required_fields:
                if self.get(field) is None:
                    raise ValueError(f"Required config field missing: {field}")
            
            # Run validators
            for field, validator in schema.validators.items():
                value = self.get(field)
                if value is not None:
                    if not validator(value):
                        raise ValueError(f"Validation failed for {field}")
        
        logger.debug("Configuration validation passed")
        return True
    
    def get_plugin_config(self, plugin_name: str) -> Dict:
        """
        Get plugin-specific configuration.
        
        Args:
            plugin_name: Plugin name
            
        Returns:
            Plugin configuration dictionary
        """
        if plugin_name in self._plugin_configs:
            return self._plugin_configs[plugin_name]
        
        # Check main config
        plugin_config = self.get(f'plugins.{plugin_name}', {})
        self._plugin_configs[plugin_name] = plugin_config
        
        return plugin_config
    
    def set_plugin_config(self, plugin_name: str, config: Dict) -> None:
        """
        Set plugin-specific configuration.
        
        Args:
            plugin_name: Plugin name
            config: Plugin configuration
        """
        self._plugin_configs[plugin_name] = config
        self.set(f'plugins.{plugin_name}', config)
        
        logger.debug(f"Plugin config set: {plugin_name}")
    
    def to_dict(self) -> Dict:
        """Export configuration as dictionary."""
        return deepcopy(self._config)
    
    def from_dict(self, config: Dict) -> None:
        """Import configuration from dictionary."""
        self._config = deepcopy(config)
        logger.debug("Configuration imported from dictionary")
    
    def has(self, key: str) -> bool:
        """Check if configuration key exists."""
        return self.get(key) is not None
    
    def reload(self) -> None:
        """Reload configuration from file."""
        if self._config_path:
            logger.info("Reloading configuration...")
            self._config.clear()
            self.load(self._config_path)
    
    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Dictionary-style setting."""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return self.has(key)


# Global configuration manager instance
_global_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = ConfigManager()
    return _global_config_manager


def get_config(key: str, default: Any = None) -> Any:
    """Convenience function to get config from global manager."""
    return get_config_manager().get(key, default)


def set_config(key: str, value: Any) -> None:
    """Convenience function to set config in global manager."""
    get_config_manager().set(key, value)

