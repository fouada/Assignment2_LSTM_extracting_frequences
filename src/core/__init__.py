"""
Core Framework Module
Production-level ML framework core with plugin architecture.

Author: Professional ML Engineering Team
Date: 2025
"""

from .plugin import Plugin, PluginManager, PluginMetadata
from .hooks import HookManager, Hook, HookPriority
from .registry import Registry, ComponentRegistry
from .events import Event, EventManager, EventPriority
from .container import Container, ServiceProvider
from .config import ConfigManager, ConfigSchema

__all__ = [
    'Plugin',
    'PluginManager',
    'PluginMetadata',
    'HookManager',
    'Hook',
    'HookPriority',
    'Registry',
    'ComponentRegistry',
    'Event',
    'EventManager',
    'EventPriority',
    'Container',
    'ServiceProvider',
    'ConfigManager',
    'ConfigSchema',
]

