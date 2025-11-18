"""
Hook System for fine-grained extensibility points.

Hooks allow plugins to inject custom behavior at specific points in the pipeline.

Author: Professional ML Engineering Team
Date: 2025
"""

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class HookPriority(IntEnum):
    """Hook execution priority."""
    HIGHEST = 0
    HIGH = 25
    NORMAL = 50
    LOW = 75
    LOWEST = 100


@dataclass
class Hook:
    """Hook definition."""
    name: str
    callback: Callable
    priority: HookPriority
    enabled: bool = True
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class HookManager:
    """
    Manages execution hooks throughout the system.
    
    Hooks provide fine-grained control points where plugins can inject behavior:
    - before_training_start
    - after_epoch_end
    - before_forward_pass
    - after_loss_calculation
    - etc.
    """
    
    # Training hooks
    BEFORE_TRAINING_START = "before_training_start"
    AFTER_TRAINING_END = "after_training_end"
    BEFORE_EPOCH_START = "before_epoch_start"
    AFTER_EPOCH_END = "after_epoch_end"
    BEFORE_BATCH_START = "before_batch_start"
    AFTER_BATCH_END = "after_batch_end"
    
    # Model hooks
    BEFORE_FORWARD = "before_forward"
    AFTER_FORWARD = "after_forward"
    BEFORE_BACKWARD = "before_backward"
    AFTER_BACKWARD = "after_backward"
    BEFORE_OPTIMIZER_STEP = "before_optimizer_step"
    AFTER_OPTIMIZER_STEP = "after_optimizer_step"
    
    # Data hooks
    BEFORE_DATA_LOAD = "before_data_load"
    AFTER_DATA_LOAD = "after_data_load"
    BEFORE_BATCH_PREPROCESS = "before_batch_preprocess"
    AFTER_BATCH_PREPROCESS = "after_batch_preprocess"
    
    # Evaluation hooks
    BEFORE_EVALUATION = "before_evaluation"
    AFTER_EVALUATION = "after_evaluation"
    BEFORE_METRIC_COMPUTE = "before_metric_compute"
    AFTER_METRIC_COMPUTE = "after_metric_compute"
    
    # Checkpoint hooks
    BEFORE_CHECKPOINT_SAVE = "before_checkpoint_save"
    AFTER_CHECKPOINT_SAVE = "after_checkpoint_save"
    BEFORE_CHECKPOINT_LOAD = "before_checkpoint_load"
    AFTER_CHECKPOINT_LOAD = "after_checkpoint_load"
    
    # Visualization hooks
    BEFORE_PLOT = "before_plot"
    AFTER_PLOT = "after_plot"
    
    def __init__(self):
        """Initialize hook manager."""
        self._hooks: Dict[str, List[Hook]] = defaultdict(list)
        self._hook_counts: Dict[str, int] = defaultdict(int)
        
        logger.info("HookManager initialized")
    
    def register(
        self,
        hook_name: str,
        callback: Callable,
        priority: HookPriority = HookPriority.NORMAL,
        **metadata
    ) -> Hook:
        """
        Register a hook callback.
        
        Args:
            hook_name: Name of the hook point
            callback: Callable to execute at hook point
            priority: Execution priority
            **metadata: Additional metadata
            
        Returns:
            The created Hook object
        """
        hook = Hook(
            name=hook_name,
            callback=callback,
            priority=priority,
            metadata=metadata
        )
        
        self._hooks[hook_name].append(hook)
        
        # Sort by priority
        self._hooks[hook_name].sort(key=lambda h: h.priority.value)
        
        logger.debug(f"Hook registered: {hook_name} (priority: {priority.name})")
        
        return hook
    
    def unregister(self, hook_name: str, callback: Callable) -> bool:
        """
        Unregister a hook callback.
        
        Args:
            hook_name: Hook name
            callback: Callback to remove
            
        Returns:
            True if hook was found and removed
        """
        if hook_name in self._hooks:
            original_len = len(self._hooks[hook_name])
            self._hooks[hook_name] = [
                h for h in self._hooks[hook_name]
                if h.callback != callback
            ]
            removed = original_len != len(self._hooks[hook_name])
            
            if removed:
                logger.debug(f"Hook unregistered: {hook_name}")
            
            return removed
        
        return False
    
    def execute(
        self,
        hook_name: str,
        *args,
        stop_on_error: bool = False,
        **kwargs
    ) -> List[Any]:
        """
        Execute all registered hooks for a hook point.
        
        Args:
            hook_name: Name of hook point
            *args: Positional arguments to pass to hooks
            stop_on_error: Whether to stop execution on first error
            **kwargs: Keyword arguments to pass to hooks
            
        Returns:
            List of results from all hook executions
        """
        results = []
        
        if hook_name not in self._hooks:
            return results
        
        self._hook_counts[hook_name] += 1
        
        for hook in self._hooks[hook_name]:
            if not hook.enabled:
                continue
            
            try:
                result = hook.callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error executing hook '{hook_name}': {e}", exc_info=True)
                if stop_on_error:
                    break
        
        logger.debug(f"Executed {len(results)} hook(s) for: {hook_name}")
        
        return results
    
    def has_hooks(self, hook_name: str) -> bool:
        """Check if hook point has any registered hooks."""
        return hook_name in self._hooks and len(self._hooks[hook_name]) > 0
    
    def get_hooks(self, hook_name: str) -> List[Hook]:
        """Get all hooks for a hook point."""
        return self._hooks.get(hook_name, [])
    
    def get_hook_count(self, hook_name: str) -> int:
        """Get number of times a hook has been executed."""
        return self._hook_counts.get(hook_name, 0)
    
    def clear(self, hook_name: Optional[str] = None) -> None:
        """
        Clear hooks.
        
        Args:
            hook_name: Specific hook to clear, or None for all
        """
        if hook_name:
            if hook_name in self._hooks:
                del self._hooks[hook_name]
                logger.debug(f"Hooks cleared for: {hook_name}")
        else:
            self._hooks.clear()
            self._hook_counts.clear()
            logger.debug("All hooks cleared")
    
    def list_hook_points(self) -> List[str]:
        """List all registered hook points."""
        return list(self._hooks.keys())
    
    def enable_hook(self, hook_name: str, callback: Callable) -> None:
        """Enable a specific hook."""
        for hook in self._hooks.get(hook_name, []):
            if hook.callback == callback:
                hook.enabled = True
                logger.debug(f"Hook enabled: {hook_name}")
    
    def disable_hook(self, hook_name: str, callback: Callable) -> None:
        """Disable a specific hook."""
        for hook in self._hooks.get(hook_name, []):
            if hook.callback == callback:
                hook.enabled = False
                logger.debug(f"Hook disabled: {hook_name}")
    
    def __len__(self) -> int:
        """Return total number of registered hooks."""
        return sum(len(hooks) for hooks in self._hooks.values())


# Global hook manager instance
_global_hook_manager: Optional[HookManager] = None


def get_hook_manager() -> HookManager:
    """Get the global hook manager instance."""
    global _global_hook_manager
    if _global_hook_manager is None:
        _global_hook_manager = HookManager()
    return _global_hook_manager


def register_hook(
    hook_name: str,
    callback: Callable,
    priority: HookPriority = HookPriority.NORMAL,
    **metadata
) -> Hook:
    """Convenience function to register hook with global manager."""
    return get_hook_manager().register(hook_name, callback, priority, **metadata)


def execute_hooks(hook_name: str, *args, **kwargs) -> List[Any]:
    """Convenience function to execute hooks with global manager."""
    return get_hook_manager().execute(hook_name, *args, **kwargs)

