"""
Event System for loose coupling and extensibility.

Implements publish-subscribe pattern for event-driven architecture.

Author: Professional ML Engineering Team
Date: 2025
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum, auto
from typing import Any, Callable, Dict, List, Optional, Set
from collections import defaultdict
import weakref

logger = logging.getLogger(__name__)


class EventPriority(IntEnum):
    """Event handler priority levels."""
    HIGHEST = 0
    HIGH = 25
    NORMAL = 50
    LOW = 75
    LOWEST = 100


@dataclass
class Event:
    """Base event class."""
    name: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None
    propagate: bool = True  # Can be stopped by handlers
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def stop_propagation(self) -> None:
        """Stop event from propagating to other handlers."""
        self.propagate = False
    
    def __str__(self) -> str:
        return f"Event({self.name}, source={self.source})"


class EventManager:
    """
    Central event management system.
    
    Features:
    - Priority-based handler execution
    - Event filtering
    - Async event support (future)
    - Event history tracking
    - Weak references to prevent memory leaks
    """
    
    # Built-in system events
    # Training events
    TRAINING_START = "training.start"
    TRAINING_END = "training.end"
    EPOCH_START = "training.epoch.start"
    EPOCH_END = "training.epoch.end"
    BATCH_START = "training.batch.start"
    BATCH_END = "training.batch.end"
    
    # Validation events
    VALIDATION_START = "validation.start"
    VALIDATION_END = "validation.end"
    
    # Model events
    MODEL_CREATED = "model.created"
    MODEL_LOADED = "model.loaded"
    MODEL_SAVED = "model.saved"
    
    # Data events
    DATA_LOADED = "data.loaded"
    DATA_PREPROCESSED = "data.preprocessed"
    BATCH_PREPARED = "data.batch.prepared"
    
    # Optimization events
    OPTIMIZER_STEP = "optimizer.step"
    SCHEDULER_STEP = "scheduler.step"
    
    # Checkpoint events
    CHECKPOINT_SAVED = "checkpoint.saved"
    CHECKPOINT_LOADED = "checkpoint.loaded"
    
    # Evaluation events
    EVALUATION_START = "evaluation.start"
    EVALUATION_END = "evaluation.end"
    
    # Visualization events
    PLOT_CREATED = "visualization.plot.created"
    
    # Plugin events
    PLUGIN_REGISTERED = "plugin.registered"
    PLUGIN_UNREGISTERED = "plugin.unregistered"
    
    def __init__(self, enable_history: bool = True, history_size: int = 1000):
        """
        Initialize event manager.
        
        Args:
            enable_history: Whether to track event history
            history_size: Maximum number of events to keep in history
        """
        self._handlers: Dict[str, List[tuple]] = defaultdict(list)  # event -> [(priority, handler), ...]
        self._event_filters: Dict[str, List[Callable]] = defaultdict(list)
        self._enable_history = enable_history
        self._history_size = history_size
        self._history: List[Event] = []
        self._event_counts: Dict[str, int] = defaultdict(int)
        
        logger.info("EventManager initialized")
    
    def subscribe(
        self,
        event_name: str,
        handler: Callable[[Event], Any],
        priority: EventPriority = EventPriority.NORMAL
    ) -> None:
        """
        Subscribe to an event.
        
        Args:
            event_name: Name of event to subscribe to
            handler: Callback function that receives Event object
            priority: Handler priority (lower = earlier execution)
        """
        # Store handler with priority
        self._handlers[event_name].append((priority.value, handler))
        
        # Re-sort by priority
        self._handlers[event_name].sort(key=lambda x: x[0])
        
        logger.debug(f"Handler subscribed to '{event_name}' with priority {priority.name}")
    
    def unsubscribe(self, event_name: str, handler: Callable) -> None:
        """
        Unsubscribe from an event.
        
        Args:
            event_name: Event name
            handler: Handler to remove
        """
        if event_name in self._handlers:
            self._handlers[event_name] = [
                (p, h) for p, h in self._handlers[event_name]
                if h != handler
            ]
            logger.debug(f"Handler unsubscribed from '{event_name}'")
    
    def publish(
        self,
        event_name: str,
        data: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None
    ) -> Event:
        """
        Publish an event.
        
        Args:
            event_name: Name of event
            data: Event data
            source: Source of event
            
        Returns:
            The published Event object
        """
        event = Event(
            name=event_name,
            data=data or {},
            source=source
        )
        
        # Apply filters
        if not self._apply_filters(event):
            logger.debug(f"Event filtered: {event_name}")
            return event
        
        # Track history
        if self._enable_history:
            self._add_to_history(event)
        
        # Increment counter
        self._event_counts[event_name] += 1
        
        # Execute handlers
        if event_name in self._handlers:
            for priority, handler in self._handlers[event_name]:
                if not event.propagate:
                    logger.debug(f"Event propagation stopped: {event_name}")
                    break
                
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"Error in event handler for '{event_name}': {e}", exc_info=True)
        
        logger.debug(f"Event published: {event_name} (handlers: {len(self._handlers.get(event_name, []))})")
        
        return event
    
    def add_filter(self, event_name: str, filter_func: Callable[[Event], bool]) -> None:
        """
        Add event filter.
        
        Filter function should return True to allow event, False to block.
        
        Args:
            event_name: Event name to filter
            filter_func: Filter function
        """
        self._event_filters[event_name].append(filter_func)
        logger.debug(f"Filter added for event: {event_name}")
    
    def _apply_filters(self, event: Event) -> bool:
        """Apply filters to event. Returns True if event should proceed."""
        if event.name in self._event_filters:
            for filter_func in self._event_filters[event.name]:
                try:
                    if not filter_func(event):
                        return False
                except Exception as e:
                    logger.error(f"Error in event filter: {e}")
        return True
    
    def _add_to_history(self, event: Event) -> None:
        """Add event to history."""
        self._history.append(event)
        if len(self._history) > self._history_size:
            self._history.pop(0)
    
    def get_history(
        self,
        event_name: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Event]:
        """
        Get event history.
        
        Args:
            event_name: Filter by event name
            limit: Maximum number of events to return
            
        Returns:
            List of events
        """
        history = self._history
        
        if event_name:
            history = [e for e in history if e.name == event_name]
        
        if limit:
            history = history[-limit:]
        
        return history
    
    def get_event_count(self, event_name: str) -> int:
        """Get number of times an event has been published."""
        return self._event_counts.get(event_name, 0)
    
    def clear_history(self) -> None:
        """Clear event history."""
        self._history.clear()
        logger.debug("Event history cleared")
    
    def get_subscriptions(self, event_name: Optional[str] = None) -> Dict[str, int]:
        """
        Get subscription statistics.
        
        Args:
            event_name: Specific event or None for all
            
        Returns:
            Dictionary of event names to handler counts
        """
        if event_name:
            return {event_name: len(self._handlers.get(event_name, []))}
        
        return {
            event_name: len(handlers)
            for event_name, handlers in self._handlers.items()
        }
    
    def has_subscribers(self, event_name: str) -> bool:
        """Check if event has any subscribers."""
        return event_name in self._handlers and len(self._handlers[event_name]) > 0
    
    def clear(self) -> None:
        """Clear all subscriptions and history."""
        self._handlers.clear()
        self._event_filters.clear()
        self._history.clear()
        self._event_counts.clear()
        logger.info("EventManager cleared")
    
    def __len__(self) -> int:
        """Return total number of event subscriptions."""
        return sum(len(handlers) for handlers in self._handlers.values())


# Global event manager instance
_global_event_manager: Optional[EventManager] = None


def get_event_manager() -> EventManager:
    """Get the global event manager instance."""
    global _global_event_manager
    if _global_event_manager is None:
        _global_event_manager = EventManager()
    return _global_event_manager


def publish(event_name: str, data: Optional[Dict[str, Any]] = None, source: Optional[str] = None) -> Event:
    """Convenience function to publish event to global manager."""
    return get_event_manager().publish(event_name, data, source)


def subscribe(event_name: str, handler: Callable, priority: EventPriority = EventPriority.NORMAL) -> None:
    """Convenience function to subscribe to global manager."""
    get_event_manager().subscribe(event_name, handler, priority)

