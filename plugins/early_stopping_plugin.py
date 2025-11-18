"""
Early Stopping Plugin

Demonstrates callback-style plugin for training control.

Author: Professional ML Engineering Team
Date: 2025
"""

import logging
import numpy as np

from src.core.plugin import Plugin, PluginMetadata
from src.core.events import Event, EventManager, EventPriority

logger = logging.getLogger(__name__)


class EarlyStoppingPlugin(Plugin):
    """
    Advanced early stopping with multiple monitoring modes.
    
    Features:
    - Multiple metric monitoring
    - Min-delta threshold
    - Restore best weights
    - Patience countdown
    """
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="early_stopping",
            version="1.0.0",
            author="ML Team",
            description="Advanced early stopping with metric monitoring",
            tags=["training", "optimization", "early_stopping"],
            priority=10  # High priority to stop training early
        )
    
    def setup(self, **kwargs) -> None:
        """Setup the plugin."""
        self.monitor = kwargs.get('monitor', 'val_loss')
        self.patience = kwargs.get('patience', 10)
        self.min_delta = kwargs.get('min_delta', 1e-6)
        self.mode = kwargs.get('mode', 'min')  # 'min' or 'max'
        self.restore_best_weights = kwargs.get('restore_best_weights', True)
        
        self.best_value = np.inf if self.mode == 'min' else -np.inf
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        
        # Register event handlers
        event_manager = kwargs.get('event_manager')
        if event_manager:
            self._register_events(event_manager)
        
        logger.info(f"Early stopping plugin initialized: monitor={self.monitor}, patience={self.patience}")
    
    def _register_events(self, event_manager: EventManager) -> None:
        """Register event handlers."""
        event_manager.subscribe(
            EventManager.EPOCH_END,
            self.on_epoch_end,
            EventPriority.HIGHEST  # Check early stopping first
        )
        
        event_manager.subscribe(
            EventManager.TRAINING_END,
            self.on_training_end,
            EventPriority.NORMAL
        )
    
    def on_epoch_end(self, event: Event) -> None:
        """Check if training should stop."""
        current_value = event.data.get(self.monitor)
        
        if current_value is None:
            logger.warning(f"Metric '{self.monitor}' not found in event data")
            return
        
        epoch = event.data.get('epoch', 0)
        
        # Check if improved
        if self._is_improvement(current_value):
            self.best_value = current_value
            self.best_epoch = epoch
            self.wait = 0
            
            # Save best weights
            if self.restore_best_weights and 'model' in event.data:
                import copy
                self.best_weights = copy.deepcopy(event.data['model'].state_dict())
            
            logger.info(f"Early stopping: New best {self.monitor}={current_value:.6f}")
        else:
            self.wait += 1
            logger.debug(f"Early stopping: No improvement for {self.wait} epochs")
            
            # Check if should stop
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                event.data['stop_training'] = True  # Signal to stop
                
                logger.info(
                    f"Early stopping triggered! "
                    f"Best {self.monitor}={self.best_value:.6f} at epoch {self.best_epoch}"
                )
                
                # Restore best weights
                if self.restore_best_weights and self.best_weights and 'model' in event.data:
                    event.data['model'].load_state_dict(self.best_weights)
                    logger.info("Restored best model weights")
    
    def on_training_end(self, event: Event) -> None:
        """Log final early stopping status."""
        if self.stopped_epoch > 0:
            logger.info(
                f"Early stopping summary: "
                f"Stopped at epoch {self.stopped_epoch}, "
                f"Best epoch was {self.best_epoch}"
            )
    
    def _is_improvement(self, current_value: float) -> bool:
        """Check if current value is an improvement."""
        if self.mode == 'min':
            return current_value < (self.best_value - self.min_delta)
        else:
            return current_value > (self.best_value + self.min_delta)
    
    def teardown(self) -> None:
        """Cleanup."""
        self.best_weights = None

