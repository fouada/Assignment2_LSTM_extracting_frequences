"""
TensorBoard Logging Plugin

Demonstrates how to create a plugin that hooks into training events.

Author: Professional ML Engineering Team
Date: 2025
"""

import logging
from pathlib import Path
from typing import Optional
from torch.utils.tensorboard import SummaryWriter

from src.core.plugin import Plugin, PluginMetadata
from src.core.events import Event, EventManager, EventPriority
from src.core.hooks import HookManager, HookPriority

logger = logging.getLogger(__name__)


class TensorBoardPlugin(Plugin):
    """
    Plugin for advanced TensorBoard logging.
    
    Features:
    - Custom metric tracking
    - Histogram logging
    - Model graph visualization
    - Embedding visualization
    """
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="tensorboard",
            version="1.0.0",
            author="ML Team",
            description="Advanced TensorBoard logging and visualization",
            tags=["logging", "visualization", "tensorboard"],
            priority=50
        )
    
    def setup(self, **kwargs) -> None:
        """Setup the plugin."""
        self.log_dir = kwargs.get('log_dir', './runs')
        self.log_histograms = kwargs.get('log_histograms', True)
        self.log_interval = kwargs.get('log_interval', 100)
        
        self.writer: Optional[SummaryWriter] = None
        self.step = 0
        
        # Register event handlers
        event_manager = kwargs.get('event_manager')
        if event_manager:
            self._register_events(event_manager)
        
        # Register hooks
        hook_manager = kwargs.get('hook_manager')
        if hook_manager:
            self._register_hooks(hook_manager)
        
        logger.info(f"TensorBoard plugin initialized: {self.log_dir}")
    
    def _register_events(self, event_manager: EventManager) -> None:
        """Register event handlers."""
        event_manager.subscribe(
            EventManager.TRAINING_START,
            self.on_training_start,
            EventPriority.NORMAL
        )
        
        event_manager.subscribe(
            EventManager.BATCH_END,
            self.on_batch_end,
            EventPriority.NORMAL
        )
        
        event_manager.subscribe(
            EventManager.EPOCH_END,
            self.on_epoch_end,
            EventPriority.NORMAL
        )
        
        event_manager.subscribe(
            EventManager.TRAINING_END,
            self.on_training_end,
            EventPriority.NORMAL
        )
    
    def _register_hooks(self, hook_manager: HookManager) -> None:
        """Register hooks."""
        hook_manager.register(
            HookManager.AFTER_OPTIMIZER_STEP,
            self.log_gradients,
            HookPriority.LOW
        )
    
    def on_training_start(self, event: Event) -> None:
        """Handle training start event."""
        log_dir = Path(event.data.get('experiment_dir', self.log_dir)) / 'tensorboard'
        self.writer = SummaryWriter(str(log_dir))
        
        logger.info(f"TensorBoard logging started: {log_dir}")
    
    def on_batch_end(self, event: Event) -> None:
        """Handle batch end event."""
        if self.writer is None:
            return
        
        self.step += 1
        
        # Log loss
        if 'loss' in event.data:
            self.writer.add_scalar('train/batch_loss', event.data['loss'], self.step)
        
        # Log learning rate
        if 'lr' in event.data:
            self.writer.add_scalar('train/learning_rate', event.data['lr'], self.step)
        
        # Log custom metrics
        if 'metrics' in event.data:
            for name, value in event.data['metrics'].items():
                self.writer.add_scalar(f'train/{name}', value, self.step)
    
    def on_epoch_end(self, event: Event) -> None:
        """Handle epoch end event."""
        if self.writer is None:
            return
        
        epoch = event.data.get('epoch', 0)
        
        # Log epoch metrics
        if 'train_loss' in event.data:
            self.writer.add_scalar('epoch/train_loss', event.data['train_loss'], epoch)
        
        if 'val_loss' in event.data:
            self.writer.add_scalar('epoch/val_loss', event.data['val_loss'], epoch)
        
        # Log model weights histograms
        if self.log_histograms and 'model' in event.data:
            model = event.data['model']
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.writer.add_histogram(f'weights/{name}', param.data, epoch)
    
    def on_training_end(self, event: Event) -> None:
        """Handle training end event."""
        if self.writer:
            self.writer.close()
            logger.info("TensorBoard logging ended")
    
    def log_gradients(self, model, **kwargs) -> None:
        """Log gradient statistics."""
        if self.writer is None or not self.log_histograms:
            return
        
        if self.step % self.log_interval == 0:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    self.writer.add_histogram(f'gradients/{name}', param.grad, self.step)
    
    def teardown(self) -> None:
        """Cleanup when plugin is disabled."""
        if self.writer:
            self.writer.close()

