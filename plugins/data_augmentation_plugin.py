"""
Data Augmentation Plugin

Demonstrates data preprocessing hooks and transformations.

Author: Professional ML Engineering Team
Date: 2025
"""

import logging
import torch
import numpy as np
from typing import Dict

from src.core.plugin import Plugin, PluginMetadata
from src.core.hooks import HookManager, HookPriority

logger = logging.getLogger(__name__)


class DataAugmentationPlugin(Plugin):
    """
    Plugin for data augmentation during training.
    
    Features:
    - Noise injection
    - Signal scaling
    - Time shifting
    - Hook-based augmentation
    """
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="data_augmentation",
            version="1.0.0",
            author="ML Team",
            description="Data augmentation for signal processing",
            tags=["data", "augmentation", "preprocessing"],
            priority=80
        )
    
    def setup(self, **kwargs) -> None:
        """Setup the plugin."""
        self.noise_std = kwargs.get('noise_std', 0.01)
        self.scale_range = kwargs.get('scale_range', (0.95, 1.05))
        self.enabled_augmentations = kwargs.get('augmentations', ['noise', 'scale'])
        self.augmentation_prob = kwargs.get('augmentation_prob', 0.5)
        
        # Register hooks
        hook_manager = kwargs.get('hook_manager')
        if hook_manager:
            self._register_hooks(hook_manager)
        
        logger.info(f"Data augmentation plugin initialized: {self.enabled_augmentations}")
    
    def _register_hooks(self, hook_manager: HookManager) -> None:
        """Register hooks for data augmentation."""
        hook_manager.register(
            HookManager.AFTER_BATCH_PREPROCESS,
            self.augment_batch,
            HookPriority.HIGH
        )
    
    def augment_batch(self, batch: Dict, training: bool = True, **kwargs) -> Dict:
        """
        Augment a batch of data.
        
        Args:
            batch: Batch dictionary with 'input' and 'target'
            training: Whether in training mode
            **kwargs: Additional arguments
            
        Returns:
            Augmented batch
        """
        # Only augment during training
        if not training:
            return batch
        
        # Random chance to apply augmentation
        if np.random.random() > self.augmentation_prob:
            return batch
        
        inputs = batch['input']
        
        # Apply augmentations
        if 'noise' in self.enabled_augmentations:
            inputs = self._add_noise(inputs)
        
        if 'scale' in self.enabled_augmentations:
            inputs = self._scale_signal(inputs)
        
        batch['input'] = inputs
        
        logger.debug("Batch augmented")
        
        return batch
    
    def _add_noise(self, inputs: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to inputs."""
        noise = torch.randn_like(inputs) * self.noise_std
        return inputs + noise
    
    def _scale_signal(self, inputs: torch.Tensor) -> torch.Tensor:
        """Randomly scale signal amplitude."""
        scale = np.random.uniform(*self.scale_range)
        
        # Only scale the signal value (first feature), not one-hot encoding
        if inputs.dim() == 2:  # (batch, features)
            inputs[:, 0] *= scale
        elif inputs.dim() == 3:  # (batch, seq, features)
            inputs[:, :, 0] *= scale
        
        return inputs

