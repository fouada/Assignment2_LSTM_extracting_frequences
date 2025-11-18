"""
Custom Metrics Plugin

Demonstrates how to add custom evaluation metrics.

Author: Professional ML Engineering Team
Date: 2025
"""

import logging
import torch
import numpy as np
from typing import Callable, Dict

from src.core.plugin import Plugin, PluginMetadata
from src.core.registry import get_component_registry
from src.core.hooks import HookManager, HookPriority

logger = logging.getLogger(__name__)


class CustomMetricsPlugin(Plugin):
    """
    Plugin for registering custom evaluation metrics.
    
    Demonstrates:
    - Metric registration
    - Hook-based metric computation
    - Custom metric definitions
    """
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="custom_metrics",
            version="1.0.0",
            author="ML Team",
            description="Custom evaluation metrics for frequency extraction",
            tags=["metrics", "evaluation"],
            priority=60
        )
    
    def setup(self, **kwargs) -> None:
        """Setup the plugin."""
        # Get registry
        registry = get_component_registry()
        
        # Register custom metrics
        self._register_metrics(registry)
        
        # Register hooks
        hook_manager = kwargs.get('hook_manager')
        if hook_manager:
            self._register_hooks(hook_manager)
        
        logger.info("Custom metrics plugin initialized")
    
    def _register_metrics(self, registry) -> None:
        """Register custom metrics."""
        # Frequency-specific error metric
        registry.metrics.register_factory(
            'frequency_error',
            self.frequency_error_factory,
            description="Mean error per frequency component"
        )
        
        # Signal to noise ratio
        registry.metrics.register_factory(
            'snr',
            self.snr_factory,
            description="Signal-to-noise ratio"
        )
        
        # Normalized RMSE
        registry.metrics.register_factory(
            'nrmse',
            self.nrmse_factory,
            description="Normalized root mean squared error"
        )
        
        logger.debug("Custom metrics registered")
    
    def frequency_error_factory(self) -> Callable:
        """Factory for frequency error metric."""
        def frequency_error(predictions: torch.Tensor, targets: torch.Tensor) -> float:
            """Calculate mean absolute error per frequency."""
            error = torch.abs(predictions - targets)
            return error.mean().item()
        
        return frequency_error
    
    def snr_factory(self) -> Callable:
        """Factory for SNR metric."""
        def snr(predictions: torch.Tensor, targets: torch.Tensor) -> float:
            """Calculate signal-to-noise ratio in dB."""
            signal_power = torch.mean(targets ** 2)
            noise_power = torch.mean((targets - predictions) ** 2)
            
            if noise_power < 1e-10:
                return 100.0  # Very high SNR
            
            snr_value = 10 * torch.log10(signal_power / noise_power)
            return snr_value.item()
        
        return snr
    
    def nrmse_factory(self) -> Callable:
        """Factory for normalized RMSE metric."""
        def nrmse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
            """Calculate normalized root mean squared error."""
            rmse = torch.sqrt(torch.mean((predictions - targets) ** 2))
            target_range = targets.max() - targets.min()
            
            if target_range < 1e-10:
                return 0.0
            
            nrmse_value = rmse / target_range
            return nrmse_value.item()
        
        return nrmse
    
    def _register_hooks(self, hook_manager: HookManager) -> None:
        """Register hooks for metric computation."""
        hook_manager.register(
            HookManager.AFTER_EVALUATION,
            self.compute_custom_metrics,
            HookPriority.NORMAL
        )
    
    def compute_custom_metrics(self, results: Dict, **kwargs) -> None:
        """Compute custom metrics after evaluation."""
        if 'predictions' in results and 'targets' in results:
            predictions = results['predictions']
            targets = results['targets']
            
            # Compute custom metrics
            if isinstance(predictions, np.ndarray):
                predictions = torch.from_numpy(predictions)
            if isinstance(targets, np.ndarray):
                targets = torch.from_numpy(targets)
            
            # Add custom metrics to results
            results['custom_metrics'] = {
                'frequency_error': self.frequency_error_factory()(predictions, targets),
                'snr': self.snr_factory()(predictions, targets),
                'nrmse': self.nrmse_factory()(predictions, targets)
            }
            
            logger.debug("Custom metrics computed")

