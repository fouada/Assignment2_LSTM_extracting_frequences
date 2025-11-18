"""
Evaluation Metrics Module
Professional implementation of evaluation metrics for frequency extraction.

Author: Professional ML Engineering Team
Date: 2025
"""

import torch
import numpy as np
from typing import Dict, Tuple, List
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)


class FrequencyExtractionMetrics:
    """
    Comprehensive metrics calculator for frequency extraction task.
    """
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.predictions = []
        self.targets = []
        self.frequency_indices = []
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        freq_idx: int
    ):
        """
        Update metrics with new batch.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            freq_idx: Frequency index for this batch
        """
        pred_np = predictions.detach().cpu().numpy().flatten()
        target_np = targets.detach().cpu().numpy().flatten()
        
        self.predictions.extend(pred_np.tolist())
        self.targets.extend(target_np.tolist())
        self.frequency_indices.extend([freq_idx] * len(pred_np))
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary of metric names and values
        """
        if len(self.predictions) == 0:
            logger.warning("No predictions to compute metrics on")
            return {}
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Overall metrics
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        # Correlation
        correlation = np.corrcoef(targets, predictions)[0, 1]
        
        # Signal-to-noise ratio (SNR)
        signal_power = np.mean(targets ** 2)
        noise_power = np.mean((targets - predictions) ** 2)
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2_score': float(r2),
            'correlation': float(correlation),
            'snr_db': float(snr_db)
        }
        
        return metrics
    
    def compute_per_frequency_metrics(self) -> Dict[int, Dict[str, float]]:
        """
        Compute metrics for each frequency separately.
        
        Returns:
            Dictionary mapping frequency index to metrics
        """
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        freq_indices = np.array(self.frequency_indices)
        
        per_freq_metrics = {}
        
        for freq_idx in np.unique(freq_indices):
            mask = freq_indices == freq_idx
            freq_pred = predictions[mask]
            freq_target = targets[mask]
            
            mse = mean_squared_error(freq_target, freq_pred)
            mae = mean_absolute_error(freq_target, freq_pred)
            r2 = r2_score(freq_target, freq_pred)
            
            per_freq_metrics[int(freq_idx)] = {
                'mse': float(mse),
                'mae': float(mae),
                'r2_score': float(r2),
                'num_samples': int(np.sum(mask))
            }
        
        return per_freq_metrics
    
    def get_predictions_and_targets(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get all accumulated predictions, targets, and frequency indices.
        
        Returns:
            Tuple of (predictions, targets, frequency_indices)
        """
        return (
            np.array(self.predictions),
            np.array(self.targets),
            np.array(self.frequency_indices)
        )


def evaluate_model(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    compute_per_frequency: bool = True
) -> Dict:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained model
        data_loader: Data loader for evaluation
        device: Device to run evaluation on
        compute_per_frequency: Whether to compute per-frequency metrics
        
    Returns:
        Dictionary with overall and per-frequency metrics
    """
    model.eval()
    metrics_calculator = FrequencyExtractionMetrics()
    
    logger.info("Starting model evaluation...")
    
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            freq_idx = batch['freq_idx']
            is_first_batch = batch['is_first_batch']
            
            # Reset state at start of each frequency
            if is_first_batch:
                model.reset_state()
            
            # Forward pass
            predictions = model(inputs, reset_state=False)
            
            # Update metrics
            metrics_calculator.update(predictions, targets, freq_idx)
    
    # Compute metrics
    overall_metrics = metrics_calculator.compute_metrics()
    
    results = {
        'overall': overall_metrics
    }
    
    if compute_per_frequency:
        per_freq_metrics = metrics_calculator.compute_per_frequency_metrics()
        results['per_frequency'] = per_freq_metrics
    
    # Get all predictions for visualization
    predictions, targets, freq_indices = metrics_calculator.get_predictions_and_targets()
    results['predictions'] = predictions
    results['targets'] = targets
    results['frequency_indices'] = freq_indices
    
    logger.info("Evaluation completed!")
    logger.info(f"Overall MSE: {overall_metrics['mse']:.6f}")
    logger.info(f"Overall R²: {overall_metrics['r2_score']:.4f}")
    
    return results


def compare_train_test_performance(
    train_metrics: Dict,
    test_metrics: Dict
) -> Dict:
    """
    Compare training and test performance for generalization analysis.
    
    Args:
        train_metrics: Metrics on training set
        test_metrics: Metrics on test set
        
    Returns:
        Dictionary with comparison statistics
    """
    comparison = {}
    
    # Overall metrics comparison
    for metric_name in ['mse', 'mae', 'r2_score']:
        train_val = train_metrics['overall'][metric_name]
        test_val = test_metrics['overall'][metric_name]
        
        # Calculate relative difference
        if metric_name in ['mse', 'mae']:
            # Lower is better
            rel_diff = (test_val - train_val) / (train_val + 1e-10)
        else:
            # Higher is better (R²)
            rel_diff = (train_val - test_val) / (train_val + 1e-10)
        
        comparison[metric_name] = {
            'train': train_val,
            'test': test_val,
            'relative_difference': rel_diff,
            'generalization_good': abs(rel_diff) < 0.1  # Within 10%
        }
    
    # Overall generalization assessment
    mse_gen_good = comparison['mse']['generalization_good']
    r2_gen_good = comparison['r2_score']['generalization_good']
    
    comparison['overall_generalization'] = {
        'good': mse_gen_good and r2_gen_good,
        'status': 'Good' if (mse_gen_good and r2_gen_good) else 'Needs Improvement'
    }
    
    logger.info(f"Generalization Status: {comparison['overall_generalization']['status']}")
    
    return comparison

