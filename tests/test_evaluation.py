"""
Comprehensive Unit Tests for Evaluation Module
Tests metrics calculation, edge cases, and error handling.
Target: 85%+ coverage with documented edge cases.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from src.evaluation.metrics import (
    FrequencyExtractionMetrics,
    evaluate_model,
    compare_train_test_performance
)


class TestFrequencyExtractionMetrics:
    """Test metrics calculator initialization and basic operations."""
    
    def test_initialization(self):
        """Test metrics calculator initializes correctly."""
        metrics = FrequencyExtractionMetrics()
        
        assert metrics.predictions == []
        assert metrics.targets == []
        assert metrics.frequency_indices == []
    
    def test_reset(self):
        """Test reset functionality."""
        metrics = FrequencyExtractionMetrics()
        
        # Add some data
        metrics.predictions = [1.0, 2.0, 3.0]
        metrics.targets = [1.1, 2.1, 3.1]
        metrics.frequency_indices = [0, 0, 1]
        
        # Reset
        metrics.reset()
        
        assert metrics.predictions == []
        assert metrics.targets == []
        assert metrics.frequency_indices == []
    
    def test_update_with_tensors(self):
        """Test updating metrics with torch tensors."""
        metrics = FrequencyExtractionMetrics()
        
        predictions = torch.tensor([[1.0], [2.0], [3.0]])
        targets = torch.tensor([[1.1], [2.1], [3.1]])
        freq_idx = 0
        
        metrics.update(predictions, targets, freq_idx)
        
        assert len(metrics.predictions) == 3
        assert len(metrics.targets) == 3
        assert len(metrics.frequency_indices) == 3
        assert all(fi == 0 for fi in metrics.frequency_indices)
    
    def test_update_multiple_batches(self):
        """Test updating with multiple batches."""
        metrics = FrequencyExtractionMetrics()
        
        # Batch 1
        predictions1 = torch.tensor([[1.0], [2.0]])
        targets1 = torch.tensor([[1.1], [2.1]])
        metrics.update(predictions1, targets1, 0)
        
        # Batch 2
        predictions2 = torch.tensor([[3.0], [4.0]])
        targets2 = torch.tensor([[3.1], [4.1]])
        metrics.update(predictions2, targets2, 1)
        
        assert len(metrics.predictions) == 4
        assert metrics.frequency_indices == [0, 0, 1, 1]


class TestMetricsComputation:
    """Test metrics computation."""
    
    def test_compute_metrics_basic(self, sample_predictions_and_targets):
        """Test computing basic metrics."""
        predictions, targets = sample_predictions_and_targets
        
        metrics_calc = FrequencyExtractionMetrics()
        metrics_calc.predictions = predictions.tolist()
        metrics_calc.targets = targets.tolist()
        metrics_calc.frequency_indices = [0] * len(predictions)
        
        metrics = metrics_calc.compute_metrics()
        
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2_score' in metrics
        assert 'correlation' in metrics
        assert 'snr_db' in metrics
        
        # Check values are reasonable
        assert metrics['mse'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0
        assert -1 <= metrics['r2_score'] <= 1
        assert -1 <= metrics['correlation'] <= 1
    
    def test_compute_metrics_perfect_prediction(self):
        """Test metrics with perfect predictions."""
        metrics_calc = FrequencyExtractionMetrics()
        
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predictions = targets.copy()  # Perfect prediction
        
        metrics_calc.predictions = predictions.tolist()
        metrics_calc.targets = targets.tolist()
        metrics_calc.frequency_indices = [0] * len(predictions)
        
        metrics = metrics_calc.compute_metrics()
        
        assert metrics['mse'] < 1e-10
        assert metrics['mae'] < 1e-10
        assert np.isclose(metrics['r2_score'], 1.0, atol=1e-5)
        assert np.isclose(metrics['correlation'], 1.0, atol=1e-5)
    
    @pytest.mark.edge_case
    def test_compute_metrics_empty_predictions(self):
        """Test computing metrics with no predictions."""
        metrics_calc = FrequencyExtractionMetrics()
        
        metrics = metrics_calc.compute_metrics()
        
        assert metrics == {}
    
    @pytest.mark.edge_case
    def test_compute_metrics_single_value(self):
        """Test metrics with single prediction."""
        metrics_calc = FrequencyExtractionMetrics()
        
        metrics_calc.predictions = [1.0]
        metrics_calc.targets = [1.1]
        metrics_calc.frequency_indices = [0]
        
        metrics = metrics_calc.compute_metrics()
        
        assert 'mse' in metrics
        assert metrics['mse'] >= 0
    
    @pytest.mark.edge_case
    def test_compute_metrics_with_zeros(self):
        """Test metrics when target is all zeros."""
        metrics_calc = FrequencyExtractionMetrics()
        
        targets = np.zeros(100)
        predictions = np.random.randn(100) * 0.1
        
        metrics_calc.predictions = predictions.tolist()
        metrics_calc.targets = targets.tolist()
        metrics_calc.frequency_indices = [0] * len(predictions)
        
        metrics = metrics_calc.compute_metrics()
        
        assert 'mse' in metrics
        assert not np.isnan(metrics['mse'])
    
    @pytest.mark.edge_case
    def test_compute_metrics_constant_prediction(self):
        """Test metrics when predictions are constant."""
        metrics_calc = FrequencyExtractionMetrics()
        
        targets = np.sin(np.linspace(0, 2*np.pi, 100))
        predictions = np.ones(100) * np.mean(targets)  # Constant at mean
        
        metrics_calc.predictions = predictions.tolist()
        metrics_calc.targets = targets.tolist()
        metrics_calc.frequency_indices = [0] * len(predictions)
        
        metrics = metrics_calc.compute_metrics()
        
        # R² should be close to 0 for constant prediction at mean
        assert np.isclose(metrics['r2_score'], 0.0, atol=0.1)


class TestPerFrequencyMetrics:
    """Test per-frequency metrics computation."""
    
    def test_per_frequency_metrics_basic(self):
        """Test computing per-frequency metrics."""
        metrics_calc = FrequencyExtractionMetrics()
        
        # Frequency 0
        metrics_calc.predictions = [1.0, 1.1, 1.2]
        metrics_calc.targets = [1.05, 1.15, 1.25]
        metrics_calc.frequency_indices = [0, 0, 0]
        
        # Frequency 1
        metrics_calc.predictions += [2.0, 2.1, 2.2]
        metrics_calc.targets += [2.05, 2.15, 2.25]
        metrics_calc.frequency_indices += [1, 1, 1]
        
        per_freq = metrics_calc.compute_per_frequency_metrics()
        
        assert 0 in per_freq
        assert 1 in per_freq
        assert 'mse' in per_freq[0]
        assert 'mae' in per_freq[0]
        assert 'r2_score' in per_freq[0]
        assert 'num_samples' in per_freq[0]
        assert per_freq[0]['num_samples'] == 3
        assert per_freq[1]['num_samples'] == 3
    
    def test_per_frequency_metrics_different_sizes(self):
        """Test per-frequency metrics with different sample counts."""
        metrics_calc = FrequencyExtractionMetrics()
        
        # Frequency 0 - 10 samples
        predictions_f0 = list(range(10))
        targets_f0 = [x + 0.1 for x in predictions_f0]
        
        # Frequency 1 - 20 samples
        predictions_f1 = list(range(20))
        targets_f1 = [x + 0.1 for x in predictions_f1]
        
        metrics_calc.predictions = predictions_f0 + predictions_f1
        metrics_calc.targets = targets_f0 + targets_f1
        metrics_calc.frequency_indices = [0]*10 + [1]*20
        
        per_freq = metrics_calc.compute_per_frequency_metrics()
        
        assert per_freq[0]['num_samples'] == 10
        assert per_freq[1]['num_samples'] == 20
    
    @pytest.mark.edge_case
    def test_per_frequency_metrics_single_frequency(self):
        """Test per-frequency metrics with only one frequency."""
        metrics_calc = FrequencyExtractionMetrics()
        
        metrics_calc.predictions = [1.0, 2.0, 3.0]
        metrics_calc.targets = [1.1, 2.1, 3.1]
        metrics_calc.frequency_indices = [0, 0, 0]
        
        per_freq = metrics_calc.compute_per_frequency_metrics()
        
        assert len(per_freq) == 1
        assert 0 in per_freq


class TestGetPredictionsAndTargets:
    """Test retrieving predictions and targets."""
    
    def test_get_predictions_and_targets(self):
        """Test getting all accumulated data."""
        metrics_calc = FrequencyExtractionMetrics()
        
        predictions_list = [1.0, 2.0, 3.0]
        targets_list = [1.1, 2.1, 3.1]
        freq_indices_list = [0, 0, 1]
        
        metrics_calc.predictions = predictions_list
        metrics_calc.targets = targets_list
        metrics_calc.frequency_indices = freq_indices_list
        
        preds, targs, freqs = metrics_calc.get_predictions_and_targets()
        
        assert isinstance(preds, np.ndarray)
        assert isinstance(targs, np.ndarray)
        assert isinstance(freqs, np.ndarray)
        assert len(preds) == 3
        assert len(targs) == 3
        assert len(freqs) == 3
    
    @pytest.mark.edge_case
    def test_get_predictions_empty(self):
        """Test getting predictions when empty."""
        metrics_calc = FrequencyExtractionMetrics()
        
        preds, targs, freqs = metrics_calc.get_predictions_and_targets()
        
        assert len(preds) == 0
        assert len(targs) == 0
        assert len(freqs) == 0


class TestEvaluateModel:
    """Test full model evaluation function."""
    
    def test_evaluate_model_basic(self, minimal_model, minimal_dataloader, device):
        """Test basic model evaluation."""
        minimal_model.eval()
        
        results = evaluate_model(
            model=minimal_model,
            data_loader=minimal_dataloader,
            device=device,
            compute_per_frequency=True
        )
        
        assert 'overall' in results
        assert 'per_frequency' in results
        assert 'predictions' in results
        assert 'targets' in results
        assert 'frequency_indices' in results
        
        assert 'mse' in results['overall']
        assert 'r2_score' in results['overall']
    
    def test_evaluate_model_without_per_frequency(self, minimal_model, minimal_dataloader, device):
        """Test evaluation without per-frequency metrics."""
        minimal_model.eval()
        
        results = evaluate_model(
            model=minimal_model,
            data_loader=minimal_dataloader,
            device=device,
            compute_per_frequency=False
        )
        
        assert 'overall' in results
        assert 'per_frequency' not in results
    
    def test_evaluate_model_predictions_shape(self, minimal_model, minimal_dataloader, device):
        """Test that predictions have correct shape."""
        minimal_model.eval()
        
        results = evaluate_model(
            model=minimal_model,
            data_loader=minimal_dataloader,
            device=device
        )
        
        predictions = results['predictions']
        targets = results['targets']
        
        assert predictions.shape == targets.shape
        assert len(predictions) > 0
    
    @pytest.mark.edge_case
    def test_evaluate_model_resets_state(self, minimal_model, minimal_dataloader, device):
        """Test that model state is properly reset during evaluation."""
        minimal_model.eval()
        
        # Set some state
        minimal_model.hidden_state = torch.randn(1, 1, 32)
        
        results = evaluate_model(
            model=minimal_model,
            data_loader=minimal_dataloader,
            device=device
        )
        
        # Should have computed results successfully
        assert 'overall' in results


class TestCompareTrainTestPerformance:
    """Test train/test performance comparison."""
    
    def test_comparison_basic(self):
        """Test basic performance comparison."""
        train_metrics = {
            'overall': {
                'mse': 0.01,
                'mae': 0.08,
                'r2_score': 0.95
            }
        }
        
        test_metrics = {
            'overall': {
                'mse': 0.012,
                'mae': 0.09,
                'r2_score': 0.93
            }
        }
        
        comparison = compare_train_test_performance(train_metrics, test_metrics)
        
        assert 'mse' in comparison
        assert 'mae' in comparison
        assert 'r2_score' in comparison
        assert 'overall_generalization' in comparison
        
        assert 'train' in comparison['mse']
        assert 'test' in comparison['mse']
        assert 'relative_difference' in comparison['mse']
        assert 'generalization_good' in comparison['mse']
    
    def test_comparison_good_generalization(self):
        """Test comparison with good generalization."""
        train_metrics = {
            'overall': {
                'mse': 0.01,
                'mae': 0.08,
                'r2_score': 0.95
            }
        }
        
        test_metrics = {
            'overall': {
                'mse': 0.011,  # Within 10%
                'mae': 0.085,
                'r2_score': 0.94
            }
        }
        
        comparison = compare_train_test_performance(train_metrics, test_metrics)
        
        assert comparison['mse']['generalization_good']
        assert comparison['overall_generalization']['good']
        assert comparison['overall_generalization']['status'] == 'Good'
    
    def test_comparison_poor_generalization(self):
        """Test comparison with poor generalization (overfitting)."""
        train_metrics = {
            'overall': {
                'mse': 0.01,
                'mae': 0.08,
                'r2_score': 0.95
            }
        }
        
        test_metrics = {
            'overall': {
                'mse': 0.05,  # Much worse
                'mae': 0.20,
                'r2_score': 0.70
            }
        }
        
        comparison = compare_train_test_performance(train_metrics, test_metrics)
        
        assert not comparison['overall_generalization']['good']
        assert comparison['overall_generalization']['status'] == 'Needs Improvement'
    
    @pytest.mark.edge_case
    def test_comparison_perfect_match(self):
        """Test comparison when train and test metrics are identical."""
        train_metrics = {
            'overall': {
                'mse': 0.01,
                'mae': 0.08,
                'r2_score': 0.95
            }
        }
        
        test_metrics = {
            'overall': {
                'mse': 0.01,
                'mae': 0.08,
                'r2_score': 0.95
            }
        }
        
        comparison = compare_train_test_performance(train_metrics, test_metrics)
        
        assert comparison['overall_generalization']['good']
    
    @pytest.mark.edge_case
    def test_comparison_test_better_than_train(self):
        """Test comparison when test performs better than train (rare case)."""
        train_metrics = {
            'overall': {
                'mse': 0.02,
                'mae': 0.10,
                'r2_score': 0.90
            }
        }
        
        test_metrics = {
            'overall': {
                'mse': 0.01,  # Better
                'mae': 0.08,
                'r2_score': 0.95
            }
        }
        
        comparison = compare_train_test_performance(train_metrics, test_metrics)
        
        # Should handle this gracefully
        assert 'overall_generalization' in comparison


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @pytest.mark.edge_case
    def test_metrics_with_nan_predictions(self):
        """Test handling of NaN in predictions."""
        metrics_calc = FrequencyExtractionMetrics()
        
        predictions = np.array([1.0, np.nan, 3.0])
        targets = np.array([1.1, 2.1, 3.1])
        
        metrics_calc.predictions = predictions.tolist()
        metrics_calc.targets = targets.tolist()
        metrics_calc.frequency_indices = [0, 0, 0]
        
        # Should handle NaN gracefully or raise informative error
        try:
            metrics = metrics_calc.compute_metrics()
            # If it computes, MSE should be NaN or very large
            assert np.isnan(metrics['mse']) or metrics['mse'] > 1e6
        except (ValueError, RuntimeError):
            # Or it might raise an error, which is also acceptable
            pass
    
    @pytest.mark.edge_case
    def test_metrics_with_inf_predictions(self):
        """Test handling of infinity in predictions."""
        metrics_calc = FrequencyExtractionMetrics()
        
        predictions = np.array([1.0, np.inf, 3.0])
        targets = np.array([1.1, 2.1, 3.1])
        
        metrics_calc.predictions = predictions.tolist()
        metrics_calc.targets = targets.tolist()
        metrics_calc.frequency_indices = [0, 0, 0]
        
        try:
            metrics = metrics_calc.compute_metrics()
            # Should handle infinity
            assert np.isinf(metrics['mse']) or np.isnan(metrics['mse']) or metrics['mse'] > 1e10
        except (ValueError, RuntimeError):
            pass
    
    @pytest.mark.edge_case
    def test_metrics_with_very_large_values(self):
        """Test metrics with very large numerical values."""
        metrics_calc = FrequencyExtractionMetrics()
        
        predictions = np.array([1e10, 2e10, 3e10])
        targets = np.array([1.1e10, 2.1e10, 3.1e10])
        
        metrics_calc.predictions = predictions.tolist()
        metrics_calc.targets = targets.tolist()
        metrics_calc.frequency_indices = [0, 0, 0]
        
        metrics = metrics_calc.compute_metrics()
        
        # Should compute without overflow
        assert not np.isnan(metrics['mse'])
        assert not np.isinf(metrics['mse'])
    
    @pytest.mark.edge_case
    def test_metrics_with_very_small_values(self):
        """Test metrics with very small numerical values."""
        metrics_calc = FrequencyExtractionMetrics()
        
        predictions = np.array([1e-10, 2e-10, 3e-10])
        targets = np.array([1.1e-10, 2.1e-10, 3.1e-10])
        
        metrics_calc.predictions = predictions.tolist()
        metrics_calc.targets = targets.tolist()
        metrics_calc.frequency_indices = [0, 0, 0]
        
        metrics = metrics_calc.compute_metrics()
        
        # Should compute without underflow
        assert metrics['mse'] >= 0
    
    @pytest.mark.edge_case
    def test_snr_with_zero_noise(self):
        """Test SNR calculation with perfect prediction (zero noise)."""
        metrics_calc = FrequencyExtractionMetrics()
        
        targets = np.array([1.0, 2.0, 3.0, 4.0])
        predictions = targets.copy()  # Perfect prediction
        
        metrics_calc.predictions = predictions.tolist()
        metrics_calc.targets = targets.tolist()
        metrics_calc.frequency_indices = [0] * len(predictions)
        
        metrics = metrics_calc.compute_metrics()
        
        # SNR should be very high (but not infinite due to epsilon)
        assert metrics['snr_db'] > 100  # Very high SNR
    
    @pytest.mark.edge_case
    def test_correlation_with_negative_correlation(self):
        """Test correlation metric with negatively correlated predictions."""
        metrics_calc = FrequencyExtractionMetrics()
        
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predictions = -targets  # Perfectly negatively correlated
        
        metrics_calc.predictions = predictions.tolist()
        metrics_calc.targets = targets.tolist()
        metrics_calc.frequency_indices = [0] * len(predictions)
        
        metrics = metrics_calc.compute_metrics()
        
        # Correlation should be close to -1
        assert -1.05 <= metrics['correlation'] <= -0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

