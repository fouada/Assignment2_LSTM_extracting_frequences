"""
Pytest Configuration and Shared Fixtures
Provides common test fixtures, utilities, and configuration for all tests.
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Tuple

from src.data import SignalGenerator, SignalConfig, FrequencyExtractionDataset
from src.models import StatefulLSTMExtractor, create_model
from src.training.trainer import LSTMTrainer
from src.data.dataset import StatefulDataLoader, create_dataloaders


# ============================================================================
# Session-Level Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def device():
    """Get device for testing (prefer CPU for reproducibility)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def test_seed():
    """Fixed random seed for reproducible tests."""
    return 42


@pytest.fixture(scope="session")
def set_random_seeds(test_seed):
    """Set all random seeds for reproducibility."""
    np.random.seed(test_seed)
    torch.manual_seed(test_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(test_seed)
    return test_seed


# ============================================================================
# Temporary Directory Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def experiment_dir(temp_dir):
    """Create experiment directory structure."""
    exp_dir = temp_dir / "test_experiment"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "plots").mkdir(exist_ok=True)
    (exp_dir / "tensorboard").mkdir(exist_ok=True)
    
    yield exp_dir


# ============================================================================
# Signal Generation Fixtures
# ============================================================================

@pytest.fixture
def minimal_signal_config(test_seed):
    """Minimal configuration for fast testing."""
    return SignalConfig(
        frequencies=[1.0, 3.0],
        sampling_rate=100,
        duration=1.0,
        amplitude_range=(0.8, 1.2),
        phase_range=(0, 2*np.pi),
        seed=test_seed
    )


@pytest.fixture
def standard_signal_config(test_seed):
    """Standard configuration matching assignment specs."""
    return SignalConfig(
        frequencies=[1.0, 3.0, 5.0, 7.0],
        sampling_rate=1000,
        duration=10.0,
        amplitude_range=(0.8, 1.2),
        phase_range=(0, 2*np.pi),
        seed=test_seed
    )


@pytest.fixture
def edge_case_signal_config(test_seed):
    """Configuration for edge case testing."""
    return SignalConfig(
        frequencies=[0.1, 100.0],  # Very low and high frequencies
        sampling_rate=1000,
        duration=2.0,
        amplitude_range=(0.8, 1.2),
        phase_range=(0, 2*np.pi),
        seed=test_seed
    )


@pytest.fixture
def minimal_generator(minimal_signal_config):
    """Create minimal signal generator for fast tests."""
    return SignalGenerator(minimal_signal_config)


@pytest.fixture
def standard_generator(standard_signal_config):
    """Create standard signal generator."""
    return SignalGenerator(standard_signal_config)


# ============================================================================
# Dataset Fixtures
# ============================================================================

@pytest.fixture
def minimal_dataset(minimal_generator):
    """Create minimal dataset for fast testing."""
    return FrequencyExtractionDataset(
        minimal_generator,
        normalize=True,
        device='cpu'
    )


@pytest.fixture
def standard_dataset(standard_generator):
    """Create standard dataset."""
    return FrequencyExtractionDataset(
        standard_generator,
        normalize=True,
        device='cpu'
    )


@pytest.fixture
def minimal_dataloader(minimal_dataset):
    """Create minimal dataloader."""
    return StatefulDataLoader(
        minimal_dataset,
        batch_size=16,
        shuffle_frequencies=False
    )


@pytest.fixture
def minimal_train_loader(minimal_dataset):
    """Create minimal train loader (alias for compatibility)."""
    return StatefulDataLoader(
        minimal_dataset,
        batch_size=16,
        shuffle_frequencies=False
    )


# ============================================================================
# Model Fixtures
# ============================================================================

@pytest.fixture
def minimal_model_config():
    """Minimal model configuration for fast testing."""
    return {
        'input_size': 3,  # For 2 frequencies: S[t] + 2 one-hot
        'hidden_size': 32,
        'num_layers': 1,
        'output_size': 1,
        'dropout': 0.1,
        'bidirectional': False
    }


@pytest.fixture
def standard_model_config():
    """Standard model configuration matching assignment."""
    return {
        'input_size': 5,  # For 4 frequencies: S[t] + 4 one-hot
        'hidden_size': 128,
        'num_layers': 2,
        'output_size': 1,
        'dropout': 0.2,
        'bidirectional': False
    }


@pytest.fixture
def minimal_model(minimal_model_config, device):
    """Create minimal model for fast testing."""
    model = create_model(minimal_model_config)
    return model.to(device)


@pytest.fixture
def standard_model(standard_model_config, device):
    """Create standard model."""
    model = create_model(standard_model_config)
    return model.to(device)


# ============================================================================
# Training Configuration Fixtures
# ============================================================================

@pytest.fixture
def minimal_training_config():
    """Minimal training configuration for fast tests."""
    return {
        'epochs': 2,
        'batch_size': 16,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'scheduler': None,
        'gradient_clip_value': 1.0,
        'early_stopping_patience': 10,
        'min_delta': 1e-6,
        'log_frequency': 10
    }


@pytest.fixture
def standard_training_config():
    """Standard training configuration."""
    return {
        'epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'weight_decay': 1e-5,
        'scheduler': 'reduce_on_plateau',
        'scheduler_factor': 0.5,
        'scheduler_patience': 5,
        'gradient_clip_value': 1.0,
        'early_stopping_patience': 10,
        'min_delta': 1e-6,
        'log_frequency': 100
    }


# ============================================================================
# Trainer Fixture
# ============================================================================

@pytest.fixture
def minimal_trainer(minimal_model, minimal_dataloader, minimal_training_config, device, experiment_dir):
    """Create minimal trainer for testing."""
    return LSTMTrainer(
        model=minimal_model,
        train_loader=minimal_dataloader,
        val_loader=minimal_dataloader,  # Use same for validation in tests
        config=minimal_training_config,
        device=device,
        experiment_dir=experiment_dir / "checkpoints"
    )


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def sample_batch(minimal_dataset):
    """Create a sample batch for testing."""
    batch_size = 16
    inputs = []
    targets = []
    
    for i in range(batch_size):
        inp, tgt = minimal_dataset[i]
        inputs.append(inp)
        targets.append(tgt)
    
    return {
        'input': torch.stack(inputs),
        'target': torch.stack(targets),
        'freq_idx': 0,
        'time_range': (0, batch_size),
        'is_first_batch': True,
        'is_last_batch': False
    }


@pytest.fixture
def sample_predictions_and_targets():
    """Create sample predictions and targets for metrics testing."""
    np.random.seed(42)
    targets = np.sin(np.linspace(0, 4*np.pi, 1000))
    predictions = targets + np.random.normal(0, 0.1, 1000)
    return predictions, targets


# ============================================================================
# Edge Case Data Fixtures
# ============================================================================

@pytest.fixture
def edge_case_data():
    """Provide edge case test data."""
    return {
        'empty_array': np.array([]),
        'single_value': np.array([1.0]),
        'nan_values': np.array([1.0, np.nan, 3.0]),
        'inf_values': np.array([1.0, np.inf, 3.0]),
        'all_zeros': np.zeros(100),
        'all_ones': np.ones(100),
        'very_large': np.array([1e10, 1e11, 1e12]),
        'very_small': np.array([1e-10, 1e-11, 1e-12]),
        'negative': np.array([-1.0, -2.0, -3.0]),
        'mixed': np.array([-1e10, 0, 1e10])
    }


@pytest.fixture
def invalid_configs():
    """Provide invalid configurations for testing error handling."""
    return {
        'negative_frequency': {
            'frequencies': [-1.0, 3.0],
            'sampling_rate': 1000,
            'duration': 10.0
        },
        'zero_frequency': {
            'frequencies': [0.0, 3.0],
            'sampling_rate': 1000,
            'duration': 10.0
        },
        'nyquist_violation': {
            'frequencies': [1.0, 600.0],  # 600 Hz with 1000 Hz sampling
            'sampling_rate': 1000,
            'duration': 10.0
        },
        'negative_duration': {
            'frequencies': [1.0, 3.0],
            'sampling_rate': 1000,
            'duration': -10.0
        },
        'zero_sampling_rate': {
            'frequencies': [1.0, 3.0],
            'sampling_rate': 0,
            'duration': 10.0
        },
        'duplicate_frequencies': {
            'frequencies': [1.0, 1.0, 3.0],
            'sampling_rate': 1000,
            'duration': 10.0
        },
        'empty_frequencies': {
            'frequencies': [],
            'sampling_rate': 1000,
            'duration': 10.0
        }
    }


# ============================================================================
# Helper Functions
# ============================================================================

def assert_tensor_properties(tensor: torch.Tensor, expected_shape: tuple, 
                            expected_dtype=torch.float32, check_finite=True):
    """Assert tensor has expected properties."""
    assert isinstance(tensor, torch.Tensor), f"Expected torch.Tensor, got {type(tensor)}"
    assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"
    assert tensor.dtype == expected_dtype, f"Expected dtype {expected_dtype}, got {tensor.dtype}"
    if check_finite:
        assert torch.isfinite(tensor).all(), "Tensor contains non-finite values"


def assert_array_properties(array: np.ndarray, expected_shape: tuple,
                           expected_dtype=np.float64, check_finite=True):
    """Assert numpy array has expected properties."""
    assert isinstance(array, np.ndarray), f"Expected np.ndarray, got {type(array)}"
    assert array.shape == expected_shape, f"Expected shape {expected_shape}, got {array.shape}"
    if expected_dtype is not None:
        assert array.dtype == expected_dtype, f"Expected dtype {expected_dtype}, got {array.dtype}"
    if check_finite:
        assert np.isfinite(array).all(), "Array contains non-finite values"


# ============================================================================
# Pytest Hooks
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "edge_case: marks tests that check edge cases"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add markers based on test names
        if "edge" in item.nodeid.lower():
            item.add_marker(pytest.mark.edge_case)
        if "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)
        if "performance" in item.nodeid.lower() or "stress" in item.nodeid.lower():
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
        if "slow" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)


# ============================================================================
# Test Utilities
# ============================================================================

class TestMetrics:
    """Utility class for tracking test metrics."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
    
    def record_result(self, outcome: str):
        """Record test outcome."""
        if outcome == 'passed':
            self.passed += 1
        elif outcome == 'failed':
            self.failed += 1
        elif outcome == 'skipped':
            self.skipped += 1
    
    def get_summary(self) -> Dict[str, int]:
        """Get test metrics summary."""
        return {
            'passed': self.passed,
            'failed': self.failed,
            'skipped': self.skipped,
            'total': self.passed + self.failed + self.skipped
        }


@pytest.fixture
def test_metrics():
    """Provide test metrics tracker."""
    return TestMetrics()

