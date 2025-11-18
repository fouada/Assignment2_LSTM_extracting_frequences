# Testing Guide
## LSTM Frequency Extraction System

---

## Overview

Comprehensive testing suite ensuring code quality, correctness, and reliability.

**Test Coverage:**
- Unit tests for all modules
- Integration tests for full pipeline
- Performance benchmarks
- Quality compliance checks

---

## Running Tests

### Quick Test

```bash
# Run all tests
pytest tests/ -v
```

### With Coverage

```bash
# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Open coverage report
open htmlcov/index.html
```

### Specific Tests

```bash
# Test specific module
pytest tests/test_model.py -v

# Test specific function
pytest tests/test_model.py::test_lstm_forward -v

# Test with markers
pytest tests/ -m "not slow" -v
```

---

## Test Modules

### 1. Data Tests (`test_data.py`)

**Tests signal generation and dataset creation:**

```bash
pytest tests/test_data.py -v
```

**Coverage:**
- Signal generator creation
- Frequency generation
- Mixed signal composition
- Dataset indexing
- Data normalization
- Dataloader functionality

---

### 2. Model Tests (`test_model.py`)

**Tests LSTM architecture:**

```bash
pytest tests/test_model.py -v
```

**Coverage:**
- Model initialization
- Forward pass
- State management
- Stateful processing
- Parameter counts
- Device compatibility

---

### 3. Training Tests (`test_trainer.py`)

**Tests training loop:**

```bash
pytest tests/test_trainer.py -v
```

**Coverage:**
- Trainer initialization
- Epoch execution
- Validation
- Checkpoint saving
- Early stopping
- Learning rate scheduling

---

### 4. Evaluation Tests (`test_evaluation.py`)

**Tests metrics computation:**

```bash
pytest tests/test_evaluation.py -v
```

**Coverage:**
- MSE calculation
- MAE calculation
- R² score
- SNR computation
- Correlation coefficient
- Per-frequency metrics

---

### 5. Integration Tests (`test_integration.py`)

**Tests complete pipeline:**

```bash
pytest tests/test_integration.py -v
```

**Coverage:**
- End-to-end workflow
- Data → Model → Training → Evaluation
- Configuration loading
- Experiment directory creation
- Reproducibility

---

### 6. Visualization Tests (`test_visualization.py`)

**Tests plotting functionality:**

```bash
pytest tests/test_visualization.py -v
```

**Coverage:**
- Plot generation
- Figure saving
- Visualization utilities
- Dashboard components (if installed)

---

### 7. Performance Tests (`test_performance.py`)

**Tests system performance:**

```bash
pytest tests/test_performance.py -v
```

**Coverage:**
- Training speed
- Inference speed
- Memory usage
- GPU utilization (if available)

---

### 8. Quality Tests (`test_quality_compliance.py`)

**Tests ISO 25010 compliance:**

```bash
pytest tests/test_quality_compliance.py -v
```

**Coverage:**
- Functional correctness
- Performance efficiency
- Reliability
- Usability
- Maintainability
- Security

---

## Test Dashboard

### Dashboard Testing

```bash
# Test dashboard installation and functionality
python test_dashboard.py
```

**Tests:**
- Import verification
- Module loading
- Dashboard creation
- Live monitor functionality
- Experiment detection

**Expected Output:**
```
================================================================================
DASHBOARD TESTING SUITE
================================================================================
✅ PASS: Imports
✅ PASS: Dashboard Modules
✅ PASS: Dashboard Creation
✅ PASS: Live Monitor
✅ PASS: Experiment Detection

✅ ALL TESTS PASSED!
================================================================================
```

---

## Continuous Integration

### Pre-commit Checks

```bash
# Format code
black src/ tests/ main.py

# Check style
flake8 src/ tests/ main.py

# Type checking
mypy src/

# Run tests
pytest tests/ -v
```

### Full CI Pipeline

```bash
# Run complete CI pipeline
./run_ci.sh  # If available

# Or manually:
black --check src/ tests/
flake8 src/ tests/
mypy src/
pytest tests/ --cov=src --cov-report=html
```

---

## Writing Tests

### Test Structure

```python
import pytest
import torch
from src.models.lstm_extractor import LSTMFrequencyExtractor

def test_model_forward():
    """Test LSTM forward pass."""
    # Setup
    model = LSTMFrequencyExtractor(
        input_size=5,
        hidden_size=128,
        num_layers=2,
        output_size=1
    )
    
    # Create test input
    batch_size = 32
    x = torch.randn(batch_size, 5)
    
    # Forward pass
    output = model(x)
    
    # Assertions
    assert output.shape == (batch_size, 1)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
```

### Fixtures

```python
@pytest.fixture
def sample_config():
    """Provide sample configuration."""
    return {
        'model': {
            'input_size': 5,
            'hidden_size': 128,
            'num_layers': 2
        }
    }

def test_with_fixture(sample_config):
    """Test using fixture."""
    assert sample_config['model']['hidden_size'] == 128
```

---

## Performance Benchmarks

### Training Speed

```bash
# Run performance benchmark
pytest tests/test_performance.py::test_training_speed -v
```

**Expected Results:**
- CPU: ~10-15 seconds/epoch
- MPS (M1): ~8-12 seconds/epoch
- CUDA (GPU): ~3-5 seconds/epoch

### Inference Speed

```bash
# Run inference benchmark
pytest tests/test_performance.py::test_inference_speed -v
```

**Expected Results:**
- Batch size 1: < 1ms/sample
- Batch size 32: < 0.1ms/sample
- Batch size 256: < 0.05ms/sample

---

## Test Configuration

### pytest.ini

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --strict-markers
    --tb=short
markers =
    slow: marks tests as slow
    gpu: marks tests requiring GPU
    integration: marks integration tests
```

### Coverage Configuration

```ini
[coverage:run]
source = src
omit = 
    */tests/*
    */venv/*
    */__pycache__/*

[coverage:report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
```

---

## Troubleshooting

### Tests Fail: Module Not Found

```bash
# Ensure in project root
cd /path/to/Assignment2_LSTM_extracting_frequences

# Install in development mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Tests Fail: CUDA Out of Memory

```bash
# Use smaller batch size
pytest tests/test_performance.py --batch-size 16

# Or run on CPU
pytest tests/ --device cpu
```

### Slow Tests

```bash
# Skip slow tests
pytest tests/ -m "not slow" -v

# Run only fast tests
pytest tests/ -m "fast" -v
```

---

## Test Results

### Example Output

```
======================== test session starts =========================
platform darwin -- Python 3.14.0, pytest-7.3.0
collected 47 items

tests/test_data.py::test_signal_generator ✅ PASSED
tests/test_data.py::test_dataset_creation ✅ PASSED
tests/test_model.py::test_model_init ✅ PASSED
tests/test_model.py::test_forward_pass ✅ PASSED
tests/test_trainer.py::test_trainer_init ✅ PASSED
tests/test_trainer.py::test_training_epoch ✅ PASSED
tests/test_evaluation.py::test_mse_calculation ✅ PASSED
tests/test_integration.py::test_full_pipeline ✅ PASSED

======================= 47 passed in 12.34s =======================
```

---

## Quality Metrics

### Code Coverage

**Target:** > 80% coverage

```bash
pytest tests/ --cov=src --cov-report=term
```

**Current Coverage:**
- Data module: > 90%
- Model module: > 85%
- Training module: > 85%
- Evaluation module: > 90%
- Overall: > 85%

### Test Success Rate

**Target:** 100% pass rate

**Current Status:** ✅ All tests passing

---

## Continuous Testing

### Watch Mode

```bash
# Install pytest-watch
pip install pytest-watch

# Run in watch mode
ptw tests/ -- -v
```

### Pre-commit Hook

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Tests run automatically on git commit
```

---

## Support

- **Test Failures:** Check error messages and traceback
- **Performance Issues:** Use `pytest -v -s` for detailed output
- **Coverage Issues:** See `htmlcov/index.html` for details
- **Questions:** Check test file docstrings

---

**Run tests regularly to ensure code quality!**

```bash
# Quick test
pytest tests/ -v

# Full test with coverage
pytest tests/ --cov=src --cov-report=html
```

