# How to Run Tests

## Quick Test Commands

### Run All Tests with Coverage
```bash
pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=85
```

### Run Only New Sequence Dataset Tests
```bash
pytest tests/test_sequence_dataset.py -v
```

### Run Model Tests (Including New Coverage)
```bash
pytest tests/test_model.py -v
```

### Generate HTML Coverage Report
```bash
pytest tests/ --cov=src --cov-report=html
# Then open htmlcov/index.html in browser
```

## Expected Test Results

### Before Fix
- **Tests**: 199 passed
- **Coverage**: 84.50% (FAIL ❌)
- **Missing**: sequence_dataset.py (108 lines, 0% coverage)

### After Fix
- **Tests**: 225+ passed (199 + 23 new + 3 enhanced)
- **Coverage**: ~95%+ (PASS ✅)
- **New Coverage**: 
  - sequence_dataset.py: ~95%+
  - lstm_extractor.py: ~99%+

## Test File Structure

```
tests/
├── test_data.py              # Original data tests
├── test_sequence_dataset.py  # NEW: Sequence dataset tests (23 tests)
├── test_model.py             # Enhanced with 3 new tests
├── test_trainer.py           # Original trainer tests
├── test_evaluation.py        # Original evaluation tests
├── test_visualization.py     # Original visualization tests
├── test_integration.py       # Original integration tests
├── test_performance.py       # Original performance tests
└── test_quality_compliance.py # Original quality tests
```

## What's Tested Now

### SequenceDataset Class
- ✅ Initialization (default and custom stride)
- ✅ Dataset length calculation
- ✅ __getitem__ method (shapes, metadata, one-hot encoding)
- ✅ Signal normalization
- ✅ Sequence indexing and stride
- ✅ Full time series extraction
- ✅ Edge cases (L=1, large stride, device parameter)

### SequenceDataLoader Class
- ✅ Initialization and configuration
- ✅ __len__ method
- ✅ __iter__ method (batching, structure)
- ✅ Shuffle functionality
- ✅ Drop_last behavior

### Factory Functions
- ✅ create_sequence_dataloaders()
- ✅ Custom stride configuration
- ✅ Normalization control
- ✅ Train/test differentiation

### StatefulLSTMExtractor (New Coverage)
- ✅ Forward with return_state=True
- ✅ predict_sequence() method
- ✅ predict_sequence() with reset

## CI/CD Integration

The tests will run automatically on:
- Push to any branch
- Pull request creation
- Manual workflow dispatch

The GitHub Actions workflow will:
1. Set up Python 3.11
2. Install dependencies
3. Run all 225+ tests
4. Generate coverage report
5. Fail if coverage < 85%
6. Upload coverage to codecov (if configured)

## Local Development

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Tests Locally
```bash
# All tests
pytest

# With coverage
pytest --cov=src

# Specific test file
pytest tests/test_sequence_dataset.py -v

# Specific test class
pytest tests/test_sequence_dataset.py::TestSequenceDataset -v

# Specific test method
pytest tests/test_sequence_dataset.py::TestSequenceDataset::test_dataset_initialization -v
```

### Debug Failing Tests
```bash
# Show print statements
pytest tests/test_sequence_dataset.py -v -s

# Drop into debugger on failure
pytest tests/test_sequence_dataset.py --pdb

# Show locals on failure
pytest tests/test_sequence_dataset.py -v -l
```

## Coverage Analysis

### View Uncovered Lines
```bash
pytest --cov=src --cov-report=term-missing
```

### Generate Detailed Report
```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

### Check Specific Module
```bash
pytest --cov=src.data.sequence_dataset --cov-report=term-missing
```

## Troubleshooting

### ImportError
If you see `ModuleNotFoundError: No module named 'src'`:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest
```

### Coverage Not Counting
Make sure `pytest.ini` doesn't exclude the file you're testing.

### Tests Pass Locally But Fail in CI
Check Python version compatibility:
```bash
python --version  # Should be 3.11.x
```

## Performance

Expected test execution time:
- New sequence_dataset tests: ~5-10 seconds
- All tests: ~40-60 seconds
- With coverage: ~60-90 seconds

