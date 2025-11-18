# Coverage Fix for CI/CD

## Problem
The integration tests were passing but coverage was at 72.33%, below the required 85% threshold.

## Root Cause Analysis
The coverage report showed several files with low coverage:
1. **src/quality/__init__.py** - 0% (imports not exercised)
2. **src/visualization/plotter.py** - 23.56% (many methods not called)
3. **src/training/trainer.py** - 81.95% (SGD optimizer and some scheduler paths)
4. **src/data/dataset.py** - 84.47% (get_sequence and get_time_series_for_frequency methods)

## Solution Implemented

### 1. Added Quality Module Import Test
**File**: `tests/test_integration.py`
- Added `TestQualityModuleIntegration` class with `test_quality_module_imports()`
- This test imports all classes from `src.quality` module, covering the `__init__.py` file
- Ensures all quality module components are accessible

### 2. Added Dataset Advanced Methods Tests
**File**: `tests/test_integration.py`
- Added `TestDatasetAdvancedMethods` class
- **test_get_sequence()**: Tests the `get_sequence()` method for sequential data retrieval
- **test_get_time_series_for_frequency()**: Tests frequency-specific time series extraction
- These methods are used in visualization but weren't covered

### 3. Added Trainer Variant Tests
**File**: `tests/test_integration.py`
- Added `TestTrainerVariants` class with multiple optimizer and scheduler tests:
  - **test_trainer_with_sgd()**: Tests SGD optimizer path (lines 111-119)
  - **test_trainer_with_adamw()**: Tests AdamW optimizer path (lines 105-110)
  - **test_trainer_with_cosine_scheduler()**: Tests cosine annealing scheduler (lines 134-143)
  - **test_trainer_with_reduce_on_plateau_scheduler()**: Tests ReduceLROnPlateau scheduler (lines 127-133)
- These tests cover the previously untested optimizer and scheduler creation paths

### 4. Added Visualization Integration Test
**File**: `tests/test_integration.py`
- Added `TestVisualizationIntegration` class
- **test_create_all_visualizations_integration()**: 
  - Creates real model predictions for all frequencies
  - Calls `create_all_visualizations()` with complete data
  - Verifies all plots are generated
  - Covers lines 341-407 in plotter.py (the main visualization function)

### 5. Added matplotlib Backend Configuration
**File**: `tests/test_integration.py`
- Added `matplotlib.use('Agg')` at module level
- Ensures non-interactive backend is used in CI environment
- Prevents display-related errors during test execution

### 6. Added Missing Test Fixture
**File**: `tests/conftest.py`
- Added `minimal_train_loader` fixture (alias for `minimal_dataloader`)
- Ensures compatibility with new trainer variant tests

## Expected Coverage Improvement

Based on the new tests:
- **src/quality/__init__.py**: 0% → 100% (all 5 lines covered)
- **src/visualization/plotter.py**: 23.56% → ~70%+ (covering create_all_visualizations function)
- **src/training/trainer.py**: 81.95% → ~90%+ (covering all optimizer and scheduler paths)
- **src/data/dataset.py**: 84.47% → ~95%+ (covering get_sequence and get_time_series_for_frequency)

**Estimated Overall Coverage**: 72.33% → 85%+

## Test Count
- Added 9 new integration tests
- Total new test methods: 9
- All tests follow existing patterns and use established fixtures

## Changes Summary
1. **tests/test_integration.py**: Added 4 new test classes with 9 test methods
2. **tests/conftest.py**: Added 1 new fixture (minimal_train_loader)

## Verification
The tests follow the existing test patterns and should pass in CI. They test:
- ✅ Previously untested code paths
- ✅ Real integration scenarios
- ✅ Edge cases for configuration variants
- ✅ Complete workflow from data to visualization

## Next Steps
1. Push changes to trigger CI/CD pipeline
2. Verify coverage reaches 85%+ threshold
3. If any tests fail, investigate and fix
4. Monitor CI logs for any specific issues

## Notes
- All new tests are marked with `@pytest.mark.integration`
- Tests are deterministic and reproducible
- No external dependencies or network access required
- Tests use existing fixtures from conftest.py
- matplotlib configured for non-interactive backend in CI

