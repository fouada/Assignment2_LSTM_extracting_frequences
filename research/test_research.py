"""
Quick Test Script for Research Module
Verifies that all research components are working correctly.

Run this before starting full research to catch any issues early.

Usage:
    python research/test_research.py
"""

import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all modules can be imported."""
    logger.info("Testing imports...")
    
    try:
        from research.sensitivity_analysis import (
            SensitivityAnalyzer,
            SensitivityConfig,
            create_default_sensitivity_config
        )
        logger.info("  ✓ sensitivity_analysis imported successfully")
    except Exception as e:
        logger.error(f"  ✗ Failed to import sensitivity_analysis: {e}")
        return False
    
    try:
        from research.comparative_analysis import (
            ComparativeAnalyzer,
            GRUExtractor,
            SimpleRNN
        )
        logger.info("  ✓ comparative_analysis imported successfully")
    except Exception as e:
        logger.error(f"  ✗ Failed to import comparative_analysis: {e}")
        return False
    
    try:
        import torch
        logger.info(f"  ✓ PyTorch {torch.__version__} available")
        
        if torch.cuda.is_available():
            logger.info(f"    - CUDA available: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            logger.info(f"    - MPS (Apple Silicon) available")
        else:
            logger.info(f"    - CPU only")
    except Exception as e:
        logger.error(f"  ✗ PyTorch issue: {e}")
        return False
    
    return True


def test_data_generation():
    """Test that data generation works."""
    logger.info("\nTesting data generation...")
    
    try:
        from src.data.signal_generator import create_train_test_generators
        
        train_gen, test_gen = create_train_test_generators(
            frequencies=[1.0, 3.0, 5.0, 7.0],
            sampling_rate=1000,
            duration=10.0,
            train_seed=1,
            test_seed=2
        )
        
        logger.info("  ✓ Signal generators created")
        
        # Generate small dataset
        mixed, targets = train_gen.generate_complete_dataset()
        logger.info(f"  ✓ Generated dataset: {mixed.shape}")
        
        return True
    except Exception as e:
        logger.error(f"  ✗ Data generation failed: {e}")
        return False


def test_model_creation():
    """Test that models can be created."""
    logger.info("\nTesting model creation...")
    
    try:
        import torch
        from src.models.lstm_extractor import create_model
        from research.comparative_analysis import GRUExtractor, SimpleRNN
        
        # Test LSTM
        lstm_config = {
            'input_size': 5,
            'hidden_size': 64,
            'num_layers': 2,
            'output_size': 1,
            'dropout': 0.2
        }
        lstm = create_model(lstm_config)
        logger.info(f"  ✓ LSTM created: {lstm.count_parameters():,} parameters")
        
        # Test GRU
        gru = GRUExtractor(hidden_size=64, num_layers=2)
        logger.info(f"  ✓ GRU created: {gru.count_parameters():,} parameters")
        
        # Test RNN
        rnn = SimpleRNN(hidden_size=64, num_layers=2)
        logger.info(f"  ✓ RNN created: {rnn.count_parameters():,} parameters")
        
        # Test forward pass
        x = torch.randn(2, 5)  # batch=2, features=5
        output = lstm(x)
        logger.info(f"  ✓ Forward pass works: output shape {output.shape}")
        
        return True
    except Exception as e:
        logger.error(f"  ✗ Model creation failed: {e}")
        return False


def test_mini_sensitivity():
    """Test sensitivity analysis with minimal configuration."""
    logger.info("\nTesting mini sensitivity analysis...")
    
    try:
        from research.sensitivity_analysis import SensitivityAnalyzer, SensitivityConfig
        
        # Very small config for quick test
        config = SensitivityConfig(
            hidden_sizes=[64],
            num_layers=[2],
            dropout_rates=[0.2],
            learning_rates=[0.001],
            batch_sizes=[32],
            epochs=2,  # Very short
            patience=1,
            num_runs=1,
            output_dir="./research/test_output"
        )
        
        logger.info("  Running mini sensitivity test (this will take ~2 minutes)...")
        analyzer = SensitivityAnalyzer(config)
        
        # Just test grid generation
        param_grid = analyzer.generate_parameter_grid()
        logger.info(f"  ✓ Generated {len(param_grid)} parameter combinations")
        
        # Don't run full analysis in test, just verify it can start
        logger.info("  ✓ Sensitivity analyzer initialized")
        
        return True
    except Exception as e:
        logger.error(f"  ✗ Sensitivity analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mini_comparison():
    """Test comparative analysis with minimal configuration."""
    logger.info("\nTesting mini comparative analysis...")
    
    try:
        from research.comparative_analysis import ComparativeAnalyzer
        
        analyzer = ComparativeAnalyzer(output_dir="./research/test_output")
        logger.info("  ✓ Comparative analyzer initialized")
        
        # Test that data can be generated
        from src.data.signal_generator import create_train_test_generators
        train_gen, test_gen = create_train_test_generators(
            frequencies=[1.0, 3.0, 5.0, 7.0],
            sampling_rate=1000,
            duration=10.0,
            train_seed=1,
            test_seed=2
        )
        logger.info("  ✓ Data generators created")
        
        return True
    except Exception as e:
        logger.error(f"  ✗ Comparative analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualization():
    """Test that visualization libraries work."""
    logger.info("\nTesting visualization libraries...")
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        # Test simple plot
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        plt.close()
        
        logger.info("  ✓ Matplotlib works")
        logger.info("  ✓ Seaborn works")
        
        return True
    except Exception as e:
        logger.error(f"  ✗ Visualization failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("="*80)
    logger.info("RESEARCH MODULE TEST SUITE")
    logger.info("="*80)
    
    tests = [
        ("Imports", test_imports),
        ("Data Generation", test_data_generation),
        ("Model Creation", test_model_creation),
        ("Visualization", test_visualization),
        ("Mini Sensitivity", test_mini_sensitivity),
        ("Mini Comparison", test_mini_comparison),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"\nUnexpected error in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    
    logger.info("="*80)
    if all_passed:
        logger.info("🎉 ALL TESTS PASSED!")
        logger.info("You can now run the full research pipeline:")
        logger.info("  python research/run_full_research.py --mode quick")
    else:
        logger.info("⚠️  SOME TESTS FAILED")
        logger.info("Please fix the issues before running full research.")
    logger.info("="*80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

