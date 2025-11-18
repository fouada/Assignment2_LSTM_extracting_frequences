#!/usr/bin/env python3
"""
Dashboard Testing Script
Verify dashboard components and functionality

Author: Professional ML Engineering Team
Date: 2025
"""

import sys
from pathlib import Path

def test_imports():
    """Test if all required imports are available."""
    print("Testing imports...")
    
    try:
        import dash
        print(f"✅ dash {dash.__version__}")
    except ImportError as e:
        print(f"❌ dash not found: {e}")
        return False
    
    try:
        import dash_bootstrap_components as dbc
        print(f"✅ dash-bootstrap-components {dbc.__version__}")
    except ImportError as e:
        print(f"❌ dash-bootstrap-components not found: {e}")
        return False
    
    try:
        import plotly
        print(f"✅ plotly {plotly.__version__}")
    except ImportError as e:
        print(f"❌ plotly not found: {e}")
        return False
    
    try:
        import kaleido
        print(f"✅ kaleido available")
    except ImportError as e:
        print(f"⚠️  kaleido not found (optional for export): {e}")
    
    return True


def test_dashboard_modules():
    """Test if dashboard modules can be imported."""
    print("\nTesting dashboard modules...")
    
    try:
        from src.visualization.interactive_dashboard import LSTMFrequencyDashboard, create_dashboard
        print("✅ Interactive dashboard module loaded")
    except ImportError as e:
        print(f"❌ Error loading dashboard module: {e}")
        return False
    
    try:
        from src.visualization.live_monitor import LiveTrainingMonitor, create_live_monitor
        print("✅ Live monitor module loaded")
    except ImportError as e:
        print(f"❌ Error loading live monitor module: {e}")
        return False
    
    return True


def test_dashboard_creation():
    """Test if dashboard can be created."""
    print("\nTesting dashboard creation...")
    
    try:
        from src.visualization.interactive_dashboard import create_dashboard
        
        # Create dashboard without experiment data
        dashboard = create_dashboard(experiment_dir=None, port=8050)
        print("✅ Dashboard instance created successfully")
        print(f"   - App title: {dashboard.app.title}")
        print(f"   - Port: {dashboard.port}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating dashboard: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_live_monitor():
    """Test if live monitor can be created."""
    print("\nTesting live monitor...")
    
    try:
        from src.visualization.live_monitor import LiveTrainingMonitor
        import tempfile
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir)
            monitor = LiveTrainingMonitor(exp_dir, update_interval=1.0)
            
            print("✅ Live monitor instance created successfully")
            print(f"   - Experiment dir: {exp_dir}")
            print(f"   - Update interval: {monitor.update_interval}s")
            
            # Test epoch update
            monitor.update_epoch(
                epoch=1,
                train_loss=0.01,
                val_loss=0.012,
                learning_rate=0.001
            )
            print("✅ Epoch update works")
            
            # Test test results update
            monitor.update_test_results(
                overall_metrics={'mse': 0.001, 'mae': 0.02},
                per_frequency_metrics={0: {'mse': 0.001}}
            )
            print("✅ Test results update works")
            
            # Get summary
            summary = monitor.get_training_summary()
            print(f"✅ Training summary retrieved: {summary['status']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing live monitor: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_experiment_detection():
    """Test if existing experiments can be detected."""
    print("\nTesting experiment detection...")
    
    experiments_dir = Path('experiments')
    
    if not experiments_dir.exists():
        print("⚠️  No experiments directory found")
        print("   Run 'python main.py' first to create experiments")
        return True
    
    exp_dirs = [d for d in experiments_dir.iterdir() 
                if d.is_dir() and d.name.startswith('lstm_frequency_extraction_')]
    
    if not exp_dirs:
        print("⚠️  No experiments found")
        print("   Run 'python main.py' first to create experiments")
        return True
    
    print(f"✅ Found {len(exp_dirs)} experiment(s):")
    for exp_dir in sorted(exp_dirs, key=lambda p: p.stat().st_mtime, reverse=True)[:3]:
        print(f"   - {exp_dir.name}")
        
        # Check for required files
        config_exists = (exp_dir / 'config.yaml').exists()
        plots_exist = (exp_dir / 'plots').exists()
        checkpoints_exist = (exp_dir / 'checkpoints').exists()
        
        print(f"     Config: {'✅' if config_exists else '❌'}")
        print(f"     Plots: {'✅' if plots_exist else '❌'}")
        print(f"     Checkpoints: {'✅' if checkpoints_exist else '❌'}")
    
    return True


def main():
    """Run all tests."""
    print("="*80)
    print("DASHBOARD TESTING SUITE")
    print("="*80)
    
    results = []
    
    # Test 1: Imports
    results.append(("Imports", test_imports()))
    
    # Test 2: Dashboard modules
    if results[-1][1]:  # Only if imports work
        results.append(("Dashboard Modules", test_dashboard_modules()))
    
    # Test 3: Dashboard creation
    if results[-1][1]:  # Only if modules load
        results.append(("Dashboard Creation", test_dashboard_creation()))
    
    # Test 4: Live monitor
    if results[-1][1]:  # Only if dashboard creation works
        results.append(("Live Monitor", test_live_monitor()))
    
    # Test 5: Experiment detection
    results.append(("Experiment Detection", test_experiment_detection()))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        print("\nDashboard is ready to use!")
        print("\nNext steps:")
        print("1. Run: python dashboard.py")
        print("2. Open: http://localhost:8050")
        print("3. Enjoy interactive visualizations!")
    else:
        print("❌ SOME TESTS FAILED")
        print("="*80)
        print("\nPlease install missing dependencies:")
        print("pip install -r requirements.txt")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())

