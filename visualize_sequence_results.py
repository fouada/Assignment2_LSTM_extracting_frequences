"""
Quick Visualization Script for Sequence Length Experiment Results

Usage:
    python visualize_sequence_results.py [--results-dir experiments/sequence_length_comparison]
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def load_results(results_dir: Path) -> dict:
    """Load results from JSON file."""
    results_file = results_dir / "results_summary.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'r') as f:
        return json.load(f)


def plot_quick_comparison(results: dict, output_dir: Path):
    """Create quick comparison plots."""
    
    # Convert string keys to integers and sort
    L_values = sorted([int(k) for k in results.keys()])
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Final MSE Comparison
    ax = axes[0, 0]
    train_mses = [results[str(L)]['train_mse'] for L in L_values]
    test_mses = [results[str(L)]['test_mse'] for L in L_values]
    
    x = np.arange(len(L_values))
    width = 0.35
    ax.bar(x - width/2, train_mses, width, label='Train', alpha=0.8)
    ax.bar(x + width/2, test_mses, width, label='Test', alpha=0.8)
    ax.set_xlabel('Sequence Length (L)', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('Final Performance by Sequence Length', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'L={L}' for L in L_values])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Training Time
    ax = axes[0, 1]
    training_times = [results[str(L)]['training_time'] for L in L_values]
    ax.bar(range(len(L_values)), training_times, alpha=0.8, color='coral')
    ax.set_xlabel('Sequence Length (L)', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(L_values)))
    ax.set_xticklabels([f'L={L}' for L in L_values])
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Convergence Speed
    ax = axes[1, 0]
    convergence_speeds = [results[str(L)]['convergence_speed'] for L in L_values]
    ax.bar(range(len(L_values)), convergence_speeds, alpha=0.8, color='green')
    ax.set_xlabel('Sequence Length (L)', fontsize=12)
    ax.set_ylabel('Epochs to Convergence', fontsize=12)
    ax.set_title('Convergence Speed', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(L_values)))
    ax.set_xticklabels([f'L={L}' for L in L_values])
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Generalization Gap
    ax = axes[1, 1]
    gaps = [results[str(L)]['test_mse'] - results[str(L)]['train_mse'] for L in L_values]
    colors = ['red' if g > 0 else 'blue' for g in gaps]
    ax.bar(range(len(L_values)), gaps, alpha=0.8, color=colors)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Sequence Length (L)', fontsize=12)
    ax.set_ylabel('Gap (Test - Train MSE)', fontsize=12)
    ax.set_title('Generalization Analysis', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(L_values)))
    ax.set_xticklabels([f'L={L}' for L in L_values])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = output_dir / "quick_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {plot_path}")
    plt.close()
    
    # Print summary table
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'L':<8} {'Train MSE':<12} {'Test MSE':<12} {'Gap':<12} {'Time(s)':<10} {'Conv.Epochs':<12}")
    print("-"*80)
    for L in L_values:
        r = results[str(L)]
        gap = r['test_mse'] - r['train_mse']
        print(f"{L:<8} {r['train_mse']:<12.6f} {r['test_mse']:<12.6f} "
              f"{gap:<12.6f} {r['training_time']:<10.2f} {r['convergence_speed']:<12}")
    print("="*80 + "\n")
    
    # Find best
    best_test = min([(L, results[str(L)]['test_mse']) for L in L_values], key=lambda x: x[1])
    fastest = min([(L, results[str(L)]['training_time']) for L in L_values], key=lambda x: x[1])
    
    print(f"✓ Best Test Performance: L={best_test[0]} (MSE={best_test[1]:.6f})")
    print(f"✓ Fastest Training: L={fastest[0]} ({fastest[1]:.2f}s)")
    print()


def main():
    parser = argparse.ArgumentParser(description='Visualize sequence length experiment results')
    parser.add_argument(
        '--results-dir',
        type=str,
        default='experiments/sequence_length_comparison',
        help='Directory containing results'
    )
    
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    print("Loading results...")
    try:
        results = load_results(results_dir)
        print(f"✓ Loaded results for {len(results)} experiments")
        
        print("\nGenerating visualizations...")
        plot_quick_comparison(results, results_dir)
        
        print("✓ Visualization complete!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure experiments have finished running.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

