"""
Generate Required Assignment Graphs (Section 5.2)

Creates the two graphs specified in the assignment:
1. Single frequency comparison (Target + LSTM Output + Mixed Signal)
2. Four subplot grid showing all frequencies

Usage:
    python generate_assignment_graphs.py [--model-path best_model_L1.pt]
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
from pathlib import Path

sys.path.insert(0, 'src')

from src.data.signal_generator import create_train_test_generators
from src.data.dataset import FrequencyExtractionDataset
from src.models.lstm_extractor import StatefulLSTMExtractor


def load_model(model_path: Path, device: torch.device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model with saved config
    if 'config' in checkpoint and 'model' in checkpoint['config']:
        model_config = checkpoint['config']['model']
    else:
        # Default config
        model_config = {
            'input_size': 5,
            'hidden_size': 128,
            'num_layers': 2,
            'output_size': 1,
            'dropout': 0.2,
            'bidirectional': False
        }
    
    model = StatefulLSTMExtractor(**model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded from {model_path}")
    print(f"  Hidden size: {model_config['hidden_size']}")
    print(f"  Num layers: {model_config['num_layers']}")
    
    return model


def generate_predictions(model, dataset, device, num_samples=1000):
    """
    Generate predictions for visualization.
    
    Args:
        model: Trained LSTM model
        dataset: Test dataset
        device: Computation device
        num_samples: Number of samples to use (default: 1000 for graphs)
    
    Returns:
        Dictionary with predictions, targets, and mixed signal for each frequency
    """
    print(f"\nGenerating predictions for {num_samples} samples...")
    
    results = {
        'frequencies': dataset.generator.frequencies,
        'time': dataset.generator.time_vector[:num_samples],
        'mixed_signal': dataset.mixed_signal[:num_samples],
        'predictions': {},
        'targets': {}
    }
    
    # Generate predictions for each frequency
    for freq_idx in range(dataset.num_frequencies):
        print(f"  Processing frequency {freq_idx+1}/4 ({dataset.generator.frequencies[freq_idx]} Hz)...")
        
        predictions = []
        targets = []
        
        # Reset state for this frequency
        model.reset_state()
        
        with torch.no_grad():
            for t in range(num_samples):
                # Get sample for this frequency and time
                idx = freq_idx * dataset.num_time_samples + t
                input_tensor, target_tensor = dataset[idx]
                
                # Add batch dimension
                input_batch = input_tensor.unsqueeze(0).to(device)
                
                # Forward pass (state preserved automatically)
                output = model(input_batch, reset_state=False)
                
                predictions.append(output.cpu().item())
                targets.append(target_tensor.item())
        
        results['predictions'][freq_idx] = np.array(predictions)
        results['targets'][freq_idx] = np.array(targets)
    
    print("✓ Predictions generated")
    return results


def create_graph1_single_frequency(results, freq_idx, output_path):
    """
    Create Graph 1: Single frequency comparison
    Shows Target (line), LSTM Output (dots), and Mixed Signal (background)
    """
    freq_names = ['1 Hz', '3 Hz', '5 Hz', '7 Hz']
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    time = results['time']
    mixed = results['mixed_signal']
    target = results['targets'][freq_idx]
    prediction = results['predictions'][freq_idx]
    
    # Plot mixed signal as background (chaotic)
    ax.plot(time, mixed, 'gray', alpha=0.3, linewidth=0.8, label='S (Mixed Noisy Signal)', zorder=1)
    
    # Plot target (pure sine, line)
    ax.plot(time, target, 'b-', linewidth=2, label=f'Target (Pure {freq_names[freq_idx]})', zorder=3)
    
    # Plot LSTM output (dots)
    # Use every Nth point for clearer visualization
    step = max(1, len(time) // 500)  # Show ~500 points
    ax.scatter(time[::step], prediction[::step], c='red', s=10, alpha=0.6, 
               label='LSTM Output', zorder=2)
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title(f'Frequency Extraction: {freq_names[freq_idx]} (Test Set)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(time[0], time[-1])
    
    # Add statistics
    mse = np.mean((prediction - target) ** 2)
    mae = np.mean(np.abs(prediction - target))
    textstr = f'MSE: {mse:.6f}\nMAE: {mae:.6f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Graph 1 saved to: {output_path}")
    plt.close()


def create_graph2_all_frequencies(results, output_path):
    """
    Create Graph 2: Four subplots showing all frequency extractions
    """
    freq_names = ['1 Hz', '3 Hz', '5 Hz', '7 Hz']
    colors = ['blue', 'green', 'orange', 'red']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    time = results['time']
    
    for freq_idx in range(4):
        ax = axes[freq_idx]
        
        target = results['targets'][freq_idx]
        prediction = results['predictions'][freq_idx]
        
        # Plot target (line)
        ax.plot(time, target, colors[freq_idx], linewidth=2, 
                label=f'Target ({freq_names[freq_idx]})', alpha=0.8)
        
        # Plot LSTM output (dots, sparser for clarity)
        step = max(1, len(time) // 300)
        ax.scatter(time[::step], prediction[::step], c='red', s=8, alpha=0.5, 
                   label='LSTM Output', zorder=3)
        
        # Calculate MSE
        mse = np.mean((prediction - target) ** 2)
        
        ax.set_xlabel('Time (seconds)', fontsize=10)
        ax.set_ylabel('Amplitude', fontsize=10)
        ax.set_title(f'Frequency {freq_idx+1}: {freq_names[freq_idx]} (MSE: {mse:.6f})', 
                     fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(time[0], time[-1])
    
    plt.suptitle('LSTM Frequency Extraction - All Frequencies (Test Set)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Graph 2 saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate assignment graphs')
    parser.add_argument(
        '--model-path',
        type=str,
        default='experiments/sequence_length_comparison/best_model_L1.pt',
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='assignment_graphs',
        help='Output directory for graphs'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=1000,
        help='Number of samples to visualize (default: 1000)'
    )
    parser.add_argument(
        '--selected-freq',
        type=int,
        default=1,
        help='Frequency index for Graph 1 (0-3, default: 1 for 3Hz)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("ASSIGNMENT GRAPH GENERATION")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_path}")
    print(f"  Output: {output_dir}")
    print(f"  Samples: {args.num_samples}")
    print(f"  Selected frequency for Graph 1: {args.selected_freq}")
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"  Device: {device}")
    
    # Load model
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"\n❌ Error: Model not found at {model_path}")
        print(f"   Available models:")
        for p in Path('experiments/sequence_length_comparison').glob('best_model_L*.pt'):
            print(f"     - {p}")
        return
    
    model = load_model(model_path, device)
    
    # Create test dataset
    print("\nCreating test dataset...")
    _, test_gen = create_train_test_generators(
        frequencies=[1.0, 3.0, 5.0, 7.0],
        sampling_rate=1000,
        duration=10.0,
        train_seed=1,
        test_seed=2
    )
    
    test_dataset = FrequencyExtractionDataset(
        test_gen,
        normalize=True,
        device=str(device)
    )
    print(f"✓ Test dataset created: {len(test_dataset)} samples")
    
    # Generate predictions
    results = generate_predictions(model, test_dataset, device, args.num_samples)
    
    # Create Graph 1: Single frequency comparison
    print("\nGenerating Graph 1: Single Frequency Comparison...")
    graph1_path = output_dir / "graph1_single_frequency_comparison.png"
    create_graph1_single_frequency(results, args.selected_freq, graph1_path)
    
    # Create Graph 2: All frequencies
    print("\nGenerating Graph 2: All Frequencies...")
    graph2_path = output_dir / "graph2_all_frequencies.png"
    create_graph2_all_frequencies(results, graph2_path)
    
    # Summary statistics
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    freq_names = ['1 Hz', '3 Hz', '5 Hz', '7 Hz']
    print(f"\n{'Frequency':<15} {'MSE':<12} {'MAE':<12} {'R²':<12}")
    print("-"*51)
    
    total_mse = 0
    for freq_idx in range(4):
        pred = results['predictions'][freq_idx]
        target = results['targets'][freq_idx]
        
        mse = np.mean((pred - target) ** 2)
        mae = np.mean(np.abs(pred - target))
        
        # R² score
        ss_res = np.sum((target - pred) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        print(f"{freq_names[freq_idx]:<15} {mse:<12.6f} {mae:<12.6f} {r2:<12.6f}")
        total_mse += mse
    
    avg_mse = total_mse / 4
    print("-"*51)
    print(f"{'Average':<15} {avg_mse:<12.6f}")
    
    print("\n" + "="*70)
    print("✓ GRAPH GENERATION COMPLETE!")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  1. {graph1_path}")
    print(f"  2. {graph2_path}")
    print(f"\nThese graphs fulfill Section 5.2 requirements of the assignment.")
    print("Ready for submission! 🎉")


if __name__ == "__main__":
    main()

