"""
Standalone Data Generation Flow
Demonstrates signal generation without full training

Author: Fouad Azem & Tal Goldengorn
Purpose: Quick demo of data generation for screenshots and understanding
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.data.signal_generator import create_train_test_generators

def main():
    """Run standalone data generation demo."""
    
    print("="*80)
    print("STANDALONE DATA GENERATION DEMO")
    print("="*80)
    
    # Load config
    print("\n📋 Loading configuration...")
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print("✅ Configuration loaded")
    
    # Generate data
    print("\n🔧 Generating train and test signals...")
    train_gen, test_gen = create_train_test_generators(
        frequencies=config['data']['frequencies'],
        sampling_rate=config['data']['sampling_rate'],
        duration=config['data']['duration'],
        amplitude_range=config['data']['amplitude_range'],
        phase_range=config['data']['phase_range'],
        train_seed=config['data']['train_seed'],
        test_seed=config['data']['test_seed']
    )
    
    # Print info
    print("\n" + "="*80)
    print("DATA GENERATION SUMMARY")
    print("="*80)
    print(f"✅ Train generator created (seed={config['data']['train_seed']})")
    print(f"✅ Test generator created (seed={config['data']['test_seed']})")
    print(f"\n📊 Signal Properties:")
    print(f"   - Mixed signal shape: {train_gen.mixed_signal.shape}")
    print(f"   - Pure targets shape: {train_gen.pure_targets.shape}")
    print(f"   - Frequencies: {train_gen.frequencies} Hz")
    print(f"   - Sampling rate: {train_gen.sampling_rate} Hz")
    print(f"   - Duration: {train_gen.duration} seconds")
    print(f"   - Time samples: {len(train_gen.time)}")
    
    print(f"\n🔊 Signal Statistics (Train Set):")
    print(f"   - Mixed signal mean: {np.mean(train_gen.mixed_signal):.6f}")
    print(f"   - Mixed signal std: {np.std(train_gen.mixed_signal):.6f}")
    print(f"   - Mixed signal range: [{np.min(train_gen.mixed_signal):.3f}, {np.max(train_gen.mixed_signal):.3f}]")
    
    # Create visualization
    print("\n🎨 Creating visualizations...")
    output_dir = Path("demo_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Figure 1: Mixed Signal + All Pure Frequencies
    fig = plt.figure(figsize=(15, 10))
    
    # Plot mixed signal
    ax1 = plt.subplot(3, 2, 1)
    samples_to_plot = 2000
    ax1.plot(train_gen.time[:samples_to_plot], train_gen.mixed_signal[:samples_to_plot], 
             linewidth=1, alpha=0.7, color='gray')
    ax1.set_title("Mixed Noisy Signal (Train Set, seed=1)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, train_gen.time[samples_to_plot-1])
    
    # Plot each pure frequency
    frequency_colors = ['blue', 'green', 'red', 'purple']
    for i, (freq, color) in enumerate(zip(train_gen.frequencies, frequency_colors)):
        ax = plt.subplot(3, 2, i+2)
        ax.plot(train_gen.time[:samples_to_plot], train_gen.pure_targets[i, :samples_to_plot], 
                linewidth=1.5, color=color, label=f'{freq} Hz')
        ax.set_title(f"Pure Frequency: f{i+1} = {freq} Hz", fontsize=12, fontweight='bold')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(0, train_gen.time[samples_to_plot-1])
        ax.set_ylim(-1.5, 1.5)
    
    # Comparison plot
    ax6 = plt.subplot(3, 2, 6)
    ax6.plot(train_gen.time[:samples_to_plot], train_gen.mixed_signal[:samples_to_plot], 
             linewidth=1, alpha=0.5, color='gray', label='Mixed Signal')
    for i, (freq, color) in enumerate(zip(train_gen.frequencies, frequency_colors)):
        ax6.plot(train_gen.time[:samples_to_plot], train_gen.pure_targets[i, :samples_to_plot], 
                linewidth=0.8, alpha=0.6, color=color, label=f'{freq} Hz')
    ax6.set_title("All Signals Overlaid", fontsize=12, fontweight='bold')
    ax6.set_xlabel("Time (s)")
    ax6.set_ylabel("Amplitude")
    ax6.grid(True, alpha=0.3)
    ax6.legend(loc='upper right', fontsize=8)
    ax6.set_xlim(0, train_gen.time[samples_to_plot-1])
    
    plt.tight_layout()
    output_path_1 = output_dir / "data_generation_demo.png"
    plt.savefig(output_path_1, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path_1}")
    plt.close()
    
    # Figure 2: Train vs Test comparison (showing different seeds)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    samples_to_plot = 1000
    
    # Train mixed signal
    axes[0, 0].plot(train_gen.time[:samples_to_plot], train_gen.mixed_signal[:samples_to_plot], 
                    linewidth=1, color='blue', alpha=0.7)
    axes[0, 0].set_title("Train Set: Mixed Signal (seed=1)", fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Test mixed signal
    axes[0, 1].plot(test_gen.time[:samples_to_plot], test_gen.mixed_signal[:samples_to_plot], 
                    linewidth=1, color='red', alpha=0.7)
    axes[0, 1].set_title("Test Set: Mixed Signal (seed=2)", fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Train pure frequency (f2 = 3Hz)
    axes[1, 0].plot(train_gen.time[:samples_to_plot], train_gen.pure_targets[1, :samples_to_plot], 
                    linewidth=1.5, color='blue')
    axes[1, 0].set_title("Train Set: Pure 3Hz (same for both)", fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Amplitude")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(-1.5, 1.5)
    
    # Test pure frequency (f2 = 3Hz) - should be identical to train
    axes[1, 1].plot(test_gen.time[:samples_to_plot], test_gen.pure_targets[1, :samples_to_plot], 
                    linewidth=1.5, color='red')
    axes[1, 1].set_title("Test Set: Pure 3Hz (same for both)", fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Amplitude")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(-1.5, 1.5)
    
    plt.tight_layout()
    output_path_2 = output_dir / "train_vs_test_comparison.png"
    plt.savefig(output_path_2, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path_2}")
    plt.close()
    
    # Figure 3: Noise visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    samples_to_plot = 500
    
    # Show one frequency with and without noise
    freq_idx = 1  # 3 Hz
    freq = train_gen.frequencies[freq_idx]
    
    # Noisy components
    noisy_component_train = train_gen.noisy_components[freq_idx, :samples_to_plot]
    pure_target = train_gen.pure_targets[freq_idx, :samples_to_plot]
    
    axes[0].plot(train_gen.time[:samples_to_plot], pure_target, 
                 linewidth=2, color='blue', label='Pure Target (no noise)', alpha=0.8)
    axes[0].plot(train_gen.time[:samples_to_plot], noisy_component_train, 
                 linewidth=1, color='red', label='Noisy Version (A & φ randomized)', alpha=0.6)
    axes[0].set_title(f"Effect of Random Amplitude and Phase (f={freq} Hz, seed=1)", 
                     fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Show noise (difference)
    noise = noisy_component_train - pure_target
    axes[1].plot(train_gen.time[:samples_to_plot], noise, 
                linewidth=1, color='orange', alpha=0.7)
    axes[1].set_title(f"Extracted Noise: Noisy - Pure (f={freq} Hz)", 
                     fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Noise Amplitude")
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    
    plt.tight_layout()
    output_path_3 = output_dir / "noise_visualization.png"
    plt.savefig(output_path_3, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path_3}")
    plt.close()
    
    print("\n" + "="*80)
    print("DATA GENERATION DEMO COMPLETE")
    print("="*80)
    print(f"📁 All outputs saved to: {output_dir}/")
    print("\n📸 Screenshot Recommendations:")
    print("   1. Screenshot this terminal output")
    print("   2. Open and screenshot: data_generation_demo.png")
    print("   3. Open and screenshot: train_vs_test_comparison.png")
    print("   4. Open and screenshot: noise_visualization.png")
    print("\n💡 These visualizations demonstrate:")
    print("   ✓ Mixed signal composition")
    print("   ✓ Individual frequency components")
    print("   ✓ Train vs Test set differences (different seeds)")
    print("   ✓ Effect of random amplitude and phase noise")
    print("="*80)


if __name__ == '__main__':
    main()

