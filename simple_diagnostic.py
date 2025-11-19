"""
Simple Diagnostic Without PyTorch
Checks data generation and identifies noise issues.
"""

import numpy as np
import yaml


def test_signal_parameters():
    """Test the configuration parameters."""
    print("\n" + "="*80)
    print("CONFIGURATION ANALYSIS")
    print("="*80)
    
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    amp_range = config['data']['amplitude_range']
    phase_range = config['data']['phase_range']
    
    amp_variation = ((amp_range[1] - amp_range[0]) / 2) * 100
    phase_variation_rad = (phase_range[1] - phase_range[0])
    
    print(f"\nNoise Parameters:")
    print(f"  Amplitude range: {amp_range}")
    print(f"  → Variation: ±{amp_variation:.1f}%")
    
    print(f"\n  Phase range: {phase_range}")
    print(f"  → Variation: {phase_variation_rad:.3f} radians ({np.degrees(phase_variation_rad):.1f}°)")
    
    print(f"\nTraining Parameters:")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Epochs: {config['training']['epochs']}")
    
    # Analysis
    print("\n" + "-"*80)
    print("ANALYSIS:")
    
    if amp_variation > 10:
        print(f"❌ CRITICAL: Amplitude noise is {amp_variation:.1f}% - TOO HIGH!")
        print("   → Model will struggle to learn with >10% noise")
        print("   → Recommended: [0.95, 1.05] for 5% noise")
    elif amp_variation > 5:
        print(f"⚠  WARNING: Amplitude noise is {amp_variation:.1f}% - HIGH")
        print("   → May need more epochs to converge")
    else:
        print(f"✅ GOOD: Amplitude noise is {amp_variation:.1f}% - reasonable")
    
    if phase_variation_rad > np.pi:
        print(f"❌ CRITICAL: Phase noise covers {np.degrees(phase_variation_rad):.0f}° - FULL RANGE!")
        print("   → This creates completely random signals")
        print("   → Consider reducing to [0, π/4] or [0, π/2]")
    else:
        print(f"⚠  Phase noise: {np.degrees(phase_variation_rad):.0f}°")
    
    return config


def test_signal_generation(config):
    """Test actual signal generation."""
    print("\n" + "="*80)
    print("SIGNAL GENERATION TEST")
    print("="*80)
    
    # Simple sine wave with noise
    frequencies = config['data']['frequencies']
    sampling_rate = config['data']['sampling_rate']
    duration = 1.0  # Just 1 second for testing
    
    amp_range = config['data']['amplitude_range']
    phase_range = config['data']['phase_range']
    
    t = np.linspace(0, duration, int(sampling_rate * duration))
    
    for freq in frequencies:
        # Generate noisy sine
        np.random.seed(42)
        amplitudes = np.random.uniform(amp_range[0], amp_range[1], size=len(t))
        phases = np.random.uniform(phase_range[0], phase_range[1], size=len(t))
        
        noisy_sine = amplitudes * np.sin(2 * np.pi * freq * t + phases)
        pure_sine = np.sin(2 * np.pi * freq * t)
        
        # Calculate correlation
        correlation = np.corrcoef(noisy_sine, pure_sine)[0, 1]
        
        print(f"\nFrequency: {freq} Hz")
        print(f"  Noisy signal range: [{noisy_sine.min():.3f}, {noisy_sine.max():.3f}]")
        print(f"  Pure signal range: [{pure_sine.min():.3f}, {pure_sine.max():.3f}]")
        print(f"  Correlation: {correlation:.4f}")
        
        if correlation < 0.5:
            print(f"  ❌ CRITICAL: Correlation too low - noise destroying signal!")
        elif correlation < 0.7:
            print(f"  ⚠  WARNING: Low correlation - high noise")
        else:
            print(f"  ✅ GOOD: Signal preserved despite noise")
    
    # Test mixed signal
    print(f"\n" + "-"*80)
    print("Mixed Signal Test:")
    
    np.random.seed(42)
    mixed = np.zeros(len(t))
    for freq in frequencies:
        amplitudes = np.random.uniform(amp_range[0], amp_range[1], size=len(t))
        phases = np.random.uniform(phase_range[0], phase_range[1], size=len(t))
        mixed += amplitudes * np.sin(2 * np.pi * freq * t + phases)
    
    mixed /= len(frequencies)
    
    print(f"  Mixed signal mean: {mixed.mean():.6f} (should be ~0)")
    print(f"  Mixed signal std: {mixed.std():.6f}")
    print(f"  Mixed signal range: [{mixed.min():.3f}, {mixed.max():.3f}]")
    
    if np.abs(mixed.mean()) > 0.1:
        print(f"  ⚠  WARNING: Mean far from zero")
    
    if mixed.std() < 0.01:
        print(f"  ❌ CRITICAL: Signal variance too low - may have collapsed!")


def test_normalization_effect(config):
    """Test how normalization affects the signal."""
    print("\n" + "="*80)
    print("NORMALIZATION EFFECT TEST")
    print("="*80)
    
    sampling_rate = config['data']['sampling_rate']
    duration = 1.0
    t = np.linspace(0, duration, int(sampling_rate * duration))
    
    # Generate a sample signal
    np.random.seed(42)
    frequencies = config['data']['frequencies']
    amp_range = config['data']['amplitude_range']
    phase_range = config['data']['phase_range']
    
    mixed = np.zeros(len(t))
    for freq in frequencies:
        amplitudes = np.random.uniform(amp_range[0], amp_range[1], size=len(t))
        phases = np.random.uniform(phase_range[0], phase_range[1], size=len(t))
        mixed += amplitudes * np.sin(2 * np.pi * freq * t + phases)
    mixed /= len(frequencies)
    
    # Apply normalization
    mean = np.mean(mixed)
    std = np.std(mixed)
    normalized = (mixed - mean) / (std + 1e-8)
    
    print(f"Before normalization:")
    print(f"  Mean: {mean:.6f}")
    print(f"  Std: {std:.6f}")
    print(f"  Range: [{mixed.min():.3f}, {mixed.max():.3f}]")
    
    print(f"\nAfter normalization:")
    print(f"  Mean: {normalized.mean():.6f}")
    print(f"  Std: {normalized.std():.6f}")
    print(f"  Range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    # Check if normalization is reasonable
    if std < 0.1:
        print(f"\n  ❌ CRITICAL: Original std too low - normalization may amplify noise!")
    else:
        print(f"\n  ✅ Normalization looks reasonable")


def provide_recommendations(config):
    """Provide actionable recommendations."""
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    amp_range = config['data']['amplitude_range']
    amp_variation = ((amp_range[1] - amp_range[0]) / 2) * 100
    lr = config['training']['learning_rate']
    
    recommendations = []
    
    if amp_variation > 10:
        recommendations.append({
            'priority': 'CRITICAL',
            'issue': f'Amplitude noise too high ({amp_variation:.1f}%)',
            'fix': 'Change amplitude_range to [0.95, 1.05]',
            'config_path': 'data.amplitude_range'
        })
    
    if lr > 0.0005:
        recommendations.append({
            'priority': 'HIGH',
            'issue': f'Learning rate may be too high ({lr})',
            'fix': 'Reduce learning_rate to 0.0001',
            'config_path': 'training.learning_rate'
        })
    
    if config['training']['batch_size'] < 64:
        recommendations.append({
            'priority': 'MEDIUM',
            'issue': 'Small batch size may cause unstable gradients',
            'fix': 'Increase batch_size to 64 or 128',
            'config_path': 'training.batch_size'
        })
    
    if len(recommendations) == 0:
        print("✅ Configuration looks good! If model still fails:")
        print("   1. Check that PyTorch is installed")
        print("   2. Run full diagnostic: python diagnostic_test.py")
        print("   3. Try training for more epochs")
    else:
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. [{rec['priority']}] {rec['issue']}")
            print(f"   Fix: {rec['fix']}")
            print(f"   Location: {rec['config_path']}")
    
    print("\n" + "="*80)


def main():
    print("\n" + "#"*80)
    print("SIMPLE DIAGNOSTIC - NO PYTORCH REQUIRED")
    print("#"*80)
    
    config = test_signal_parameters()
    test_signal_generation(config)
    test_normalization_effect(config)
    provide_recommendations(config)
    
    print("\n✅ Diagnostic complete!")
    print("\nNext steps:")
    print("  1. Apply recommended fixes to config/config.yaml")
    print("  2. Install PyTorch: pip install torch")
    print("  3. Run full diagnostic: python diagnostic_test.py")
    print("  4. Train the model: python main.py")


if __name__ == '__main__':
    main()
