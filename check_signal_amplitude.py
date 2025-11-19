import numpy as np
import sys
sys.path.append('.')
from src.data.signal_generator import create_train_test_generators

# Generate data
train_gen, test_gen = create_train_test_generators(
    frequencies=[1.0, 3.0, 5.0, 7.0],
    sampling_rate=1000,
    duration=10.0,
    train_seed=1,
    test_seed=2
)

# Get signals
mixed_signal_train, targets_train = train_gen.generate_complete_dataset()
mixed_signal_test, targets_test = test_gen.generate_complete_dataset()

print("="*80)
print("SIGNAL AMPLITUDE ANALYSIS")
print("="*80)

print("\n1. MIXED SIGNAL (S) - The noisy input:")
print("-"*80)
print(f"Train mixed signal:")
print(f"  Mean: {np.mean(mixed_signal_train):.6f}")
print(f"  Std:  {np.std(mixed_signal_train):.6f}")
print(f"  Min:  {np.min(mixed_signal_train):.6f}")
print(f"  Max:  {np.max(mixed_signal_train):.6f}")
print(f"  Range: {np.max(mixed_signal_train) - np.min(mixed_signal_train):.6f}")

print(f"\nTest mixed signal:")
print(f"  Mean: {np.mean(mixed_signal_test):.6f}")
print(f"  Std:  {np.std(mixed_signal_test):.6f}")
print(f"  Min:  {np.min(mixed_signal_test):.6f}")
print(f"  Max:  {np.max(mixed_signal_test):.6f}")

print("\n2. TARGET SIGNALS (Pure Sine) - What model should output:")
print("-"*80)
for i, freq in enumerate([1.0, 3.0, 5.0, 7.0]):
    target = targets_train[i]
    print(f"Frequency {i+1} ({freq} Hz):")
    print(f"  Mean: {np.mean(target):.6f}")
    print(f"  Std:  {np.std(target):.6f}")
    print(f"  Min:  {np.min(target):.6f}")
    print(f"  Max:  {np.max(target):.6f}")

print("\n3. NOISY COMPONENTS - Individual noisy sines:")
print("-"*80)
for i, freq in enumerate([1.0, 3.0, 5.0, 7.0]):
    # Generate one noisy sine to check amplitude
    noisy = train_gen.generate_noisy_sine(freq, train_gen.time_vector)
    print(f"Noisy sine {i+1} ({freq} Hz):")
    print(f"  Mean: {np.mean(noisy):.6f}")
    print(f"  Std:  {np.std(noisy):.6f}")
    print(f"  Min:  {np.min(noisy):.6f}")
    print(f"  Max:  {np.max(noisy):.6f}")

print("\n4. ANALYSIS:")
print("-"*80)

# Check if mixed signal amplitude is reasonable
mixed_amplitude = np.max(np.abs(mixed_signal_train))
target_amplitude = 1.0  # Pure sine is ±1

print(f"Mixed signal amplitude: {mixed_amplitude:.3f}")
print(f"Target amplitude: {target_amplitude:.3f}")
print(f"Amplitude ratio: {mixed_amplitude/target_amplitude:.3f}x")

if mixed_amplitude > 2.0:
    print(f"\n⚠️  WARNING: Mixed signal amplitude is HIGH ({mixed_amplitude:.3f})")
    print(f"   This could make training difficult!")
    print(f"   Reason: Random A(t)~U(0.8,1.2) at EACH sample creates high variance")
elif mixed_amplitude < 0.5:
    print(f"\n⚠️  WARNING: Mixed signal amplitude is LOW ({mixed_amplitude:.3f})")
    print(f"   This could indicate a problem with signal generation")
else:
    print(f"\n✅ Mixed signal amplitude is reasonable ({mixed_amplitude:.3f})")

print("\n5. NORMALIZED VALUES (what model sees after normalization):")
print("-"*80)
mean_train = np.mean(mixed_signal_train)
std_train = np.std(mixed_signal_train)
normalized_train = (mixed_signal_train - mean_train) / std_train

print(f"After normalization:")
print(f"  Input mean: {np.mean(normalized_train):.6f} (should be ~0)")
print(f"  Input std:  {np.std(normalized_train):.6f} (should be ~1)")
print(f"  Input min:  {np.min(normalized_train):.3f}")
print(f"  Input max:  {np.max(normalized_train):.3f}")
print(f"  Target min: {np.min(targets_train[0]):.3f} (stays at ±1)")
print(f"  Target max: {np.max(targets_train[0]):.3f} (stays at ±1)")

print("\n6. EXPECTED LOSS RANGE:")
print("-"*80)
# Random prediction MSE
random_pred_mse = np.mean((np.random.randn(1000) * 0.5)**2)  # Random ~N(0, 0.5)
print(f"Random predictions MSE: ~{random_pred_mse:.3f}")

# Predicting mean (0) MSE
zero_pred_mse = np.mean(targets_train[0]**2)
print(f"Predicting zero MSE: ~{zero_pred_mse:.3f}")

# Good model MSE
print(f"Good model MSE: ~0.001-0.005")

if zero_pred_mse > 1.0:
    print(f"\n⚠️  If initial loss > {zero_pred_mse:.1f}, something is wrong!")
else:
    print(f"\n✅ Initial loss should be around {zero_pred_mse:.3f}")

print("\n" + "="*80)
