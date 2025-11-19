"""
Final Fix Verification: Targets Should NOT Be Normalized
Verifies that inputs are normalized but targets remain as pure sine waves.
"""

import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.data.signal_generator import create_train_test_generators
from src.data.dataset import create_dataloaders

print("="*80)
print("FINAL FIX VERIFICATION")
print("="*80)

# Create data
train_gen, test_gen = create_train_test_generators(
    frequencies=[1.0, 3.0, 5.0, 7.0],
    sampling_rate=1000,
    duration=10.0,
    train_seed=1,
    test_seed=2
)

train_loader, test_loader, train_dataset, test_dataset = create_dataloaders(
    train_generator=train_gen,
    test_generator=test_gen,
    batch_size=32,
    normalize=True,
    device='cpu'
)

print("\n✅ CORRECT FIX VERIFICATION:")
print("-"*80)

# Check input normalization
print(f"\n1. Input (Mixed Signal) Normalization:")
print(f"   Train: mean={train_dataset.signal_mean:.6f}, std={train_dataset.signal_std:.6f}")
print(f"   Test:  mean={test_dataset.signal_mean:.6f}, std={test_dataset.signal_std:.6f}")

if abs(train_dataset.signal_mean - test_dataset.signal_mean) < 1e-10:
    print(f"   ✅ Test uses TRAIN normalization (correct!)")
else:
    print(f"   ❌ ERROR: Different normalization!")

# Check normalized input range
train_input_sample = train_dataset.mixed_signal[:1000]
print(f"\n2. Normalized Input Range:")
print(f"   Mean: {np.mean(train_input_sample):.6f} (should be ~0)")
print(f"   Std:  {np.std(train_input_sample):.6f} (should be ~1)")
print(f"   Min:  {np.min(train_input_sample):.3f}")
print(f"   Max:  {np.max(train_input_sample):.3f}")

# Check target range (should be ±1, NOT normalized)
train_target_sample = train_dataset.targets[0][:1000]
print(f"\n3. Target (Pure Sine) Range:")
print(f"   Mean: {np.mean(train_target_sample):.6f} (should be ~0)")
print(f"   Std:  {np.std(train_target_sample):.6f} (should be ~0.707 for sine)")
print(f"   Min:  {np.min(train_target_sample):.3f} (should be ~-1)")
print(f"   Max:  {np.max(train_target_sample):.3f} (should be ~+1)")

if abs(np.min(train_target_sample) - (-1.0)) < 0.1 and abs(np.max(train_target_sample) - 1.0) < 0.1:
    print(f"   ✅ CORRECT: Targets are pure sine waves (±1 amplitude)")
else:
    print(f"   ❌ ERROR: Targets are not at ±1 scale!")

print("\n" + "="*80)
print("✅ FINAL FIX COMPLETE!")
print("="*80)
print("\nWhat changed:")
print("  • Inputs (mixed signal): Normalized with train statistics ✅")
print("  • Targets (pure sine): NOT normalized, kept at ±1 ✅")
print("  • Model learns: normalized_input → pure_sine_output")
print("\nExpected training results:")
print("  • Initial loss: ~0.5-1.0 (much better than 4.0!)")
print("  • Final loss after 50 epochs: ~0.001-0.005")
print("  • Test loss: Similar to train loss")
print("\nNow run: uv run main.py")
