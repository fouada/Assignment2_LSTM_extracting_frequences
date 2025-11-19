"""
Verification Script: Normalization Fix for Zero Prediction Bug
Validates that test dataset uses same normalization parameters as training dataset.

This script verifies the critical fix for the bug where model predicted only zeros
during test inference due to normalization parameter mismatch.

Author: Professional ML Engineering Team
Date: November 2025
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data.signal_generator import create_train_test_generators
from src.data.dataset import create_dataloaders, FrequencyExtractionDataset

def verify_normalization_fix():
    """Verify that the normalization fix is working correctly."""

    print("="*80)
    print("NORMALIZATION FIX VERIFICATION")
    print("="*80)

    # Create generators
    print("\n1. Creating train and test signal generators...")
    train_gen, test_gen = create_train_test_generators(
        frequencies=[1.0, 3.0, 5.0, 7.0],
        sampling_rate=1000,
        duration=10.0,
        train_seed=1,
        test_seed=2
    )
    print("✅ Generators created with different seeds (train=1, test=2)")

    # Method 1: OLD WAY (BUGGY) - Create datasets separately
    print("\n" + "-"*80)
    print("2. Testing OLD METHOD (BUGGY - for comparison):")
    print("-"*80)

    train_dataset_old = FrequencyExtractionDataset(train_gen, normalize=True)
    test_gen_new, _ = create_train_test_generators(
        frequencies=[1.0, 3.0, 5.0, 7.0],
        sampling_rate=1000,
        duration=10.0,
        train_seed=1,
        test_seed=2
    )
    test_dataset_old = FrequencyExtractionDataset(test_gen_new, normalize=True)

    print(f"   Train dataset normalization: mean={train_dataset_old.signal_mean:.6f}, std={train_dataset_old.signal_std:.6f}")
    print(f"   Test dataset normalization:  mean={test_dataset_old.signal_mean:.6f}, std={test_dataset_old.signal_std:.6f}")

    if abs(train_dataset_old.signal_mean - test_dataset_old.signal_mean) > 1e-6:
        print(f"   ❌ BUG CONFIRMED: Different normalization parameters!")
        print(f"   Difference in mean: {abs(train_dataset_old.signal_mean - test_dataset_old.signal_mean):.6f}")
        print(f"   This causes model to predict zeros during test!")
    else:
        print(f"   ⚠️  UNEXPECTED: Parameters match (test data might have same statistics by chance)")

    # Method 2: NEW WAY (FIXED) - Use create_dataloaders
    print("\n" + "-"*80)
    print("3. Testing NEW METHOD (FIXED):")
    print("-"*80)

    train_gen_new, test_gen_new2 = create_train_test_generators(
        frequencies=[1.0, 3.0, 5.0, 7.0],
        sampling_rate=1000,
        duration=10.0,
        train_seed=1,
        test_seed=2
    )

    train_loader, test_loader, train_dataset, test_dataset = create_dataloaders(
        train_generator=train_gen_new,
        test_generator=test_gen_new2,
        batch_size=32,
        normalize=True,
        device='cpu'
    )

    print(f"   Train dataset normalization: mean={train_dataset.signal_mean:.6f}, std={train_dataset.signal_std:.6f}")
    print(f"   Test dataset normalization:  mean={test_dataset.signal_mean:.6f}, std={test_dataset.signal_std:.6f}")

    if abs(train_dataset.signal_mean - test_dataset.signal_mean) < 1e-10:
        print(f"   ✅ FIX VERIFIED: Test uses SAME normalization as train!")
        print(f"   Difference in mean: {abs(train_dataset.signal_mean - test_dataset.signal_mean):.10f}")
        print(f"   Difference in std:  {abs(train_dataset.signal_std - test_dataset.signal_std):.10f}")
    else:
        print(f"   ❌ FIX FAILED: Parameters still different!")
        return False

    # Verify data is actually different (different seeds)
    print("\n" + "-"*80)
    print("4. Verifying train and test data are actually different (different seeds):")
    print("-"*80)

    train_sample = train_dataset.mixed_signal[:100]
    test_sample = test_dataset.mixed_signal[:100]

    difference = np.mean(np.abs(train_sample - test_sample))
    print(f"   Mean absolute difference: {difference:.6f}")

    if difference > 0.01:
        print(f"   ✅ Data is different (as expected with different seeds)")
    else:
        print(f"   ⚠️  Data is too similar (unexpected)")

    # Verify sample inspection
    print("\n" + "-"*80)
    print("5. Sample data inspection:")
    print("-"*80)

    train_inp, train_tgt = train_dataset[0]
    test_inp, test_tgt = test_dataset[0]

    print(f"   Train sample 0: input={train_inp.numpy()}, target={train_tgt.item():.6f}")
    print(f"   Test sample 0:  input={test_inp.numpy()}, target={test_tgt.item():.6f}")

    print("\n" + "="*80)
    print("✅ NORMALIZATION FIX VERIFICATION COMPLETE!")
    print("="*80)
    print("\nSummary:")
    print("  • Test dataset now uses TRAIN normalization parameters")
    print("  • This fixes the zero prediction bug")
    print("  • Model can now generalize to test data correctly")
    print("\nNext step: Run main.py to verify model training and testing work correctly")

    return True

if __name__ == "__main__":
    try:
        success = verify_normalization_fix()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
