"""
Comprehensive Training Convergence Validation
Validates that MSE decreases during training and achieves best loss on test.

This script:
1. Trains a model for a few epochs
2. Validates MSE decreases over time
3. Checks test performance
4. Generates a detailed validation report

Author: Professional ML Engineering Team
Date: November 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import time
from pathlib import Path
import sys
import logging
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data.signal_generator import create_train_test_generators
from src.data.dataset import create_dataloaders
from src.models.lstm_extractor import create_model
from src.evaluation.metrics import evaluate_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def quick_train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch and return loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in train_loader:
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)
        is_first_batch = batch['is_first_batch']

        # Reset state at start of each frequency
        if is_first_batch:
            model.reset_state()

        # Forward pass
        outputs = model(inputs, reset_state=False)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Detach state
        model.detach_state()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches

def quick_eval(model, data_loader, criterion, device):
    """Quick evaluation and return loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            is_first_batch = batch['is_first_batch']

            if is_first_batch:
                model.reset_state()

            outputs = model(inputs, reset_state=False)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches

def validate_training_convergence():
    """Run comprehensive training convergence validation."""

    print("="*80)
    print("TRAINING CONVERGENCE VALIDATION")
    print("="*80)

    # Setup
    device = torch.device('mps' if torch.backends.mps.is_available()
                         else 'cuda' if torch.cuda.is_available()
                         else 'cpu')
    logger.info(f"Using device: {device}")

    # Configuration (smaller for quick test)
    config = {
        'frequencies': [1.0, 3.0, 5.0, 7.0],
        'sampling_rate': 1000,
        'duration': 2.0,  # Shorter for quick test
        'batch_size': 32,
        'epochs': 15,  # Few epochs to see convergence
        'learning_rate': 0.001,
        'model': {
            'input_size': 5,
            'hidden_size': 64,  # Smaller for speed
            'num_layers': 2,
            'output_size': 1,
            'dropout': 0.2
        }
    }

    print("\n" + "-"*80)
    print("1. GENERATING DATA")
    print("-"*80)

    train_gen, test_gen = create_train_test_generators(
        frequencies=config['frequencies'],
        sampling_rate=config['sampling_rate'],
        duration=config['duration'],
        train_seed=1,
        test_seed=2
    )
    logger.info(f"✅ Data generators created")

    print("\n" + "-"*80)
    print("2. CREATING DATASETS (WITH NORMALIZATION FIX)")
    print("-"*80)

    train_loader, test_loader, train_dataset, test_dataset = create_dataloaders(
        train_generator=train_gen,
        test_generator=test_gen,
        batch_size=config['batch_size'],
        normalize=True,
        device='cpu'
    )

    # Verify normalization fix
    if abs(train_dataset.signal_mean - test_dataset.signal_mean) < 1e-10:
        logger.info(f"✅ NORMALIZATION FIX VERIFIED: Test uses train stats")
        logger.info(f"   Train: mean={train_dataset.signal_mean:.6f}, std={train_dataset.signal_std:.6f}")
        logger.info(f"   Test:  mean={test_dataset.signal_mean:.6f}, std={test_dataset.signal_std:.6f}")
    else:
        logger.error(f"❌ NORMALIZATION BUG DETECTED!")
        return False

    print("\n" + "-"*80)
    print("3. CREATING MODEL")
    print("-"*80)

    model = create_model(config['model'])
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"✅ Model created with {num_params:,} parameters")

    print("\n" + "-"*80)
    print("4. TRAINING MODEL")
    print("-"*80)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Track losses
    train_losses = []
    test_losses = []

    best_test_loss = float('inf')

    print(f"\nEpoch | Train Loss | Test Loss  | Best Test  | Status")
    print("-"*70)

    start_time = time.time()

    for epoch in range(config['epochs']):
        # Training
        train_loss = quick_train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)

        # Testing
        test_loss = quick_eval(model, test_loader, criterion, device)
        test_losses.append(test_loss)

        # Track best
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            status = "✅ NEW BEST"
        else:
            status = ""

        print(f"{epoch+1:5d} | {train_loss:10.6f} | {test_loss:10.6f} | {best_test_loss:10.6f} | {status}")

    training_time = time.time() - start_time

    print("-"*70)
    logger.info(f"Training completed in {training_time:.2f} seconds")

    print("\n" + "-"*80)
    print("5. CONVERGENCE ANALYSIS")
    print("-"*80)

    # Check if training loss decreased
    initial_train_loss = train_losses[0]
    final_train_loss = train_losses[-1]
    train_improvement = (initial_train_loss - final_train_loss) / initial_train_loss * 100

    # Check if test loss decreased
    initial_test_loss = test_losses[0]
    final_test_loss = test_losses[-1]
    test_improvement = (initial_test_loss - final_test_loss) / initial_test_loss * 100

    # Check generalization
    generalization_gap = abs(final_test_loss - final_train_loss) / final_train_loss * 100

    print(f"\nTraining Loss:")
    print(f"  Initial: {initial_train_loss:.6f}")
    print(f"  Final:   {final_train_loss:.6f}")
    print(f"  Improvement: {train_improvement:.2f}%")
    if train_improvement > 50:
        print(f"  ✅ EXCELLENT: Loss decreased by {train_improvement:.1f}%")
    elif train_improvement > 20:
        print(f"  ✅ GOOD: Loss decreased by {train_improvement:.1f}%")
    else:
        print(f"  ⚠️  POOR: Loss only decreased by {train_improvement:.1f}%")

    print(f"\nTest Loss:")
    print(f"  Initial: {initial_test_loss:.6f}")
    print(f"  Final:   {final_test_loss:.6f}")
    print(f"  Best:    {best_test_loss:.6f}")
    print(f"  Improvement: {test_improvement:.2f}%")
    if test_improvement > 50:
        print(f"  ✅ EXCELLENT: Loss decreased by {test_improvement:.1f}%")
    elif test_improvement > 20:
        print(f"  ✅ GOOD: Loss decreased by {test_improvement:.1f}%")
    else:
        print(f"  ⚠️  POOR: Loss only decreased by {test_improvement:.1f}%")

    print(f"\nGeneralization:")
    print(f"  Gap: {generalization_gap:.2f}%")
    if generalization_gap < 5:
        print(f"  ✅ EXCELLENT: Gap < 5%")
    elif generalization_gap < 10:
        print(f"  ✅ GOOD: Gap < 10%")
    elif generalization_gap < 20:
        print(f"  ⚠️  ACCEPTABLE: Gap < 20%")
    else:
        print(f"  ❌ POOR: Gap > 20% (possible overfitting)")

    print("\n" + "-"*80)
    print("6. DETAILED EVALUATION")
    print("-"*80)

    # Full evaluation with metrics
    train_results = evaluate_model(model, train_loader, device)
    test_results = evaluate_model(model, test_loader, device)

    print(f"\nTrain Metrics:")
    print(f"  MSE:         {train_results['overall']['mse']:.6f}")
    print(f"  MAE:         {train_results['overall']['mae']:.6f}")
    print(f"  R² Score:    {train_results['overall']['r2_score']:.4f}")
    print(f"  Correlation: {train_results['overall']['correlation']:.4f}")
    print(f"  SNR (dB):    {train_results['overall']['snr_db']:.2f}")

    print(f"\nTest Metrics:")
    print(f"  MSE:         {test_results['overall']['mse']:.6f}")
    print(f"  MAE:         {test_results['overall']['mae']:.6f}")
    print(f"  R² Score:    {test_results['overall']['r2_score']:.4f}")
    print(f"  Correlation: {test_results['overall']['correlation']:.4f}")
    print(f"  SNR (dB):    {test_results['overall']['snr_db']:.2f}")

    # Check if predictions are zeros
    test_preds = test_results['predictions']
    pred_std = np.std(test_preds)
    pred_mean = np.mean(test_preds)

    print(f"\nPrediction Analysis:")
    print(f"  Mean: {pred_mean:.6f}")
    print(f"  Std:  {pred_std:.6f}")
    print(f"  Min:  {np.min(test_preds):.6f}")
    print(f"  Max:  {np.max(test_preds):.6f}")

    if pred_std < 0.001:
        print(f"  ❌ PROBLEM: Predictions are nearly constant (std={pred_std:.6f})")
        print(f"  This suggests the zero prediction bug!")
    else:
        print(f"  ✅ GOOD: Predictions have variation (std={pred_std:.4f})")

    print("\n" + "-"*80)
    print("7. CREATING VISUALIZATION")
    print("-"*80)

    # Create loss plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    epochs_range = range(1, len(train_losses) + 1)
    ax1.plot(epochs_range, train_losses, 'b-o', label='Train Loss', linewidth=2)
    ax1.plot(epochs_range, test_losses, 'r-s', label='Test Loss', linewidth=2)
    ax1.axhline(y=best_test_loss, color='g', linestyle='--', label=f'Best Test: {best_test_loss:.4f}')
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('MSE Loss', fontweight='bold')
    ax1.set_title('Training Convergence', fontweight='bold', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Sample predictions vs targets
    sample_size = min(500, len(test_preds))
    sample_idx = np.random.choice(len(test_preds), sample_size, replace=False)
    sample_preds = test_preds[sample_idx]
    sample_targets = test_results['targets'][sample_idx]

    ax2.scatter(sample_targets, sample_preds, alpha=0.5, s=10)
    min_val = min(sample_targets.min(), sample_preds.min())
    max_val = max(sample_targets.max(), sample_preds.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax2.set_xlabel('Target', fontweight='bold')
    ax2.set_ylabel('Prediction', fontweight='bold')
    ax2.set_title(f'Test Predictions vs Targets\nR²={test_results["overall"]["r2_score"]:.4f}',
                  fontweight='bold', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_dir = Path('validation_results')
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / 'training_convergence_validation.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"✅ Plot saved to: {plot_path}")

    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    # Overall validation result
    validation_passed = True
    checks = []

    # Check 1: Normalization fix
    if abs(train_dataset.signal_mean - test_dataset.signal_mean) < 1e-10:
        checks.append(("✅", "Normalization fix verified"))
    else:
        checks.append(("❌", "Normalization bug detected"))
        validation_passed = False

    # Check 2: Training loss decreased
    if train_improvement > 20:
        checks.append(("✅", f"Training loss decreased by {train_improvement:.1f}%"))
    else:
        checks.append(("❌", f"Training loss only decreased by {train_improvement:.1f}%"))
        validation_passed = False

    # Check 3: Test loss decreased
    if test_improvement > 20:
        checks.append(("✅", f"Test loss decreased by {test_improvement:.1f}%"))
    else:
        checks.append(("❌", f"Test loss only decreased by {test_improvement:.1f}%"))
        validation_passed = False

    # Check 4: Generalization
    if generalization_gap < 20:
        checks.append(("✅", f"Generalization gap: {generalization_gap:.1f}%"))
    else:
        checks.append(("❌", f"Generalization gap too large: {generalization_gap:.1f}%"))
        validation_passed = False

    # Check 5: Predictions not zeros
    if pred_std > 0.01:
        checks.append(("✅", f"Predictions have variation (std={pred_std:.4f})"))
    else:
        checks.append(("❌", f"Predictions are nearly constant (std={pred_std:.6f})"))
        validation_passed = False

    # Check 6: Final MSE reasonable
    if final_test_loss < 0.1:
        checks.append(("✅", f"Final test MSE: {final_test_loss:.6f}"))
    else:
        checks.append(("⚠️ ", f"Final test MSE high: {final_test_loss:.6f}"))

    # Check 7: R² score
    if test_results['overall']['r2_score'] > 0.8:
        checks.append(("✅", f"Test R²: {test_results['overall']['r2_score']:.4f}"))
    else:
        checks.append(("⚠️ ", f"Test R² low: {test_results['overall']['r2_score']:.4f}"))

    print("\nValidation Checks:")
    for status, message in checks:
        print(f"  {status} {message}")

    print("\n" + "="*80)
    if validation_passed:
        print("✅ VALIDATION PASSED!")
        print("="*80)
        print("\n✅ All critical checks passed:")
        print("   • MSE decreases during training")
        print("   • MSE decreases on test set")
        print("   • Model achieves good test performance")
        print("   • Normalization fix working correctly")
        print("   • Model predictions are NOT zeros")
        print(f"\n📊 Results saved to: {output_dir}")
        return True
    else:
        print("❌ VALIDATION FAILED!")
        print("="*80)
        print("\n❌ Some checks failed - review the issues above")
        return False

if __name__ == "__main__":
    try:
        success = validate_training_convergence()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
