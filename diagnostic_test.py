"""
Diagnostic Test Script
Tests if the model can learn at all with minimal noise and checks for dead neurons.

Author: Diagnostic Team
Date: 2025-11-19
"""

import torch
import yaml
import numpy as np
from pathlib import Path

from src.data.signal_generator import SignalGenerator, SignalConfig
from src.data.dataset import FrequencyExtractionDataset, StatefulDataLoader
from src.models.lstm_extractor import create_model


def test_data_generation():
    """Test 1: Check if data generation produces valid signals."""
    print("\n" + "="*80)
    print("TEST 1: Data Generation Validation")
    print("="*80)
    
    # Create a simple generator with minimal noise
    config = SignalConfig(
        frequencies=[1.0],
        sampling_rate=100,
        duration=1.0,
        amplitude_range=[0.99, 1.01],  # Very minimal noise
        phase_range=[0, 0.1],  # Very minimal phase variation
        seed=42
    )
    
    generator = SignalGenerator(config)
    mixed_signal, targets = generator.generate_complete_dataset()
    
    print(f"✓ Mixed signal shape: {mixed_signal.shape}")
    print(f"✓ Mixed signal range: [{mixed_signal.min():.4f}, {mixed_signal.max():.4f}]")
    print(f"✓ Mixed signal mean: {mixed_signal.mean():.4f}")
    print(f"✓ Mixed signal std: {mixed_signal.std():.4f}")
    
    print(f"\n✓ Target signal shape: {targets[0].shape}")
    print(f"✓ Target signal range: [{targets[0].min():.4f}, {targets[0].max():.4f}]")
    print(f"✓ Target signal mean: {targets[0].mean():.4f}")
    print(f"✓ Target signal std: {targets[0].std():.4f}")
    
    # Check if signals are not all zeros
    if np.all(mixed_signal == 0):
        print("❌ ERROR: Mixed signal is all zeros!")
        return False
    if np.all(targets[0] == 0):
        print("❌ ERROR: Target signal is all zeros!")
        return False
    
    print("✅ Data generation PASSED")
    return True


def test_dataset_normalization():
    """Test 2: Check if normalization is destroying the signal."""
    print("\n" + "="*80)
    print("TEST 2: Dataset Normalization Check")
    print("="*80)
    
    config = SignalConfig(
        frequencies=[1.0],
        sampling_rate=100,
        duration=1.0,
        amplitude_range=[0.99, 1.01],
        phase_range=[0, 0.1],
        seed=42
    )
    
    generator = SignalGenerator(config)
    
    # Test with normalization
    dataset = FrequencyExtractionDataset(generator, normalize=True)
    
    # Get a few samples
    samples = [dataset[i] for i in range(10)]
    inputs = torch.stack([s[0] for s in samples])
    targets = torch.stack([s[1] for s in samples])
    
    print(f"✓ Input features shape: {inputs.shape}")
    print(f"✓ Input signal values (first 5): {inputs[:5, 0].numpy()}")
    print(f"✓ Input signal range: [{inputs[:, 0].min():.4f}, {inputs[:, 0].max():.4f}]")
    
    print(f"\n✓ Target values (first 5): {targets[:5].squeeze().numpy()}")
    print(f"✓ Target range: [{targets.min():.4f}, {targets.max():.4f}]")
    
    # Check if normalization killed the signal
    if torch.all(inputs[:, 0] == inputs[0, 0]):
        print("❌ ERROR: All normalized signal values are identical!")
        return False
    
    if torch.all(targets == 0):
        print("❌ ERROR: All targets are zero!")
        return False
    
    print("✅ Normalization PASSED")
    return True


def test_model_forward_pass():
    """Test 3: Check if model can do forward pass without collapsing."""
    print("\n" + "="*80)
    print("TEST 3: Model Forward Pass & Output Range")
    print("="*80)
    
    # Load model config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model = create_model(config['model'])
    
    # Create dummy input
    batch_size = 10
    dummy_input = torch.randn(batch_size, 5)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input, reset_state=True)
    
    print(f"✓ Model created successfully")
    print(f"✓ Input shape: {dummy_input.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Output values (first 5): {output[:5].squeeze().numpy()}")
    print(f"✓ Output range: [{output.min().item():.6f}, {output.max().item():.6f}]")
    print(f"✓ Output mean: {output.mean().item():.6f}")
    print(f"✓ Output std: {output.std().item():.6f}")
    
    # Check if outputs are all zeros or very close to zero
    if torch.all(torch.abs(output) < 1e-10):
        print("❌ ERROR: Model outputs are all zero!")
        return False
    
    # Check for NaN or Inf
    if torch.any(torch.isnan(output)):
        print("❌ ERROR: Model outputs contain NaN!")
        return False
    if torch.any(torch.isinf(output)):
        print("❌ ERROR: Model outputs contain Inf!")
        return False
    
    print("✅ Forward pass PASSED")
    return True


def test_gradient_flow():
    """Test 4: Check if gradients are flowing through the model."""
    print("\n" + "="*80)
    print("TEST 4: Gradient Flow Check")
    print("="*80)
    
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model = create_model(config['model'])
    criterion = torch.nn.MSELoss()
    
    # Create simple data
    inputs = torch.randn(10, 5)
    targets = torch.randn(10, 1)
    
    # Forward pass
    model.train()
    outputs = model(inputs, reset_state=True)
    loss = criterion(outputs, targets)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_gradients = False
    max_grad = 0.0
    zero_grad_layers = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            max_grad = max(max_grad, grad_norm)
            if grad_norm > 0:
                has_gradients = True
            else:
                zero_grad_layers.append(name)
            print(f"✓ {name}: grad_norm = {grad_norm:.6f}")
        else:
            print(f"⚠ {name}: No gradient!")
    
    print(f"\n✓ Loss value: {loss.item():.6f}")
    print(f"✓ Max gradient norm: {max_grad:.6f}")
    
    if not has_gradients:
        print("❌ ERROR: No gradients flowing through the model!")
        return False
    
    if len(zero_grad_layers) > 0:
        print(f"⚠ WARNING: {len(zero_grad_layers)} layers have zero gradients: {zero_grad_layers[:3]}")
    
    print("✅ Gradient flow PASSED")
    return True


def test_simple_training():
    """Test 5: Can the model overfit on 10 samples?"""
    print("\n" + "="*80)
    print("TEST 5: Simple Overfitting Test (10 samples)")
    print("="*80)
    
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create very simple data: just a constant target
    model = create_model(config['model'])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    # Simple dataset: same input -> same output
    inputs = torch.randn(10, 5)
    targets = torch.ones(10, 1) * 0.5  # All targets = 0.5
    
    print(f"Target values (should all be 0.5): {targets.squeeze()[:5].numpy()}")
    
    # Train for 100 steps
    losses = []
    model.train()
    
    for step in range(100):
        optimizer.zero_grad()
        outputs = model(inputs, reset_state=True)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % 20 == 0:
            print(f"Step {step:3d}: Loss = {loss.item():.6f}, "
                  f"Output mean = {outputs.mean().item():.4f}")
    
    # Check if loss decreased
    initial_loss = losses[0]
    final_loss = losses[-1]
    
    print(f"\n✓ Initial loss: {initial_loss:.6f}")
    print(f"✓ Final loss: {final_loss:.6f}")
    print(f"✓ Loss reduction: {((initial_loss - final_loss) / initial_loss * 100):.2f}%")
    
    # Final predictions
    model.eval()
    with torch.no_grad():
        final_outputs = model(inputs, reset_state=True)
    
    print(f"✓ Final predictions (should be ~0.5): {final_outputs.squeeze()[:5].numpy()}")
    
    if final_loss > initial_loss * 0.5:
        print("❌ ERROR: Model cannot even overfit on 10 constant samples!")
        return False
    
    print("✅ Simple training PASSED")
    return True


def main():
    """Run all diagnostic tests."""
    print("\n" + "#"*80)
    print("LSTM DIAGNOSTIC TEST SUITE")
    print("#"*80)
    
    results = {
        "Data Generation": test_data_generation(),
        "Dataset Normalization": test_dataset_normalization(),
        "Model Forward Pass": test_model_forward_pass(),
        "Gradient Flow": test_gradient_flow(),
        "Simple Training": test_simple_training()
    }
    
    print("\n" + "="*80)
    print("DIAGNOSTIC RESULTS SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<50} {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✅ All tests PASSED! The model architecture is sound.")
        print("   If training still fails, the issue is likely:")
        print("   1. Too much noise in the data")
        print("   2. Learning rate too high")
        print("   3. Need more epochs to converge")
    else:
        print("\n❌ Some tests FAILED! Please fix the identified issues.")
    
    print("="*80)
    
    return all_passed


if __name__ == '__main__':
    main()
