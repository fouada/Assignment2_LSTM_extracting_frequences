"""
Diagnostic Script: Why is Loss Stuck at 0.5?

MSE = 0.5 suggests model is predicting zeros or constant values.
This script checks:
1. Are predictions actually varying?
2. Are targets correct (pure sine waves)?
3. Are gradients flowing properly?
4. Is the model architecture correct?
"""

import torch
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.data.signal_generator import create_train_test_generators
from src.data.dataset import create_dataloaders
from src.models.lstm_extractor import create_model
import yaml

print("="*80)
print("DIAGNOSTIC: WHY IS LOSS STUCK AT 0.5?")
print("="*80)

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create data
print("\n1. Creating datasets...")
train_gen, test_gen = create_train_test_generators(
    frequencies=config['data']['frequencies'],
    sampling_rate=config['data']['sampling_rate'],
    duration=config['data']['duration'],
    train_seed=config['data']['train_seed'],
    test_seed=config['data']['test_seed']
)

train_loader, test_loader, train_dataset, test_dataset = create_dataloaders(
    train_generator=train_gen,
    test_generator=test_gen,
    batch_size=32,
    normalize=True,
    device='cpu'
)

print(f"✅ Datasets created")

# Check targets
print("\n" + "="*80)
print("2. CHECKING TARGETS (should be pure sine waves ±1)")
print("="*80)

target_sample = train_dataset.targets[0][:1000]
print(f"Target statistics:")
print(f"  Mean: {np.mean(target_sample):.6f} (should be ~0)")
print(f"  Std:  {np.std(target_sample):.6f} (should be ~0.707 for sine)")
print(f"  Min:  {np.min(target_sample):.3f} (should be ~-1)")
print(f"  Max:  {np.max(target_sample):.3f} (should be ~+1)")

expected_sine_std = 1.0 / np.sqrt(2)
if abs(np.std(target_sample) - expected_sine_std) < 0.1:
    print(f"  ✅ Targets are pure sine waves")
else:
    print(f"  ❌ WARNING: Targets don't look like pure sine waves!")

# Create model
print("\n" + "="*80)
print("3. CHECKING MODEL ARCHITECTURE")
print("="*80)

model = create_model(config['model'])
print(f"✅ Model created")
print(f"  Hidden size: {model.hidden_size}")
print(f"  Num layers: {model.num_layers}")
print(f"  Has output_activation: {hasattr(model, 'output_activation')}")
if hasattr(model, 'output_activation'):
    print(f"  Activation type: {type(model.output_activation).__name__}")

# Test forward pass
print("\n" + "="*80)
print("4. TESTING FORWARD PASS")
print("="*80)

model.eval()
batch = next(iter(train_loader))
inputs = batch['input'][:10]  # Small batch
targets = batch['target'][:10]

print(f"Input shape: {inputs.shape}")
print(f"Target shape: {targets.shape}")
print(f"\nTarget values (first 10):")
print(targets.squeeze().numpy())

with torch.no_grad():
    outputs = model(inputs, reset_state=True)

print(f"\nOutput values (first 10):")
print(outputs.squeeze().numpy())

print(f"\nPrediction statistics:")
print(f"  Mean: {outputs.mean().item():.6f}")
print(f"  Std:  {outputs.std().item():.6f}")
print(f"  Min:  {outputs.min().item():.6f}")
print(f"  Max:  {outputs.max().item():.6f}")

if outputs.std().item() < 0.01:
    print(f"  ❌ PROBLEM: Predictions are nearly constant!")
    print(f"  Model is outputting near-zero values → MSE ≈ 0.5")
else:
    print(f"  ✅ Predictions have variation")

# Calculate MSE
mse = torch.mean((outputs - targets) ** 2).item()
print(f"\nMSE: {mse:.6f}")
if abs(mse - 0.5) < 0.1:
    print(f"  ❌ MSE ≈ 0.5 confirms model is predicting near-zero!")

# Check gradients
print("\n" + "="*80)
print("5. CHECKING GRADIENTS")
print("="*80)

model.train()
criterion = torch.nn.MSELoss()

# Forward pass
outputs = model(inputs, reset_state=True)
loss = criterion(outputs, targets)

# Backward pass
model.zero_grad()
loss.backward()

# Check gradient magnitudes
print("\nGradient magnitudes:")
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        print(f"  {name}: {grad_norm:.6e}")
        if grad_norm < 1e-6:
            print(f"    ⚠️  Very small gradient - might be vanishing!")

# Check output layer specifically
if hasattr(model, 'fc2'):
    if model.fc2.weight.grad is not None:
        fc2_grad = model.fc2.weight.grad.norm().item()
        print(f"\n  Output layer (fc2) gradient: {fc2_grad:.6e}")
        if fc2_grad < 1e-6:
            print(f"    ❌ Output layer gradients are vanishing!")

print("\n" + "="*80)
print("6. DIAGNOSIS")
print("="*80)

print("\nPossible issues:")
print("  1. Learning rate too low (current: 0.0001)")
print("  2. Vanishing gradients through Tanh activation")
print("  3. Model stuck in local minimum (predicting zeros)")
print("  4. Need to check if Tanh is actually being applied")

print("\n" + "="*80)
print("RECOMMENDED FIXES:")
print("="*80)
print("\n1. Increase learning rate:")
print("   Change in config.yaml: learning_rate: 0.001 (10x higher)")
print("\n2. Remove Tanh from output OR use different activation:")
print("   Option A: No output activation (let model learn scale)")
print("   Option B: Use linear output with loss scaling")
print("\n3. Check that model is actually using the Tanh:")
print("   Verify forward() method applies self.output_activation()")

print("\nLet me check the actual model code...")
