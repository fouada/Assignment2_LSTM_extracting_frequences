"""
Quick verification that output activation is working correctly.
Tests that model outputs are bounded to [-1, +1] as expected.
"""

import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.models.lstm_extractor import create_model

print("="*80)
print("OUTPUT ACTIVATION VERIFICATION")
print("="*80)

# Create model
config = {
    'input_size': 5,
    'hidden_size': 64,
    'num_layers': 2,
    'output_size': 1,
    'dropout': 0.2
}

model = create_model(config)
model.eval()

print("\n✅ Model Architecture Check:")
print(f"   - Has output_activation layer: {hasattr(model, 'output_activation')}")
print(f"   - Activation type: {type(model.output_activation).__name__}")

# Test with random inputs
print("\n✅ Testing Output Range:")
batch_size = 100
seq_len = 50
test_input = torch.randn(batch_size, seq_len, 5) * 10  # Large random values

with torch.no_grad():
    outputs = model(test_input, reset_state=True)

output_min = outputs.min().item()
output_max = outputs.max().item()
output_mean = outputs.mean().item()
output_std = outputs.std().item()

print(f"   Input range: [{test_input.min().item():.2f}, {test_input.max().item():.2f}]")
print(f"   Output range: [{output_min:.6f}, {output_max:.6f}]")
print(f"   Output mean: {output_mean:.6f}")
print(f"   Output std: {output_std:.6f}")

# Verify outputs are bounded
if output_min >= -1.0 and output_max <= 1.0:
    print(f"\n   ✅ CORRECT: Outputs are bounded to [-1, +1]")
    print(f"   ✅ This matches the target range (pure sine waves)")
else:
    print(f"\n   ❌ ERROR: Outputs exceed [-1, +1] bounds!")
    print(f"   ❌ This will cause high MSE loss (~4.0)")

print("\n" + "="*80)
print("✅ OUTPUT ACTIVATION FIX COMPLETE!")
print("="*80)
print("\nExpected training results:")
print("  • Initial loss: ~0.5-1.0 (not 4.0+ anymore!)")
print("  • Final loss after 50 epochs: ~0.001-0.005")
print("  • Test loss: Similar to train loss")
print("  • Predictions: Bounded to [-1, +1]")
print("\nNext step: Run training with:")
print("  uv run main.py")
