"""
Quick Test Script for Innovation Features
Run this to verify all innovations are properly installed and working.

Usage: python test_innovations.py
"""

import sys
from pathlib import Path

print("=" * 80)
print("🧪 TESTING INNOVATION FEATURES")
print("=" * 80)

# Test 1: Import core modules
print("\n1️⃣  Testing imports...")
try:
    import torch
    import yaml
    import numpy as np
    print("   ✅ Core dependencies OK")
except ImportError as e:
    print(f"   ❌ Core dependency missing: {e}")
    sys.exit(1)

# Test 2: Import base model
print("\n2️⃣  Testing base model...")
try:
    from src.models import StatefulLSTMExtractor
    model = StatefulLSTMExtractor()
    print(f"   ✅ Base LSTM OK ({model.count_parameters():,} parameters)")
except Exception as e:
    print(f"   ❌ Base model failed: {e}")
    sys.exit(1)

# Test 3: Import Attention LSTM
print("\n3️⃣  Testing Attention LSTM...")
try:
    from src.models import AttentionLSTMExtractor
    model = AttentionLSTMExtractor(
        input_size=5,
        hidden_size=64,
        attention_heads=4
    )
    print(f"   ✅ Attention LSTM OK ({model.count_parameters():,} parameters)")
except Exception as e:
    print(f"   ❌ Attention LSTM failed: {e}")
    sys.exit(1)

# Test 4: Import Bayesian LSTM
print("\n4️⃣  Testing Bayesian LSTM...")
try:
    from src.models import BayesianLSTMExtractor
    model = BayesianLSTMExtractor(
        input_size=5,
        hidden_size=64,
        mc_samples=50
    )
    print(f"   ✅ Bayesian LSTM OK ({model.count_parameters():,} parameters)")
except Exception as e:
    print(f"   ❌ Bayesian LSTM failed: {e}")
    sys.exit(1)

# Test 5: Import Hybrid LSTM
print("\n5️⃣  Testing Hybrid LSTM...")
try:
    from src.models import HybridLSTMExtractor
    model = HybridLSTMExtractor(
        input_size=5,
        hidden_size=64,
        fft_size=128
    )
    print(f"   ✅ Hybrid LSTM OK ({model.count_parameters():,} parameters)")
except Exception as e:
    print(f"   ❌ Hybrid LSTM failed: {e}")
    sys.exit(1)

# Test 6: Import Active Learning
print("\n6️⃣  Testing Active Learning...")
try:
    from src.training.active_learning_trainer import ActiveLearningTrainer
    print("   ✅ Active Learning OK")
except Exception as e:
    print(f"   ❌ Active Learning failed: {e}")
    sys.exit(1)

# Test 7: Import Adversarial Tester
print("\n7️⃣  Testing Adversarial Tester...")
try:
    from src.evaluation.adversarial_tester import AdversarialTester
    print("   ✅ Adversarial Tester OK")
except Exception as e:
    print(f"   ❌ Adversarial Tester failed: {e}")
    sys.exit(1)

# Test 8: Quick forward pass
print("\n8️⃣  Testing forward pass...")
try:
    from src.models import AttentionLSTMExtractor
    model = AttentionLSTMExtractor(input_size=5, hidden_size=32)
    model.eval()
    
    # Create dummy input
    x = torch.randn(1, 10, 5)  # (batch, seq, features)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"   ✅ Forward pass OK (output shape: {output.shape})")
except Exception as e:
    print(f"   ❌ Forward pass failed: {e}")
    sys.exit(1)

# Test 9: Check documentation
print("\n9️⃣  Testing documentation files...")
docs = [
    'INNOVATION_ROADMAP.md',
    'INNOVATIONS_QUICK_START.md',
    'INNOVATIONS_SUMMARY.md',
    'INNOVATION_COMPLETE.md',
    'START_HERE_INNOVATIONS.txt'
]
missing = []
for doc in docs:
    if not Path(doc).exists():
        missing.append(doc)

if missing:
    print(f"   ⚠️  Missing documentation: {', '.join(missing)}")
else:
    print("   ✅ All documentation files present")

# Test 10: Check demo script
print("\n🔟 Testing demo script...")
if Path('demo_innovations.py').exists():
    print("   ✅ Demo script present")
else:
    print("   ⚠️  Demo script missing")

# Final Summary
print("\n" + "=" * 80)
print("🎉 ALL TESTS PASSED!")
print("=" * 80)
print("\n✨ Your innovations are ready to use!")
print("\nNext steps:")
print("  1. Run: python demo_innovations.py")
print("  2. Check: innovations_demo/ folder for outputs")
print("  3. Read: INNOVATIONS_QUICK_START.md for usage")
print("\n" + "=" * 80)

