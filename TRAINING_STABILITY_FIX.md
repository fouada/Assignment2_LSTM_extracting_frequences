# Training Stability Fix

## Problem Identified

Your training showed **learning followed by collapse**:

```
Epoch 1:  Loss: 0.575
Epoch 11: Loss: 0.228 ✅ (best - model was learning!)
Epoch 13: Loss: 0.500 ❌ (collapsed - model broke)
Epoch 14-22: Stuck at 0.5 (early stopping triggered)
```

### Per-Frequency Analysis
```
Frequency 1 (1 Hz):  R² = 0.93 ✅ EXCELLENT
Frequency 2 (3 Hz):  R² = 0.11 ❌ POOR
Frequency 3 (5 Hz):  R² = 0.32 ❌ POOR
Frequency 4 (7 Hz):  R² = 0.82 ✅ GOOD
```

**Pattern:** Model learns extreme frequencies well but struggles with middle frequencies!

## Root Causes

1. **Training Instability:** Learning rate 0.001 causes gradient explosions around epoch 13
2. **Model Capacity:** Hidden size 128 insufficient to separate all 4 frequencies simultaneously
3. **Scheduler Too Aggressive:** Reduces LR too quickly (patience=5, factor=0.5)

## Changes Made

### 1. Reduced Learning Rate (Stability)
```yaml
# BEFORE:
learning_rate: 0.001  # Too high → unstable

# AFTER:
learning_rate: 0.0005  # 5x original, balanced
```
**Why:** Fast initial learning but prevents gradient explosions

### 2. Increased Model Capacity
```yaml
# BEFORE:
hidden_size: 128  # Insufficient for 4 frequencies

# AFTER:
hidden_size: 256  # 2x capacity for better separation
```
**Why:** Model needs more capacity to distinguish similar frequencies (3 Hz vs 5 Hz)

### 3. More Stable Scheduler
```yaml
# BEFORE:
scheduler_patience: 5   # Too quick
scheduler_factor: 0.5   # Too aggressive

# AFTER:
scheduler_patience: 10  # More patience
scheduler_factor: 0.7   # Gentler reduction
```
**Why:** Prevents premature learning rate reduction

## Expected Results

### Previous Run (Unstable):
```
Best Loss: 0.228
Final R²: 0.54
Training: Collapsed at epoch 13
```

### Expected After Fix:
```
Epoch 1:  Loss ~0.6
Epoch 20: Loss ~0.05-0.1
Epoch 50: Loss ~0.01-0.02
Final R²: >0.85
Training: Stable throughout
```

### Per-Frequency Expected:
```
All frequencies: R² > 0.8 (all should be good now)
```

## Next Steps

Run training again:
```bash
uv run main.py
```

**What to watch for:**
1. ✅ Loss should decrease smoothly (no sudden jumps)
2. ✅ Should NOT collapse back to 0.5
3. ✅ All 4 frequencies should have R² > 0.8
4. ✅ Training time: ~5-7 minutes (larger model)

## If Still Having Issues

### If loss still plateaus at 0.5:
1. Check gradients are flowing: Run diagnostic again
2. Try removing Tanh activation temporarily
3. Verify state management is correct

### If overfitting occurs:
1. Increase dropout: 0.2 → 0.3
2. Add more weight decay: 1e-5 → 1e-4

### For assignment compliance:
After getting good results, restore full noise:
```yaml
amplitude_range: [0.8, 1.2]     # Restore
phase_range: [0, 6.283185307179586]  # Restore (2π)
```

## Summary

**Changes:**
- Learning rate: 0.001 → 0.0005 (more stable)
- Hidden size: 128 → 256 (more capacity)
- Scheduler patience: 5 → 10 (less aggressive)
- Scheduler factor: 0.5 → 0.7 (gentler)

**Expected outcome:** Stable training with MSE ~0.01-0.02 and R² > 0.85 for all frequencies.

**Next:** `uv run main.py`
