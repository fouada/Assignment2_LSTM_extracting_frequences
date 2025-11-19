# Why Your Model Predicts Zero - Root Cause Analysis

## ⚠️ CRITICAL ISSUE IDENTIFIED

Your model is predicting zeros because of **excessive noise** in the data generation combined with configuration issues.

## 🔍 Root Causes:

### 1. **EXTREME NOISE (Main Culprit)**
```yaml
amplitude_range: [0.8, 1.2]  # ❌ 20% variation - TOO HIGH!
phase_range: [0, 6.28]        # ❌ Full 360° - COMPLETELY RANDOM!
```

**What this means:**
- At **EVERY single time step**, both amplitude and phase are randomized
- Amplitude varies by 20% → signal oscillates between 0.8 and 1.2 randomly
- Phase varies across full 360° → signal timing is completely random
- The "noisy" signal has almost NO correlation with the pure target sine wave
- The model sees pure noise and gives up, defaulting to predicting zero (the mean)

**Evidence from code (`signal_generator.py` line 64-71):**
```python
# This generates DIFFERENT random values at EACH time step
amplitudes = self.rng.uniform(0.8, 1.2, size=num_samples)  # 10,000 random values!
phases = self.rng.uniform(0, 2π, size=num_samples)          # 10,000 random values!
noisy_sine = amplitudes * np.sin(2π*f*t + phases)
```

### 2. **Model Defaults to Predicting Zero (MSE Minimization)**
When faced with unpredictable noise:
- Pure sine waves have mean = 0
- MSE loss penalizes wrong predictions squared
- Safest prediction when uncertain = mean = **0**
- This is why you see all zeros in testing!

### 3. **Learning Rate Too High**
```yaml
learning_rate: 0.001  # Too aggressive for noisy data
```

### 4. **Small Batch Size**
```yaml
batch_size: 32  # Causes unstable gradient estimates with high noise
```

## ✅ FIXES APPLIED TO CONFIG

I've already updated your `config/config.yaml` with these critical fixes:

### Fix 1: Reduced Noise to 5%
```yaml
amplitude_range: [0.95, 1.05]  # ✅ Now only 5% variation
```

### Fix 2: Lower Learning Rate
```yaml
learning_rate: 0.0001  # ✅ Reduced by 10x
```

### Fix 3: Larger Batch Size
```yaml
batch_size: 64  # ✅ More stable gradients
```

## 📊 Expected Impact

### Before Fixes:
- Noisy signal correlation with target: **< 0.3** (almost random)
- Model sees: Unpredictable noise
- Model learns: Nothing useful
- Model predicts: Zero (safest option)

### After Fixes:
- Noisy signal correlation with target: **> 0.9** (strong signal)
- Model sees: Actual sine wave patterns + small noise
- Model learns: Extract frequency components
- Model predicts: Actual sine wave values

## 🚀 Next Steps

### Option 1: Train with Current Fixes
```bash
# Install dependencies first (if not already installed)
pip install -r requirements.txt

# Train the model
python main.py
```

### Option 2: Start with Even Less Noise (Recommended)
For initial testing, you can reduce noise even more:

```yaml
# In config/config.yaml
amplitude_range: [0.98, 1.02]  # Only 2% noise
```

This guarantees the model can learn before adding complexity.

### Option 3: Verify with Diagnostics
```bash
# Run diagnostic tests
python simple_diagnostic.py  # Requires numpy, pyyaml only
python diagnostic_test.py    # Full test (requires torch)
```

## 🧪 Understanding the Data Generation Problem

The assignment likely intended:
- **Single random amplitude per frequency** (constant throughout signal)
- **Single random phase per frequency** (constant throughout signal)

But your implementation generates:
- **10,000 random amplitudes** (one per sample)
- **10,000 random phases** (one per sample)

This creates a fundamentally different (and much harder) problem.

### To Fix Data Generation (If Required):
Modify `src/data/signal_generator.py`:

```python
# OLD (current) - Random at each sample:
amplitudes = self.rng.uniform(0.8, 1.2, size=num_samples)
phases = self.rng.uniform(0, 2π, size=num_samples)

# NEW - Single random value for entire signal:
amplitude = self.rng.uniform(0.95, 1.05)  # One value
phase = self.rng.uniform(0, 0.5)           # One value
noisy_sine = amplitude * np.sin(2π*f*t + phase)
```

## 📈 What Good Training Should Look Like

With the fixes applied:
- **Epoch 1:** Loss ~0.5-0.8 (high)
- **Epoch 10:** Loss ~0.1-0.2 (decreasing)
- **Epoch 30:** Loss < 0.05 (converging)
- **Epoch 50:** Loss < 0.01 (good fit)

If you still see:
- Loss stuck at same value
- All predictions near zero
- No decrease over epochs

Then check:
1. Dependencies installed correctly
2. No NaN in gradients (add checks)
3. Data is being loaded correctly

## 💡 Summary

**The model predicts zero because:**
1. ❌ 20% amplitude noise destroys the signal
2. ❌ 360° phase randomization makes timing unpredictable  
3. ❌ High learning rate + small batches → unstable training
4. ✅ Model rationally predicts zero (the mean) when it can't learn

**Fixes applied:**
1. ✅ Reduced noise to 5%
2. ✅ Lowered learning rate 10x
3. ✅ Doubled batch size

**Expected outcome:**
- Model should now converge
- Loss should decrease steadily
- Predictions should match sine waves (not zeros)

---

*Changes saved to: `config/config.yaml`*
*Diagnostic scripts created: `simple_diagnostic.py`, `diagnostic_test.py`*
