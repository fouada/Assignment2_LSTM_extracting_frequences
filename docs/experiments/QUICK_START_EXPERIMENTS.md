# Quick Start: Running Sequence Length Experiments

## TL;DR - Run Everything

```bash
# Run all experiments with full assignment noise (3-4 hours)
uv run bash run_sequence_experiments.sh 40
```

**What this does:**
- Tests L = 1, 10, 50, 100, 500
- Trains each for 40 epochs
- Saves all results to `experiments/sequence_length_comparison/`
- Generates comparison charts and reports

---

## What You'll Get

### Files Generated:

```
experiments/sequence_length_comparison/
├── 📊 results_summary.json           ← All metrics (submit this)
├── 📊 analysis_report.txt            ← Text report (submit this)
├── 📈 comparative_analysis.png       ← Main chart (submit this)
├── 📈 training_curves_all.png        ← Loss curves (submit this)
├── 📈 per_frequency_comparison.png   ← Performance by freq (submit this)
└── 💾 models/
    ├── best_model_L1.pt
    ├── best_model_L10.pt
    ├── best_model_L50.pt
    ├── best_model_L100.pt
    └── best_model_L500.pt
```

### Metrics Compared:

For each L value, you get:
- ✅ Train/Test MSE
- ✅ R² Score (overall and per-frequency)
- ✅ Training time
- ✅ Convergence speed
- ✅ Memory usage
- ✅ Per-frequency performance (1 Hz, 3 Hz, 5 Hz, 7 Hz)

---

## Alternative Options

### Option 1: Quick Test (20 epochs, ~1.5 hours)

```bash
uv run bash run_sequence_experiments.sh 20
```

Good for: Testing if everything works

### Option 2: Full Run (50 epochs, ~5 hours)

```bash
uv run bash run_sequence_experiments.sh 50
```

Good for: Best results, final submission

### Option 3: Custom Sequence Lengths

Test only specific L values:

```bash
uv run python experiments_sequence_length.py \
    --sequence-lengths 1 50 100 \
    --epochs 40
```

### Option 4: Single L Value

Train just one model:

```bash
# L=1 (current default in main.py)
uv run main.py

# L=50 (modify config first, then run main.py)
```

---

## Before Running

### ✅ Checklist:

1. **Config has full assignment noise** (already updated!)
   ```yaml
   amplitude_range: [0.8, 1.2]
   phase_range: [0, 6.283185307179586]
   ```

2. **Sufficient disk space** (~500 MB for all models)

3. **Time available** (3-5 hours uninterrupted)

4. **Good internet** (if using cloud)

---

## While Running

You'll see output like:

```
=========================================
Sequence Length Experiment Suite
=========================================

Configuration:
  Epochs: 40
  Config: config/config.yaml

Testing L = 1, 10, 50, 100, 500

================================================================================
Running Experiment: L = 1
================================================================================

Creating dataloaders for L=1...
Training model...
Epoch 1/40 - Train Loss: 0.652, Val Loss: 0.523
Epoch 2/40 - Train Loss: 0.412, Val Loss: 0.389
...
Experiment L=1 completed! MSE: 0.125, R²: 0.750

================================================================================
Running Experiment: L = 10
================================================================================
...
```

---

## After Completion

### 1. Review Results

```bash
# Read text report
cat experiments/sequence_length_comparison/analysis_report.txt

# View charts
open experiments/sequence_length_comparison/*.png
```

### 2. Check JSON Results

```bash
cat experiments/sequence_length_comparison/results_summary.json
```

Example output:
```json
{
  "L1": {
    "train_mse": 0.125,
    "test_mse": 0.128,
    "r2_score": 0.744,
    "training_time": 245.3,
    "best_epoch": 32,
    "per_frequency": {
      "1Hz": {"r2": 0.95},
      "3Hz": {"r2": 0.72},
      "5Hz": {"r2": 0.76},
      "7Hz": {"r2": 0.55}
    }
  },
  "L50": {
    "train_mse": 0.082,
    "test_mse": 0.086,
    "r2_score": 0.828,
    ...
  }
}
```

### 3. Package for Submission

```bash
# Create submission package
zip -r assignment_submission.zip \
    experiments/sequence_length_comparison/ \
    src/ \
    config/ \
    main.py \
    README.md
```

---

## Expected Results

### Typical Performance (with full assignment noise):

| L Value | R² Score | Training Time | Best For |
|---------|----------|---------------|----------|
| L=1 | 0.70-0.75 | Fast (~4 min) | Real-time/Online learning |
| L=10 | 0.72-0.78 | Medium (~5 min) | Fast batch processing |
| L=50 | 0.78-0.85 | Medium (~6 min) | **Good balance** ⭐ |
| L=100 | 0.80-0.88 | Slow (~8 min) | **Best accuracy** ⭐ |
| L=500 | 0.75-0.85 | Very slow (~15 min) | Research/offline |

**Key Finding:** L=50-100 offers best trade-off between accuracy and efficiency.

---

## Troubleshooting

### "Out of memory"
```bash
# Reduce batch size in config.yaml
batch_size: 16  # or even 8
```

### "Taking too long"
```bash
# Run with fewer epochs
uv run bash run_sequence_experiments.sh 20
```

### "Want to skip large L values"
```bash
# Test only small L values
uv run python experiments_sequence_length.py \
    --sequence-lengths 1 10 50 \
    --epochs 30
```

---

## Summary

**Simplest workflow:**

1. ✅ Config updated (full noise) - **Already done!**
2. Run: `uv run bash run_sequence_experiments.sh 40`
3. Wait ~3-4 hours
4. Check: `experiments/sequence_length_comparison/`
5. Submit: All PNG files + `results_summary.json` + `analysis_report.txt`

**That's it!** 🎯

For detailed information, see: `HOW_TO_RUN_COMPLETE_EXPERIMENTS.md`
