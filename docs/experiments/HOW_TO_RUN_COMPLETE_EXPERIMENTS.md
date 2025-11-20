# Complete Guide: Running All Experiments for Assignment Submission

## Overview

This guide shows you how to run **all required experiments** with different sequence lengths (L) and save results for submission.

---

## Step 1: Restore Full Assignment Noise (REQUIRED!)

Before running experiments, restore the **full assignment noise** parameters:

### Edit `config/config.yaml`:

```yaml
data:
  amplitude_range: [0.8, 1.2]              # ← Restore from [0.95, 1.05]
  phase_range: [0, 6.283185307179586]      # ← Restore from [0, 0.785398]
```

**Why?** Assignment requires testing with these specific noise parameters.

---

## Step 2: Run Sequence Length Experiments

### Option A: Quick Run (30 epochs per L, ~2-3 hours total)

```bash
uv run bash run_sequence_experiments.sh 30
```

This tests: **L = 1, 10, 50, 100, 500**

### Option B: Full Run (50 epochs per L, ~4-5 hours total)

```bash
uv run bash run_sequence_experiments.sh 50
```

Better results but takes longer.

### Option C: Custom Sequence Lengths

```bash
uv run python experiments_sequence_length.py \
    --sequence-lengths 1 10 50 100 \
    --epochs 40 \
    --config config/config.yaml \
    --output-dir experiments/sequence_length_comparison
```

---

## Step 3: What Gets Generated

After running, you'll have:

```
experiments/sequence_length_comparison/
├── results_summary.json              # All metrics in JSON format
├── comparative_analysis.png          # Visual comparison charts
├── analysis_report.txt               # Text report with findings
├── training_curves_all.png           # Loss curves for all L values
├── per_frequency_comparison.png      # Per-frequency performance
├── computational_costs.png           # Memory/time comparisons
└── models/
    ├── best_model_L1.pt              # Trained model for L=1
    ├── best_model_L10.pt             # Trained model for L=10
    ├── best_model_L50.pt             # Trained model for L=50
    ├── best_model_L100.pt            # Trained model for L=100
    └── best_model_L500.pt            # Trained model for L=500
```

---

## Step 4: Generate Final Submission Report

After experiments complete, create a comprehensive report:

```bash
uv run python visualize_sequence_results.py \
    --input experiments/sequence_length_comparison/results_summary.json \
    --output experiments/final_submission_report.pdf
```

---

## What to Submit

### Required Files for Submission:

1. **Results Summary** (JSON + Text Report)
   - `experiments/sequence_length_comparison/results_summary.json`
   - `experiments/sequence_length_comparison/analysis_report.txt`

2. **Visualizations** (All PNG files)
   - `comparative_analysis.png` - Main comparison chart
   - `training_curves_all.png` - Training convergence
   - `per_frequency_comparison.png` - Performance by frequency
   - `computational_costs.png` - Resource usage

3. **Best Models** (Optional, if required)
   - All `best_model_L*.pt` files

4. **Code** (Your implementation)
   - `src/` directory (all source code)
   - `config/config.yaml` (final configuration)
   - `main.py` (main training script)

---

## Expected Results Summary

### Metrics Tracked for Each L:

| Metric | Description |
|--------|-------------|
| **MSE (Train/Test)** | Mean Squared Error |
| **R² Score** | Coefficient of determination |
| **MAE** | Mean Absolute Error |
| **Convergence Speed** | Epochs to reach 90% performance |
| **Training Time** | Total wall-clock time |
| **Memory Usage** | Peak memory consumption |
| **Per-Frequency R²** | Performance on each frequency |

### Typical Expected Performance:

```
L=1:   R² ~ 0.70-0.75 (baseline)
L=10:  R² ~ 0.75-0.80 (slight improvement)
L=50:  R² ~ 0.78-0.85 (good temporal learning)
L=100: R² ~ 0.80-0.88 (best performance)
L=500: R² ~ 0.75-0.85 (diminishing returns)
```

**Trade-offs:**
- Low L (1-10): Fast training, good for online learning
- Medium L (50-100): Best accuracy, balanced approach
- High L (500+): Slower training, memory intensive

---

## Troubleshooting

### If experiments crash:

**1. Out of Memory:**
```bash
# Reduce batch size in config.yaml
batch_size: 16  # Down from 32
```

**2. Takes too long:**
```bash
# Run with fewer epochs for testing
uv run bash run_sequence_experiments.sh 20
```

**3. Want to test specific L values:**
```bash
# Run only L=1 and L=50
uv run python experiments_sequence_length.py \
    --sequence-lengths 1 50 \
    --epochs 30
```

---

## Complete Workflow Summary

### Full Assignment Submission Workflow:

```bash
# 1. Restore full assignment noise
# Edit config/config.yaml:
#   amplitude_range: [0.8, 1.2]
#   phase_range: [0, 6.283185307179586]

# 2. Run all sequence length experiments
uv run bash run_sequence_experiments.sh 40

# 3. Wait for completion (~3-4 hours)

# 4. Check results
ls experiments/sequence_length_comparison/

# 5. Review analysis report
cat experiments/sequence_length_comparison/analysis_report.txt

# 6. Open visualizations
open experiments/sequence_length_comparison/*.png

# 7. Package for submission
zip -r assignment_results.zip \
    experiments/sequence_length_comparison/ \
    src/ \
    config/ \
    main.py \
    README.md
```

---

## Advanced: Parallel Execution (Optional)

If you have multiple GPUs or want to run experiments in parallel:

```bash
# Terminal 1: Run L=1
uv run python experiments_sequence_length.py --sequence-lengths 1 --epochs 50

# Terminal 2: Run L=10
uv run python experiments_sequence_length.py --sequence-lengths 10 --epochs 50

# Terminal 3: Run L=50
uv run python experiments_sequence_length.py --sequence-lengths 50 --epochs 50

# etc.
```

Then manually combine results.

---

## Checklist Before Submission

- [ ] Config uses **full assignment noise** (amplitude=[0.8, 1.2], phase=[0, 2π])
- [ ] All sequence lengths tested (L=1, 10, 50, 100, 500)
- [ ] `results_summary.json` generated
- [ ] All visualizations created (PNG files)
- [ ] `analysis_report.txt` reviewed
- [ ] Code is clean and documented
- [ ] README updated with final results
- [ ] All files packaged for submission

---

## Quick Start (TL;DR)

```bash
# 1. Update config to full noise (edit config/config.yaml)
# 2. Run experiments
uv run bash run_sequence_experiments.sh 40
# 3. Check results in experiments/sequence_length_comparison/
# 4. Submit required files
```

**Estimated Time:** 3-5 hours (depending on hardware)

**Expected Outcome:** Complete comparison showing L=50-100 performs best for this task.

---

## Questions?

If any step fails, check:
1. Config file has correct parameters
2. All dependencies installed (`uv sync`)
3. Sufficient disk space for models/results
4. Sufficient memory for larger L values

Good luck! 🎯
