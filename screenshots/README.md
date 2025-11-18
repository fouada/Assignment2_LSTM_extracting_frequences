# Screenshots Directory
## LSTM Frequency Extraction System - Execution Documentation

**Purpose**: Organized storage for all execution screenshots  
**Authors**: Fouad Azem & Tal Goldengorn  
**Date**: November 2025

---

## 📁 Directory Structure

```
screenshots/
├── README.md (this file)
│
├── 01_pre_execution/
│   ├── 01_project_structure.png
│   ├── 02_configuration.png
│   └── 03_requirements.png
│
├── 02_execution/
│   ├── 04_execution_start.png
│   ├── 05_data_generation.png
│   ├── 06_dataset_creation.png
│   ├── 07_model_initialization.png
│   ├── 08_training_early.png
│   ├── 09_training_mid.png
│   ├── 10_training_completion.png
│   ├── 11_train_metrics.png
│   ├── 12_test_metrics.png
│   └── 13_final_summary.png
│
├── 03_outputs/
│   ├── 14_directory_structure.png
│   ├── 15_graph1_single_frequency_REQUIRED.png  ⭐⭐⭐
│   ├── 16_graph2_all_frequencies_REQUIRED.png   ⭐⭐⭐
│   ├── 17_training_history.png
│   ├── 18_error_distribution.png
│   ├── 19_metrics_comparison.png
│   └── 20_cost_analysis.png (optional)
│
└── 04_demo_outputs/ (optional)
    ├── data_generation_demo.png
    ├── model_architecture_demo.png
    └── quick_demo_results.png
```

---

## 📸 Screenshot Capture Checklist

### Phase 1: Pre-Execution (Before running main.py)

**Location**: `01_pre_execution/`

- [ ] **01_project_structure.png**
  - Command: `cd Assignment2_LSTM_extracting_frequences && tree -L 2`
  - Shows: Complete project structure
  - Purpose: Verify all files present

- [ ] **02_configuration.png**
  - Command: `cat config/config.yaml`
  - Shows: Full configuration file
  - Purpose: Document hyperparameters used

- [ ] **03_requirements.png**
  - Command: `cat requirements.txt`
  - Shows: All dependencies
  - Purpose: Document environment

---

### Phase 2: During Execution (While running main.py)

**Location**: `02_execution/`

- [ ] **04_execution_start.png**
  - Timing: 0:05 after start
  - Shows: Configuration loaded, device selected
  - Key info: Random seed, device (CPU/MPS/CUDA)

- [ ] **05_data_generation.png**
  - Timing: 0:10 after start
  - Shows: Data generation complete
  - Key info: Train/test seeds, dataset sizes

- [ ] **06_dataset_creation.png**
  - Timing: 0:15 after start
  - Shows: Dataset creation complete
  - Key info: 40,000 samples, batch size

- [ ] **07_model_initialization.png**
  - Timing: 0:20 after start
  - Shows: Model architecture, parameter count
  - Key info: 215,041 parameters, layer structure

- [ ] **08_training_early.png**
  - Timing: ~2:00 (epoch ~10)
  - Shows: Early training progress
  - Key info: Loss decreasing, learning rate

- [ ] **09_training_mid.png**
  - Timing: ~4:00 (epoch ~25)
  - Shows: Mid-training progress
  - Key info: Convergence trend, validation loss

- [ ] **10_training_completion.png**
  - Timing: ~6:00 (training done)
  - Shows: Training complete, early stopping
  - Key info: Total time, best model saved

- [ ] **11_train_metrics.png**
  - Timing: ~6:30
  - Shows: Train set evaluation results
  - Key info: MSE, R², MAE, correlation

- [ ] **12_test_metrics.png**
  - Timing: ~7:00
  - Shows: Test set metrics and per-frequency analysis
  - Key info: MSE, R², generalization gap

- [ ] **13_final_summary.png**
  - Timing: ~8:00 (completion)
  - Shows: Final summary, success message
  - Key info: Generalization status, file locations

---

### Phase 3: Generated Outputs (After execution)

**Location**: `03_outputs/`

- [ ] **14_directory_structure.png**
  - Command: `ls -la experiments/lstm_frequency_extraction_*/`
  - Shows: Experiment directory contents
  - Purpose: Verify all outputs generated

- [ ] **15_graph1_single_frequency_REQUIRED.png** ⭐⭐⭐
  - Source: `experiments/.../plots/graph1_single_frequency_f2.png`
  - Shows: Single frequency (3Hz) extraction
  - **CRITICAL**: Required by assignment
  - Quality: Full plot, clear labels, high resolution

- [ ] **16_graph2_all_frequencies_REQUIRED.png** ⭐⭐⭐
  - Source: `experiments/.../plots/graph2_all_frequencies.png`
  - Shows: All 4 frequencies in 2×2 grid
  - **CRITICAL**: Required by assignment
  - Quality: Full plot, all subplots visible

- [ ] **17_training_history.png**
  - Source: `experiments/.../plots/training_history.png`
  - Shows: Loss curves over epochs
  - Purpose: Demonstrate convergence

- [ ] **18_error_distribution.png**
  - Source: `experiments/.../plots/error_distribution.png`
  - Shows: Error histogram and distribution
  - Purpose: Verify error characteristics

- [ ] **19_metrics_comparison.png**
  - Source: `experiments/.../plots/metrics_comparison.png`
  - Shows: Train vs test metrics comparison
  - Purpose: Demonstrate generalization

- [ ] **20_cost_analysis.png** (optional)
  - Source: `experiments/.../cost_analysis/cost_dashboard.png`
  - Shows: Cost breakdown and recommendations
  - Purpose: Additional value-add

---

## 🎯 Priority Levels

### Critical (MUST HAVE) ⭐⭐⭐
1. **Graph 1**: Single frequency - Assignment requirement
2. **Graph 2**: All frequencies - Assignment requirement
3. **Final metrics**: Train & test MSE, generalization

### High Priority (SHOULD HAVE)
4. Execution start (shows setup)
5. Training completion (shows success)
6. Train metrics (shows performance)
7. Test metrics (shows generalization)

### Medium Priority (NICE TO HAVE)
8. Project structure
9. Configuration file
10. Training progress (mid-point)
11. Training history plot

### Low Priority (OPTIONAL)
12. Requirements file
13. Data generation logs
14. Model architecture details
15. Cost analysis

---

## 📏 Quality Standards

### Image Requirements
- **Format**: PNG (lossless)
- **Resolution**: Minimum 150 DPI
- **Size**: Readable when printed on A4
- **Content**: Complete (no cropping critical info)

### Terminal Screenshots
- **Font size**: 12-14pt minimum
- **Window width**: 80-120 characters
- **Theme**: Consistent (all light or all dark)
- **Visibility**: Text clearly readable

### Plot Screenshots
- **Quality**: Use source PNG files directly
- **Size**: Full plot (don't crop)
- **Labels**: All axes, titles, legends visible
- **Colors**: Clear and distinguishable

---

## 🔢 Naming Convention

### Format
```
[NUMBER]_[DESCRIPTION].png

Where:
- NUMBER: Sequential 01, 02, 03, etc.
- DESCRIPTION: Brief underscore-separated description
```

### Examples
- ✅ Good: `01_project_structure.png`
- ✅ Good: `15_graph1_single_frequency_REQUIRED.png`
- ❌ Bad: `screenshot.png`
- ❌ Bad: `Screenshot 2024-11-18 at 10.30.45.png`

### For Multiple Runs
```
[DATE]_[TIME]_[NUMBER]_[DESCRIPTION].png

Example:
- 20251118_103045_01_project_structure.png
- 20251118_145020_01_project_structure.png  (second run)
```

---

## 📝 Screenshot Captions Template

Create a `CAPTIONS.md` file with descriptions:

```markdown
# Screenshot Captions

## 01_project_structure.png
**When**: Pre-execution
**Command**: `tree -L 2`
**Shows**: Complete project directory structure
**Key Points**:
- All source files present in src/
- Configuration in config/
- Tests in tests/
- Documentation files at root

## 15_graph1_single_frequency_REQUIRED.png ⭐
**When**: Post-execution
**Source**: experiments/.../plots/graph1_single_frequency_f2.png
**Shows**: LSTM extraction of 3Hz frequency
**Key Points**:
- Blue line: Pure target sine wave (3Hz)
- Red line: LSTM predictions
- Gray background: Noisy mixed signal
- Demonstrates successful frequency extraction

[... continue for all screenshots ...]
```

---

## 🚀 Quick Capture Commands

### Setup
```bash
# Create directory structure
cd Assignment2_LSTM_extracting_frequences
mkdir -p screenshots/{01_pre_execution,02_execution,03_outputs,04_demo_outputs}
```

### Automated Naming
```bash
# macOS: Screenshot to numbered file
function ss() {
    local num=$(printf "%02d" $1)
    local desc=$2
    screencapture -x screenshots/${num}_${desc}.png
    echo "📸 Captured: ${num}_${desc}.png"
}

# Usage:
ss 01 project_structure
ss 02 configuration
# etc.
```

### Bulk Operations
```bash
# Copy all plots from experiment
cp experiments/lstm_frequency_extraction_*/plots/*.png screenshots/03_outputs/

# Rename with numbers
cd screenshots/03_outputs
mv graph1_single_frequency_f2.png 15_graph1_single_frequency_REQUIRED.png
mv graph2_all_frequencies.png 16_graph2_all_frequencies_REQUIRED.png
```

---

## ✅ Verification Checklist

Before considering screenshots complete:

### Completeness
- [ ] All minimum 10 screenshots present
- [ ] Graph 1 and Graph 2 present ⭐
- [ ] No missing critical screenshots
- [ ] All files properly named

### Quality
- [ ] All images high resolution (150+ DPI)
- [ ] Text readable in all screenshots
- [ ] No sensitive information visible
- [ ] Colors clear and distinguishable

### Organization
- [ ] Files in correct directories
- [ ] Sequential numbering correct
- [ ] Descriptive filenames
- [ ] README.md updated

### Documentation
- [ ] Captions written for key screenshots
- [ ] Index/list created
- [ ] Cross-referenced in main docs
- [ ] Highlighted critical outputs

---

## 📊 Screenshot Statistics

### Target Metrics
- **Minimum screenshots**: 10
- **Recommended screenshots**: 20-22
- **Critical screenshots**: 2 (Graph 1 & 2)
- **Storage estimate**: 20-50 MB total

### Time Estimates
- **Pre-execution**: 2 minutes (3 screenshots)
- **During execution**: 6 minutes (10 screenshots)
- **Post-execution**: 2 minutes (7+ screenshots)
- **Total time**: ~10 minutes

---

## 🔗 Related Documentation

- **Full Guide**: `../EXECUTION_AND_SCREENSHOT_GUIDE.md`
- **Quick Reference**: `../QUICK_SCREENSHOT_REFERENCE.md`
- **Execution Index**: `../EXECUTION_FLOWS_INDEX.md`

---

## 💡 Pro Tips

1. **Use window management**: Keep terminal and this checklist side-by-side
2. **Keyboard shortcuts**: Master your OS screenshot shortcuts
3. **Naming as you go**: Rename immediately to avoid confusion
4. **Backup originals**: Keep original experiment plots
5. **Document anomalies**: Note any unexpected outputs

---

## 🎓 For Assignment Submission

### Minimum Package
Include these screenshots in your submission:
1. Execution start → Shows setup
2. Training completion → Shows success
3. Final metrics → Shows performance
4. **Graph 1** → REQUIRED ⭐
5. **Graph 2** → REQUIRED ⭐

### Recommended Package  
Include additional screenshots:
- Project structure
- Configuration used
- Training progress
- Per-frequency analysis
- All generated plots

### Excellence Package
Include comprehensive documentation:
- All 22 screenshots
- Captions file
- Screenshot index
- Cross-referenced in docs
- Multiple run comparisons (optional)

---

## 📧 Submission Format

### Option 1: Separate Folder
```
submission/
├── code/
│   └── [all source files]
├── documentation/
│   └── [all .md files]
└── screenshots/
    └── [organized as above]
```

### Option 2: Integrated
```
Assignment2_LSTM_extracting_frequences/
├── [all files including source]
├── screenshots/
│   └── [organized as above]
└── experiments/
    └── [generated results]
```

### Option 3: Portfolio Document
Create a PDF with:
1. Cover page
2. Executive summary
3. Screenshots with captions
4. Results analysis
5. Conclusion

---

## 🏆 Success Indicators

Your screenshots are submission-ready when:

- ✅ All critical screenshots present (especially Graph 1 & 2)
- ✅ High quality and readable
- ✅ Properly organized and named
- ✅ Demonstrate successful execution
- ✅ Show metrics meeting requirements
- ✅ Cross-referenced in documentation

---

**Directory Version**: 1.0  
**Last Updated**: November 2025  
**Status**: ✅ Ready for Use

---

## Quick Reference

```bash
# Setup
mkdir -p screenshots/{01_pre_execution,02_execution,03_outputs}

# Capture flow
# 1. Pre-execution screenshots (3)
# 2. Run: python main.py
# 3. During-execution screenshots (10)
# 4. Copy output plots (7+)

# Verify
ls -R screenshots/
# Should see 20+ files across 3-4 directories

# Most important
open screenshots/03_outputs/15_graph1_*
open screenshots/03_outputs/16_graph2_*
# ⭐ These two are REQUIRED for assignment!
```

**Remember**: Graph 1 and Graph 2 are mandatory assignment requirements!

