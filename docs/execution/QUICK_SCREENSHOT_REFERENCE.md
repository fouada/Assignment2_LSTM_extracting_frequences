# Quick Screenshot Reference Guide
## Essential Screenshots for LSTM Frequency Extraction Project

**Quick Access Guide for Documentation and Presentations**

---

## 📸 Minimum Required Screenshots (10 Essential)

### For Assignment Submission

| # | Screenshot | Command | Location | Priority |
|---|------------|---------|----------|----------|
| 1 | Project Structure | `tree -L 2` or `ls -R` | Terminal | ⭐ Required |
| 2 | Configuration | `cat config/config.yaml` | Terminal | ⭐ Required |
| 3 | Execution Start | `python main.py` | Terminal | ⭐ Required |
| 4 | Training Progress | During epoch ~25 | Terminal | ⭐ Required |
| 5 | Training Complete | End of training | Terminal | ⭐ Required |
| 6 | Train Metrics | After evaluation | Terminal | ⭐ Required |
| 7 | Test Metrics | After evaluation | Terminal | ⭐ Required |
| 8 | **Graph 1** | Open PNG file | `experiments/.../plots/graph1_single_frequency_f2.png` | ⭐⭐⭐ CRITICAL |
| 9 | **Graph 2** | Open PNG file | `experiments/.../plots/graph2_all_frequencies.png` | ⭐⭐⭐ CRITICAL |
| 10 | Final Summary | End of execution | Terminal | ⭐ Required |

**Graph 1 and Graph 2 are MANDATORY per assignment requirements!**

---

## 🎯 Screenshot Workflow

### Phase 1: Pre-Execution (2 minutes)

```bash
# Terminal 1: Project overview
cd Assignment2_LSTM_extracting_frequences
ls -la
📸 Screenshot: Project root directory

# View configuration
cat config/config.yaml
📸 Screenshot: Full config file

# View structure
tree -L 2  # or: find . -maxdepth 2 -type d
📸 Screenshot: Project structure
```

---

### Phase 2: Execution (7 minutes)

```bash
# Start execution
python main.py

# 📸 Screenshot moments:

# Moment 1: Immediately after start (~5 seconds)
# Shows: Configuration loading, device selection
📸 Screenshot: "Execution start"

# Moment 2: After data generation (~10 seconds)
# Shows: Data generation complete, dataset created
📸 Screenshot: "Data pipeline"

# Moment 3: After model creation (~12 seconds)
# Shows: Model architecture, parameter count
📸 Screenshot: "Model initialization"

# Moment 4: During training epoch ~10 (~2 minutes)
# Shows: Early training progress
📸 Screenshot: "Training early"

# Moment 5: During training epoch ~25 (~3-4 minutes)
# Shows: Mid-training progress
📸 Screenshot: "Training mid"

# Moment 6: Training completion (~6 minutes)
# Shows: Early stopping, best model saved, training time
📸 Screenshot: "Training complete"

# Moment 7: After train evaluation (~6.5 minutes)
# Shows: Train set metrics (MSE, R², etc.)
📸 Screenshot: "Train metrics"

# Moment 8: After test evaluation (~7 minutes)
# Shows: Test set metrics and per-frequency analysis
📸 Screenshot: "Test metrics"

# Moment 9: After visualization (~7.5 minutes)
# Shows: Plot generation confirmation
📸 Screenshot: "Visualization complete"

# Moment 10: Final summary (~8 minutes)
# Shows: Final metrics, generalization status, success message
📸 Screenshot: "Final summary"
```

---

### Phase 3: Generated Outputs (2 minutes)

```bash
# Navigate to experiment directory
cd experiments/
ls -la
📸 Screenshot: Experiment directory listing

# View latest experiment
cd lstm_frequency_extraction_YYYYMMDD_HHMMSS/
ls -la
📸 Screenshot: Experiment structure

# Open plots (⭐⭐⭐ CRITICAL)
open plots/graph1_single_frequency_f2.png
📸 Screenshot: GRAPH 1 - Single frequency comparison

open plots/graph2_all_frequencies.png
📸 Screenshot: GRAPH 2 - All frequencies grid

# Optional additional plots
open plots/training_history.png
open plots/error_distribution.png
open plots/metrics_comparison.png
```

---

## 🚀 Fast Screenshot Commands

### macOS One-Liners

```bash
# Full execution with automatic timing hints
python main.py 2>&1 | tee execution.log &
PID=$!

# At key moments (run in separate terminal):
sleep 5  && screencapture -x screenshots/01_start.png
sleep 15 && screencapture -x screenshots/02_data.png
sleep 120 && screencapture -x screenshots/03_training_early.png
sleep 240 && screencapture -x screenshots/04_training_mid.png

# Wait for completion
wait $PID
screencapture -x screenshots/05_final.png
```

### Linux One-Liners

```bash
# Using flameshot or gnome-screenshot
gnome-screenshot -f screenshots/01_start.png
# or
flameshot gui
```

### Windows One-Liners

```powershell
# Using Snipping Tool
# Win + Shift + S (then select region)
```

---

## 📋 Screenshot Checklist Template

Print this and check off as you go:

```
BEFORE EXECUTION:
[ ] Project structure
[ ] Configuration file
[ ] Requirements.txt

DURING EXECUTION:
[ ] Execution start
[ ] Data generation
[ ] Model initialization  
[ ] Training early (epoch ~10)
[ ] Training mid (epoch ~25)
[ ] Training completion
[ ] Train set metrics
[ ] Test set metrics
[ ] Visualization confirmation
[ ] Final summary

GENERATED OUTPUTS:
[ ] Experiment directory structure
[ ] ⭐⭐⭐ GRAPH 1: Single frequency (f2=3Hz)
[ ] ⭐⭐⭐ GRAPH 2: All frequencies (2×2 grid)
[ ] Training history plot
[ ] Error distribution
[ ] Metrics comparison

OPTIONAL:
[ ] Cost analysis summary
[ ] TensorBoard dashboard
[ ] Cost dashboards
```

---

## 💡 Pro Tips

### Tip 1: Use Screen Recording
Instead of multiple screenshots, record the entire execution:

```bash
# macOS
# Cmd + Shift + 5 → Record Selected Portion

# Or terminal recording
asciinema rec execution.cast
python main.py
# Ctrl+D to stop

# Play back
asciinema play execution.cast
```

### Tip 2: Dual Monitor Setup
- Monitor 1: Run execution
- Monitor 2: Open this guide and check off items

### Tip 3: Terminal Multiplexer
```bash
# Use tmux for easy scrolling and screenshot capture
tmux new -s lstm_training
python main.py

# In another terminal, attach and navigate
tmux attach -t lstm_training
# Use Ctrl+B then [ to scroll back
```

### Tip 4: Screenshot Naming Convention
```
01_20251118_103045_project_structure.png
02_20251118_103050_execution_start.png
03_20251118_103055_data_generation.png
...

Format: NUMBER_TIMESTAMP_DESCRIPTION.png
```

### Tip 5: Batch Screenshot Taking
```bash
# macOS: Set up keyboard shortcut for screencapture
# System Preferences → Keyboard → Shortcuts → Screenshots
# Assign: Cmd + Shift + 4 → Capture region to file

# Create hotkey script:
echo '#!/bin/bash
screencapture -x ~/screenshots/$(date +%Y%m%d_%H%M%S).png' > ~/bin/quick_screenshot.sh
chmod +x ~/bin/quick_screenshot.sh

# Assign to hotkey with Automator or BetterTouchTool
```

---

## 🎨 Screenshot Quality Guidelines

### Terminal Screenshots
- **Font size**: 12-14pt (readable when printed)
- **Theme**: Dark or light with good contrast
- **Width**: 80-120 characters (avoid line wrapping)
- **Include**: Full command and relevant output
- **Exclude**: Unnecessary personal info (username, paths)

### Plot Screenshots
- **Resolution**: Minimum 150 DPI (plots are saved at 150 DPI)
- **Format**: PNG (lossless)
- **Full image**: Include title, axes, labels, legend
- **No cropping**: Show entire plot

### Recommended Settings
```python
# Already set in plotter.py:
plt.savefig('plot.png', dpi=150, bbox_inches='tight')
```

---

## 📦 Organizing Screenshots

### Folder Structure
```
screenshots/
├── README.md (this file)
├── 01_pre_execution/
│   ├── 01_project_structure.png
│   ├── 02_configuration.png
│   └── 03_requirements.png
├── 02_execution/
│   ├── 04_start.png
│   ├── 05_data.png
│   ├── 06_model.png
│   ├── 07_training_early.png
│   ├── 08_training_mid.png
│   ├── 09_training_complete.png
│   ├── 10_train_metrics.png
│   ├── 11_test_metrics.png
│   └── 12_final.png
└── 03_outputs/
    ├── 13_directory.png
    ├── 14_graph1_REQUIRED.png  ⭐⭐⭐
    ├── 15_graph2_REQUIRED.png  ⭐⭐⭐
    ├── 16_training_history.png
    ├── 17_error_distribution.png
    └── 18_metrics_comparison.png
```

### Create Structure
```bash
mkdir -p screenshots/{01_pre_execution,02_execution,03_outputs}
```

---

## 🎯 For Different Purposes

### For Assignment Submission
**Minimum**: 10 screenshots
- Focus on: Requirements satisfaction
- Must include: Graph 1 & Graph 2 ⭐
- Include: All metrics demonstrating success

### For Presentation (15 min)
**Optimal**: 5-7 slides with screenshots
1. Title + Project structure
2. Configuration highlights
3. Training progress (1-2 epochs)
4. Results: Metrics table
5. **Graph 1** ⭐
6. **Graph 2** ⭐
7. Conclusion

### For Documentation
**Comprehensive**: 20-25 screenshots
- Cover all 7 flows
- Include errors (if any)
- Show troubleshooting steps
- Demonstrate understanding

### For Quick Demo (5 min)
**Minimal**: 3 screenshots
1. Execution command + start
2. Final metrics
3. Graph 2 (all frequencies) ⭐

---

## 📱 Mobile/Tablet Screenshots

If presenting from tablet/phone:
1. Transfer plots to device
2. Use full-screen viewer
3. Screenshot individual plots
4. Ensure high resolution

**AirDrop (Mac to iPhone)**:
```bash
# Select files in Finder
# Right-click → Share → AirDrop
```

---

## ⚡ Emergency Quick Capture

If you forgot to take screenshots during execution:

```bash
# Run again (fast mode)
# Edit config.yaml:
training:
  epochs: 5  # Quick run

# Run
python main.py 2>&1 | tee quick_run.log

# Extract from log
cat quick_run.log | grep "STEP"
cat quick_run.log | grep "MSE"
cat quick_run.log | grep "SUCCESS"

# Graphs are still generated!
open experiments/lstm_frequency_extraction_*/plots/graph*.png
```

---

## 🏆 Complete Screenshot Session (15 minutes)

**Optimal workflow** for capturing everything:

```bash
# Setup
cd Assignment2_LSTM_extracting_frequences
mkdir -p screenshots/{pre,during,post}

# PRE-EXECUTION (2 min)
tree -L 2 > /tmp/structure.txt
cat /tmp/structure.txt
# 📸 Screenshot → screenshots/pre/01_structure.png

cat config/config.yaml
# 📸 Screenshot → screenshots/pre/02_config.png

# EXECUTION (8 min)
python main.py 2>&1 | tee execution.log

# During execution:
# 📸 At start → screenshots/during/03_start.png
# 📸 At epoch 10 → screenshots/during/04_early.png
# 📸 At epoch 25 → screenshots/during/05_mid.png
# 📸 At completion → screenshots/during/06_complete.png
# 📸 At train eval → screenshots/during/07_train.png
# 📸 At test eval → screenshots/during/08_test.png
# 📸 At final → screenshots/during/09_final.png

# POST-EXECUTION (5 min)
cd experiments/lstm_frequency_extraction_*/

ls -la
# 📸 Screenshot → ../../screenshots/post/10_directory.png

# Open and screenshot each plot
open plots/graph1_single_frequency_f2.png
# 📸 Screenshot → ../../screenshots/post/11_graph1_REQUIRED.png

open plots/graph2_all_frequencies.png
# 📸 Screenshot → ../../screenshots/post/12_graph2_REQUIRED.png

# Done! 12 essential screenshots captured
```

---

## 📚 Additional Resources

### Screenshot Tools
- **macOS**: Built-in (Cmd+Shift+4)
- **Windows**: Snipping Tool, Snip & Sketch
- **Linux**: Flameshot, Shutter, gnome-screenshot
- **Cross-platform**: ShareX, Lightshot

### Terminal Recording
- **asciinema**: Terminal session recording
- **ttyrec**: Terminal recording
- **script**: Built-in Unix command

### Screen Recording
- **macOS**: QuickTime, built-in screen recording
- **Windows**: Game Bar (Win+G), OBS
- **Linux**: SimpleScreenRecorder, Kazam, OBS

---

## ✅ Final Checklist

Before submitting, verify you have:

- [ ] All required screenshots (minimum 10)
- [ ] **Graph 1** and **Graph 2** (MANDATORY) ⭐⭐⭐
- [ ] Screenshots clearly show success
- [ ] Images are high quality (readable text)
- [ ] Files are organized
- [ ] Filenames are descriptive
- [ ] Screenshots reference experiment timestamp
- [ ] All metrics visible (MSE, R², generalization)

**Most Important**: GRAPH 1 and GRAPH 2 are assignment requirements!

---

**Document Version**: 1.0  
**Authors**: Fouad Azem & Tal Goldengorn  
**Last Updated**: November 2025

