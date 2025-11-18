# Execution Flows Documentation Index
## LSTM Frequency Extraction System

**Complete Guide to Running, Documenting, and Screenshotting All System Flows**

**Authors**: Fouad Azem (040830861) & Tal Goldengorn (207042573)  
**Course**: LLM And Multi Agent Orchestration
**Date**: November 2025

---

## 📚 Documentation Overview

This index provides quick access to all documentation related to executing and documenting the LSTM Frequency Extraction system flows.

**🎯 NEW**: See `COMPLETE_FEATURES_EXECUTION_GUIDE.md` for ALL 5 execution modes and 24+ features!

---

## 🎯 Quick Start Guide

### For First-Time Users
1. **Read**: `COMPLETE_FEATURES_EXECUTION_GUIDE.md` ⭐ (ALL features)
   - Or: `EXECUTION_AND_SCREENSHOT_GUIDE.md` (Mode 1 only)
2. **Quick Ref**: `QUICK_SCREENSHOT_REFERENCE.md` (Essential screenshots)
3. **Execute**: `python main.py` (Mode 1: Basic)
4. **Capture**: Follow screenshot checklist

### Advanced Users - Try Other Modes!
- **Mode 2**: `python main_with_dashboard.py` (Interactive dashboard)
- **Mode 3**: `python main_production.py` (Production features)
- **Mode 4**: `python demo_innovations.py` (5 innovations)
- **Mode 5**: `python cost_analysis_report.py` (Cost analysis)

### For Quick Demo
1. **Read**: `QUICK_SCREENSHOT_REFERENCE.md`
2. **Run**: `python test_data_generation.py` (2 min demo)
3. **Run**: `python test_model_creation.py` (1 min demo)
4. **Run**: `python main.py` (7 min full execution)

### For Assignment Submission
1. **Execute**: `python main.py`
2. **Capture**: 10 essential screenshots (see checklist)
3. **Required**: Graph 1 & Graph 2 ⭐⭐⭐
4. **Reference**: All documentation in this package

---

## 📖 Documentation Files

### Main Documentation

| Document | Purpose | Read Time | When to Use |
|----------|---------|-----------|-------------|
| **`COMPLETE_FEATURES_EXECUTION_GUIDE.md`** ⭐ NEW | ALL 5 modes, 24+ features | 40 min | Complete feature coverage |
| **`EXECUTION_AND_SCREENSHOT_GUIDE.md`** | Mode 1 (basic) execution guide | 30 min | First time, standard training |
| **`QUICK_SCREENSHOT_REFERENCE.md`** | Quick screenshot checklist | 5 min | During execution, quick ref |
| **`EXECUTION_FLOWS_INDEX.md`** | This file - navigation hub | 3 min | Finding documentation |

### Supporting Scripts

| Script | Purpose | Runtime | Output |
|--------|---------|---------|--------|
| **`test_data_generation.py`** | Demo data generation flow standalone | ~30 sec | 3 plots showing signals |
| **`test_model_creation.py`** | Demo model architecture standalone | ~5 sec | Model specs and tests |
| **`main.py`** | Full pipeline execution | ~7 min | Complete results |

---

## 🔄 System Flows Overview

### 7 Major Execution Flows

| Flow | Name | Duration | Key Outputs | Screenshot Priority |
|------|------|----------|-------------|---------------------|
| **1** | Data Generation | ~5 sec | Train/test generators | Medium |
| **2** | Dataset Creation | ~3 sec | PyTorch datasets (40k samples) | Medium |
| **3** | Model Creation | ~1 sec | LSTM model (215k params) | High |
| **4** | Model Training | ~5-7 min | Trained model checkpoint | **Critical** |
| **5** | Model Evaluation | ~30 sec | Train/test metrics | **Critical** |
| **6** | Visualization | ~10 sec | Required graphs | **Critical** |
| **7** | Cost Analysis | ~15 sec | Cost breakdown | Optional |

**Total Runtime**: ~7-10 minutes

---

## 📸 Screenshot Guide Summary

### Minimum Required (10 Screenshots)

#### Pre-Execution (3 screenshots)
1. Project structure (`tree -L 2`)
2. Configuration file (`cat config/config.yaml`)
3. Requirements (`cat requirements.txt`)

#### During Execution (5 screenshots)
4. Execution start
5. Training progress (epoch ~25)
6. Training completion
7. Train metrics
8. Test metrics + generalization

#### Generated Outputs (2 screenshots) ⭐ CRITICAL
9. **Graph 1**: Single frequency (f2=3Hz) - REQUIRED by assignment
10. **Graph 2**: All frequencies grid - REQUIRED by assignment

### Complete Documentation (22 Screenshots)

See `EXECUTION_AND_SCREENSHOT_GUIDE.md` Section: "What to Screenshot" for detailed breakdown.

---

## 🚀 Execution Methods

### Method 1: Full Pipeline (Recommended)
```bash
cd Assignment2_LSTM_extracting_frequences
python main.py
```
**Output**: All 7 flows, complete results in `experiments/`

### Method 2: Quick Demo Flows
```bash
# Flow 1-2: Data generation demo
python test_data_generation.py

# Flow 3: Model architecture demo  
python test_model_creation.py
```
**Output**: Quick visualizations and architecture specs

### Method 3: Individual Flow Testing
See `EXECUTION_AND_SCREENSHOT_GUIDE.md` Section: "Flow-by-Flow Execution"

---

## 📋 Quick Reference Tables

### Screenshot Timing Guide

| Time | Event | Action |
|------|-------|--------|
| 0:00 | Start execution | 📸 Capture command |
| 0:05 | Data loaded | 📸 Capture confirmation |
| 0:12 | Model created | 📸 Capture architecture |
| 2:00 | Epoch ~10 | 📸 Capture training progress |
| 4:00 | Epoch ~25 | 📸 Capture mid-training |
| 6:00 | Training done | 📸 Capture completion |
| 6:30 | Train eval | 📸 Capture train metrics |
| 7:00 | Test eval | 📸 Capture test metrics |
| 7:30 | Plots ready | 📸 Open and capture Graph 1 & 2 |
| 8:00 | Complete | 📸 Capture final summary |

### File Locations Quick Reference

| Output | Location | Format |
|--------|----------|--------|
| **Graph 1** ⭐ | `experiments/.../plots/graph1_single_frequency_f2.png` | PNG (150 DPI) |
| **Graph 2** ⭐ | `experiments/.../plots/graph2_all_frequencies.png` | PNG (150 DPI) |
| Training history | `experiments/.../plots/training_history.png` | PNG |
| Error dist | `experiments/.../plots/error_distribution.png` | PNG |
| Metrics | `experiments/.../plots/metrics_comparison.png` | PNG |
| Best model | `experiments/.../checkpoints/best_model.pt` | PyTorch |
| Config copy | `experiments/.../config.yaml` | YAML |
| TensorBoard | `experiments/.../checkpoints/tensorboard/` | TFEvents |

---

## 🎓 For Assignment Submission

### Required Deliverables Checklist

#### Code & Configuration
- [ ] Complete source code (`src/`)
- [ ] Main execution script (`main.py`)
- [ ] Configuration file (`config/config.yaml`)
- [ ] Requirements (`requirements.txt`)
- [ ] Tests (`tests/`)

#### Documentation
- [ ] README.md
- [ ] ARCHITECTURE.md
- [ ] PRODUCT_REQUIREMENTS_DOCUMENT.md
- [ ] DEVELOPMENT_PROMPTS_LOG.md (instructor requirement)
- [ ] EXECUTION_AND_SCREENSHOT_GUIDE.md
- [ ] This INDEX file

#### Execution Results
- [ ] Training completion log
- [ ] Final metrics (train MSE, test MSE, R²)
- [ ] Generalization analysis (< 10% gap)
- [ ] **Graph 1**: Single frequency ⭐ REQUIRED
- [ ] **Graph 2**: All frequencies ⭐ REQUIRED

#### Screenshots (Minimum 10)
- [ ] Project structure
- [ ] Configuration
- [ ] Execution progress
- [ ] Final metrics
- [ ] **Both required graphs** ⭐⭐

### Success Criteria Verification

| Criterion | Target | How to Verify | Screenshot |
|-----------|--------|---------------|------------|
| Train MSE | < 0.01 | Check terminal output after train eval | ✅ Required |
| Test MSE | < 0.01 | Check terminal output after test eval | ✅ Required |
| Generalization | < 10% gap | Check generalization analysis | ✅ Required |
| Graph 1 | Single freq plot | Open `graph1_single_frequency_f2.png` | ⭐ Critical |
| Graph 2 | All freq grid | Open `graph2_all_frequencies.png` | ⭐ Critical |
| Code runs | No errors | Complete execution without crashes | ✅ Required |
| Tests pass | All green | Run `pytest tests/ -v` | Optional |

---

## 💡 Tips & Best Practices

### For Screenshots
1. **Use high resolution**: 150+ DPI for clarity
2. **Consistent naming**: `01_description.png`, `02_description.png`
3. **Organize folders**: Pre-execution, during, post
4. **Include timestamps**: Show when each was taken
5. **Full context**: Show complete command and output

### For Execution
1. **Clean terminal**: `clear` before starting
2. **Readable font**: 12-14pt minimum
3. **Full screen**: Avoid line wrapping
4. **Save logs**: `python main.py 2>&1 | tee execution.log`
5. **Note timing**: Write down when to screenshot

### For Documentation
1. **Reference screenshots**: Caption each image
2. **Explain outputs**: Don't just show, explain
3. **Cross-reference**: Link between documents
4. **Version control**: Keep track of runs
5. **Backup results**: Save experiment directories

---

## 🔧 Troubleshooting

### Common Issues

| Issue | Solution | Reference |
|-------|----------|-----------|
| Module not found | Check directory, install requirements | Guide p.42 |
| CUDA/MPS error | Set `device: cpu` in config | Guide p.43 |
| Out of memory | Reduce batch size to 16 | Guide p.43 |
| Plots not generated | Check experiment directory created | Guide p.43 |
| Screenshots blurry | Use 150+ DPI, larger window | Quick Ref p.8 |

### Quick Fixes

```bash
# Reset environment
cd Assignment2_LSTM_extracting_frequences
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Quick test run (5 epochs)
# Edit config.yaml: epochs: 5
python main.py

# Check outputs
ls experiments/lstm_frequency_extraction_*/plots/
```

---

## 📊 Flow Execution Matrix

### What Happens in Each Flow

| Flow | Input | Processing | Output | Validates |
|------|-------|------------|--------|-----------|
| **1. Data Gen** | Config params | Generate noisy signals | Train/test generators | Different seeds work |
| **2. Dataset** | Generators | Create PyTorch datasets | 40k samples | One-hot encoding correct |
| **3. Model** | Model config | Initialize LSTM | 215k param model | Architecture correct |
| **4. Training** | Data + Model | Train with state mgmt | Trained weights | Convergence achieved |
| **5. Evaluation** | Test data + Model | Compute metrics | MSE, R², etc. | Generalization good |
| **6. Visualization** | Predictions + Targets | Generate plots | PNG files | Visual quality |
| **7. Cost** | Training logs | Analyze costs | Recommendations | Efficiency |

---

## 🎬 Demo Scripts Usage

### Quick Demo Sequence (5 minutes)

```bash
# 1. Show data generation (30 seconds)
python test_data_generation.py
# Open: demo_outputs/data_generation_demo.png

# 2. Show model architecture (10 seconds)  
python test_model_creation.py
# Scroll through terminal output

# 3. Show pre-trained results (if available)
cd experiments/lstm_frequency_extraction_LATEST/
open plots/graph2_all_frequencies.png

# Total time: ~1 minute
```

### Full Demo Sequence (15 minutes)

```bash
# 1. Data demo (2 min)
python test_data_generation.py
# Review all 3 generated plots

# 2. Model demo (2 min)
python test_model_creation.py  
# Review architecture and tests

# 3. Full training (10 min with reduced epochs)
# Edit config: epochs: 10
python main.py

# 4. Review results (1 min)
cd experiments/lstm_frequency_extraction_LATEST/plots/
open *.png
```

---

## 📚 Related Documentation

### Core Documentation
- `README.md` - Project overview
- `ARCHITECTURE.md` - System design
- `PRODUCT_REQUIREMENTS_DOCUMENT.md` - Complete PRD

### Development Documentation
- `DEVELOPMENT_PROMPTS_LOG.md` - CLI prompts (instructor requirement)
- `SUBMISSION_PACKAGE.md` - Submission overview
- `INSTRUCTOR_QUICK_REVIEW.md` - Review guide for instructor

### Execution Documentation (You Are Here)
- **`EXECUTION_AND_SCREENSHOT_GUIDE.md`** - Full guide
- **`QUICK_SCREENSHOT_REFERENCE.md`** - Quick reference
- **`EXECUTION_FLOWS_INDEX.md`** - This file

---

## 🏆 Success Metrics

### Execution Success
- ✅ All 7 flows complete without errors
- ✅ Training converges (loss decreases)
- ✅ Generalization gap < 10%
- ✅ Both required graphs generated

### Documentation Success  
- ✅ All 10 minimum screenshots captured
- ✅ Graph 1 and Graph 2 high quality
- ✅ Metrics clearly visible
- ✅ Execution flow documented

### Assignment Success
- ✅ Train MSE < 0.01
- ✅ Test MSE < 0.01
- ✅ R² > 0.95
- ✅ Graphs show clear frequency extraction
- ✅ Complete documentation package

---

## 🎯 Quick Decision Tree

### "Which document should I read?"

```
Are you running the system for the first time?
├─ YES → Read: EXECUTION_AND_SCREENSHOT_GUIDE.md
└─ NO
   └─ Do you need a quick screenshot checklist?
      ├─ YES → Read: QUICK_SCREENSHOT_REFERENCE.md
      └─ NO
         └─ Do you want to demo individual flows?
            ├─ YES → Run: test_data_generation.py or test_model_creation.py
            └─ NO → You're ready! Run: python main.py
```

### "What should I screenshot?"

```
Is this for assignment submission?
├─ YES
│  └─ Minimum 10 screenshots + Graph 1 & 2 (REQUIRED)
│     See: QUICK_SCREENSHOT_REFERENCE.md
└─ NO
   └─ Is this for comprehensive documentation?
      ├─ YES → All 22 screenshots
      │        See: EXECUTION_AND_SCREENSHOT_GUIDE.md
      └─ NO → Is this for quick demo?
             └─ YES → 3 screenshots (command, metrics, graph 2)
```

---

## 📞 Support & Resources

### Documentation Issues
- Check: `DOCUMENTATION_INDEX.md` for all docs
- Review: Section headings in guides
- Search: Use Cmd+F / Ctrl+F in markdown

### Execution Issues
- Check: Troubleshooting section in full guide
- Review: `training.log` for errors
- Test: Run demo scripts first

### Screenshot Issues  
- Check: Quality guidelines in quick reference
- Tool: Use recommended screenshot tools
- Format: PNG at 150+ DPI

---

## ✅ Final Checklist

Before considering execution documentation complete:

### Execution
- [ ] `python main.py` runs successfully
- [ ] All 7 flows complete
- [ ] No errors in output
- [ ] Experiment directory created
- [ ] All plots generated

### Screenshots
- [ ] Minimum 10 screenshots captured
- [ ] **Graph 1** high quality ⭐
- [ ] **Graph 2** high quality ⭐
- [ ] Metrics clearly visible
- [ ] Files organized in folders

### Documentation
- [ ] Screenshots have descriptive names
- [ ] Created screenshot index/list
- [ ] Added captions to images
- [ ] Cross-referenced in main docs

### Verification
- [ ] Reviewed all screenshots for clarity
- [ ] Verified metrics meet requirements
- [ ] Checked graphs show correct frequencies
- [ ] Confirmed no personal info exposed

---

## 🎓 Learning Outcomes

By completing the execution and screenshot documentation, you will have:

1. ✅ **Executed all 7 system flows** successfully
2. ✅ **Documented the execution process** with screenshots
3. ✅ **Verified system performance** against requirements
4. ✅ **Generated required outputs** (Graph 1 & 2)
5. ✅ **Demonstrated system understanding** through comprehensive docs
6. ✅ **Created submission package** ready for instructor review

---

## 🚀 Next Steps

### After Execution
1. Review all screenshots for quality
2. Organize files for submission
3. Create final submission package
4. Review against rubric
5. Submit with confidence!

### For Future Runs
1. Archive experiment results
2. Note any improvements needed
3. Document any issues encountered
4. Update configuration if needed
5. Maintain documentation

---

## 📖 Document Relationships

```
START_HERE.txt
     │
     ├─→ SUBMISSION_PACKAGE.md (Submission overview)
     │
     ├─→ PRODUCT_REQUIREMENTS_DOCUMENT.md (Complete specs)
     │
     ├─→ DEVELOPMENT_PROMPTS_LOG.md (Development process)
     │
     ├─→ EXECUTION_FLOWS_INDEX.md (THIS FILE)
     │        │
     │        ├─→ EXECUTION_AND_SCREENSHOT_GUIDE.md (Full guide)
     │        │
     │        ├─→ QUICK_SCREENSHOT_REFERENCE.md (Quick ref)
     │        │
     │        ├─→ test_data_generation.py (Demo script)
     │        │
     │        └─→ test_model_creation.py (Demo script)
     │
     └─→ INSTRUCTOR_QUICK_REVIEW.md (Evaluation guide)
```

---

## 🎯 Key Takeaways

1. **Three main documents** for execution:
   - Full guide (comprehensive)
   - Quick reference (checklists)
   - This index (navigation)

2. **Two demo scripts** for quick testing:
   - Data generation demo
   - Model architecture demo

3. **Ten minimum screenshots** required:
   - 3 pre-execution
   - 5 during execution
   - 2 outputs (Graph 1 & 2) ⭐

4. **Seven execution flows** total:
   - Data → Model → Training → Evaluation → Visualization
   - Plus optional cost analysis

5. **One main command**: `python main.py`
   - Runs all flows
   - ~7-10 minutes
   - Generates all outputs

---

**Document Version**: 1.0  
**Last Updated**: November 2025  
**Authors**: Fouad Azem & Tal Goldengorn  
**Status**: ✅ Complete and Ready for Use

---

## Quick Access Links

- **Full Execution Guide**: [EXECUTION_AND_SCREENSHOT_GUIDE.md](./EXECUTION_AND_SCREENSHOT_GUIDE.md)
- **Quick Screenshot Reference**: [QUICK_SCREENSHOT_REFERENCE.md](./QUICK_SCREENSHOT_REFERENCE.md)
- **Data Generation Demo**: [test_data_generation.py](./test_data_generation.py)
- **Model Creation Demo**: [test_model_creation.py](./test_model_creation.py)
- **Main Execution**: [main.py](./main.py)
- **Configuration**: [config/config.yaml](./config/config.yaml)

**For questions, see the Troubleshooting section in EXECUTION_AND_SCREENSHOT_GUIDE.md**

