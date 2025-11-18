# Complete Features Execution Guide
## All Execution Modes and Advanced Features

**LSTM Frequency Extraction System - Professional Implementation**

**Authors**: Fouad Azem & Tal Goldengorn  
**Date**: November 2025  
**Status**: Complete Feature Coverage

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Execution Modes](#execution-modes)
3. [Core Features (7 Basic Flows)](#core-features-7-basic-flows)
4. [Advanced Features](#advanced-features)
5. [Innovation Features](#innovation-features)
6. [Production Features](#production-features)
7. [Quick Reference Matrix](#quick-reference-matrix)
8. [Screenshot Guide by Feature](#screenshot-guide-by-feature)

---

## Overview

Your project supports **5 execution modes** with **20+ features** across multiple categories:

### Execution Modes

| Mode | Script | Purpose | Features | Duration |
|------|--------|---------|----------|----------|
| **1. Basic** | `main.py` | Standard training pipeline | Core 7 flows | ~7 min |
| **2. Dashboard** | `main_with_dashboard.py` | Real-time monitoring | Live visualization | ~7 min + server |
| **3. Production** | `main_production.py` | Plugin architecture | Event-driven, extensible | ~7 min |
| **4. Innovations** | `demo_innovations.py` | Advanced ML features | 5 cutting-edge models | ~15 min |
| **5. Cost Analysis** | `cost_analysis_report.py` | Standalone cost reports | Detailed recommendations | ~1 min |

### Feature Categories

1. **Core Features** (7 flows) - Basic training pipeline
2. **Visualization Features** (3 types) - Static, interactive, live
3. **Advanced ML Features** (5 innovations) - Attention, uncertainty, hybrid, etc.
4. **Production Features** (4 systems) - Plugins, events, hooks, registry
5. **Analysis Features** (2 types) - Cost analysis, research analysis
6. **Quality Features** (3 systems) - Compliance, testing, linting

**Total**: 24+ executable features

---

## Execution Modes

### Mode 1: Basic Training Pipeline ⭐ (START HERE)

**Script**: `main.py`

**What it does**: Complete training pipeline with all core flows

**Command**:
```bash
cd Assignment2_LSTM_extracting_frequences
python main.py
```

**Features Included**:
- ✅ Flow 1: Data Generation
- ✅ Flow 2: Dataset Creation
- ✅ Flow 3: Model Initialization
- ✅ Flow 4: Training with State Management
- ✅ Flow 5: Evaluation (Train & Test)
- ✅ Flow 6: Visualization (5 plots)
- ✅ Flow 7: Cost Analysis (optional)

**Output**:
```
experiments/lstm_frequency_extraction_YYYYMMDD_HHMMSS/
├── plots/
│   ├── graph1_single_frequency_f2.png  ⭐ REQUIRED
│   ├── graph2_all_frequencies.png       ⭐ REQUIRED
│   ├── training_history.png
│   ├── error_distribution.png
│   └── metrics_comparison.png
├── checkpoints/
│   ├── best_model.pt
│   ├── last_model.pt
│   └── tensorboard/
└── config.yaml
```

**Duration**: ~7 minutes

**When to use**:
- ✅ First execution
- ✅ Assignment submission
- ✅ Standard training
- ✅ Capturing required graphs

**Screenshots needed**: 10 minimum (see QUICK_SCREENSHOT_REFERENCE.md)

---

### Mode 2: Interactive Dashboard Training 🌐

**Script**: `main_with_dashboard.py`

**What it does**: Training + real-time web dashboard monitoring

**Commands**:
```bash
# Full training with dashboard
python main_with_dashboard.py

# Train without dashboard (same as main.py)
python main_with_dashboard.py --no-dashboard

# Dashboard only for existing experiment
python main_with_dashboard.py --dashboard-only

# Custom port
python main_with_dashboard.py --port 8080
```

**Features Included**:
- ✅ All 7 core flows
- ✅ **Real-time training monitoring**
- ✅ **Interactive plots** (zoom, pan, hover)
- ✅ **Live metric updates**
- ✅ **Per-frequency analysis** (interactive)
- ✅ **Training progress** (epoch-by-epoch)

**Dashboard Access**:
```
Open browser: http://localhost:8050

Features:
- 📊 Live training curves
- 📈 Interactive prediction plots
- 🎯 Per-frequency metrics
- 🔍 Zoom into any time range
- 💾 Export data to CSV
- 📸 Download plots
```

**Output**:
- All outputs from Mode 1
- Plus: Dashboard server running in background
- Plus: JSON data exports for dashboard

**Duration**: ~7 minutes training + continuous dashboard

**When to use**:
- ✅ Live demo presentations
- ✅ Real-time monitoring during training
- ✅ Interactive result exploration
- ✅ Debugging training issues
- ✅ Comparing different frequency extractions

**Screenshots needed**: 15-20
- All Mode 1 screenshots
- Dashboard interface
- Interactive features
- Live updates

**Pro tip**: Open dashboard in browser while training to see real-time updates!

---

### Mode 3: Production Framework 🏭

**Script**: `main_production.py`

**What it does**: Professional plugin-based architecture with event system

**Command**:
```bash
python main_production.py
```

**Features Included**:
- ✅ All 7 core flows
- ✅ **Plugin System**: Modular, extensible architecture
- ✅ **Event System**: Event-driven architecture
- ✅ **Hook System**: Customization points
- ✅ **Component Registry**: Centralized component management
- ✅ **Dependency Injection**: Loose coupling

**Built-in Plugins**:
1. **TensorBoard Plugin**: Advanced logging
2. **Early Stopping Plugin**: Intelligent stopping
3. **Custom Metrics Plugin**: Extended evaluation
4. **Data Augmentation Plugin**: Enhanced data processing

**Architecture**:
```python
ProductionMLFramework
    ├── PluginManager (load/unload plugins)
    ├── EventManager (publish/subscribe events)
    ├── HookManager (before/after hooks)
    ├── ComponentRegistry (register/resolve components)
    └── Container (dependency injection)
```

**Output**:
- All outputs from Mode 1
- Plus: Plugin logs
- Plus: Event logs
- Plus: Extended metrics

**Duration**: ~7 minutes

**When to use**:
- ✅ Production deployment
- ✅ Custom plugin development
- ✅ Advanced integration scenarios
- ✅ Team collaboration (multiple developers)
- ✅ Extending functionality without modifying core

**Configuration**:
```yaml
# In config/config.yaml
plugins:
  tensorboard:
    enabled: true
    log_dir: experiments/
  
  early_stopping:
    enabled: true
    patience: 10
  
  custom_metrics:
    enabled: true
    metrics: ["snr", "correlation"]
```

**Screenshots needed**: 12-15
- Plugin initialization
- Event logs
- Extended metrics
- Component registry output

---

### Mode 4: Innovation Showcase 🚀

**Script**: `demo_innovations.py`

**What it does**: Demonstrates 5 cutting-edge ML innovations

**Command**:
```bash
python demo_innovations.py
```

**5 Innovations Demonstrated**:

#### Innovation 1: Attention Mechanism 🧠
```python
AttentionLSTMExtractor
- Multi-head attention (4 heads)
- Attention visualization
- Explainability features
```
**Output**: `attention_heatmap.png`, attention statistics

#### Innovation 2: Uncertainty Quantification 🎲
```python
BayesianLSTMExtractor
- Monte Carlo Dropout (100 samples)
- 95% confidence intervals
- Calibration plots
```
**Output**: `uncertainty_visualization.png`, `calibration_plot.png`

#### Innovation 3: Hybrid Time-Frequency Model 🌊
```python
HybridLSTMExtractor
- Time-domain LSTM path
- Frequency-domain FFT path
- Intelligent fusion
```
**Output**: `feature_importance.png`, fusion analysis

#### Innovation 4: Active Learning 🎯
```python
ActiveLearningTrainer
- Uncertainty sampling
- Diversity sampling
- 50-70% data reduction
```
**Output**: `active_learning_curve.png`, efficiency metrics

#### Innovation 5: Adversarial Robustness 🔒
```python
AdversarialTester
- FGSM attacks
- PGD attacks
- Robustness analysis
```
**Output**: `adversarial_examples.png`, `robustness_curve.png`

**Duration**: ~15 minutes (all 5 innovations)

**Output Directory**:
```
innovations_demo/
├── attention_heatmap.png
├── uncertainty_visualization.png
├── calibration_plot.png
├── feature_importance.png
├── active_learning_curve.png
├── adversarial_examples.png
└── robustness_curve.png
```

**When to use**:
- ✅ Research demonstrations
- ✅ Conference presentations
- ✅ Advanced coursework
- ✅ Showcasing expertise
- ✅ Differentiation from standard implementations

**Individual Innovation Demos**:
You can run specific innovations:
```python
from demo_innovations import InnovationDemo

demo = InnovationDemo()
demo.demo_attention_lstm()           # ~3 min
demo.demo_uncertainty_quantification() # ~3 min
demo.demo_hybrid_model()              # ~3 min
demo.demo_active_learning()           # ~4 min
demo.demo_adversarial_robustness()    # ~2 min
```

**Screenshots needed**: 25-30
- Each innovation's outputs
- Comparison plots
- Statistics and metrics
- Visual explanations

---

### Mode 5: Cost Analysis Reports 💰

**Script**: `cost_analysis_report.py`

**What it does**: Standalone comprehensive cost analysis

**Commands**:
```bash
# Analyze latest experiment
python cost_analysis_report.py

# Analyze specific experiment
python cost_analysis_report.py --experiment-dir experiments/lstm_frequency_extraction_20251118_002838

# With custom config
python cost_analysis_report.py --config config/config_production.yaml

# Provide actual training time
python cost_analysis_report.py --training-time 420
```

**Analysis Provided**:
1. **Training Costs**
   - Local training cost (energy)
   - Cloud training comparison (AWS, Azure, GCP)
   - Time cost analysis

2. **Inference Costs**
   - Per-sample cost
   - Throughput analysis
   - Scalability metrics

3. **Resource Usage**
   - Model size
   - Memory footprint
   - Parameter count

4. **Environmental Impact**
   - Carbon footprint
   - Energy consumption
   - Sustainability metrics

5. **Optimization Recommendations**
   - Priority-ranked suggestions
   - Cost reduction estimates
   - Implementation effort

**Output**:
```
experiments/lstm_frequency_extraction_*/cost_analysis/
├── cost_analysis_dashboard.png       # Comprehensive visual
├── cost_comparison_detailed.png      # Cloud vs local
├── cost_analysis.json                # Raw data
└── COST_ANALYSIS_SUMMARY.md          # Readable report
```

**Duration**: ~1 minute

**When to use**:
- ✅ After any training run
- ✅ Comparing different configurations
- ✅ Optimization planning
- ✅ Budget justification
- ✅ Environmental impact assessment

**Screenshots needed**: 5-7
- Cost dashboard
- Comparison charts
- Recommendations table
- Summary report

---

## Core Features (7 Basic Flows)

Detailed in Mode 1. These are the foundation of all execution modes.

### Quick Summary:

| Flow | Feature | Input | Output | Duration |
|------|---------|-------|--------|----------|
| 1 | Data Generation | Config | Train/test generators | ~5s |
| 2 | Dataset Creation | Generators | 40k samples | ~3s |
| 3 | Model Init | Config | LSTM (215k params) | ~1s |
| 4 | Training | Data + Model | Trained weights | ~5-7min |
| 5 | Evaluation | Trained model | Metrics | ~30s |
| 6 | Visualization | Predictions | 5 plots | ~10s |
| 7 | Cost Analysis | Training logs | Cost breakdown | ~15s |

**Reference**: See `EXECUTION_AND_SCREENSHOT_GUIDE.md` for detailed flow documentation.

---

## Advanced Features

### Feature 1: TensorBoard Integration 📊

**Built into**: All execution modes

**Activation**:
```yaml
# In config/config.yaml (already enabled by default)
experiment:
  save_dir: "./experiments"
```

**Access**:
```bash
# After training
tensorboard --logdir experiments/

# Open browser
http://localhost:6006
```

**What you see**:
- Training loss curves
- Validation loss curves
- Learning rate schedule
- Model graph
- Histograms (weights, gradients)
- Scalars (custom metrics)

**Screenshots needed**: 5-8
- TensorBoard dashboard
- Loss curves
- Model graph
- Custom metrics

---

### Feature 2: Multiple Configuration Modes ⚙️

**Available configs**:
1. `config/config.yaml` - Standard configuration (default)
2. `config/config_production.yaml` - Production settings

**Usage**:
```bash
# Standard
python main.py

# Production (with specific config)
# Note: main.py uses config/config.yaml by default
# Edit main.py to use different config, or:
cp config/config_production.yaml config/config.yaml
python main.py
```

**Key differences**:
```yaml
# config_production.yaml additional features:
- Stricter convergence criteria
- Extended logging
- Production-grade checkpointing
- Enhanced monitoring
```

---

### Feature 3: Quality Compliance System ✅

**Location**: `src/quality/`

**Features**:
1. **ISO 25010 Compliance** (`iso_compliance.py`)
   - Code quality metrics
   - Maintainability index
   - Complexity analysis

2. **Testing Suite** (`tests/`)
   - Unit tests (9 test files)
   - Integration tests
   - Performance tests
   - Quality tests

3. **Linting & Type Checking**
   - flake8 configuration
   - mypy type checking
   - pytest configuration

**Commands**:
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Check compliance
python compliance_cli.py

# Linting
flake8 src/ tests/
mypy src/
```

**Output**:
- Test results
- Coverage report (HTML)
- Compliance report
- Linting report

**When to use**:
- ✅ Before submission
- ✅ Code review
- ✅ Quality assurance
- ✅ Continuous integration

---

### Feature 4: Research Analysis Features 🔬

**Location**: `research/`

**Features**:
1. **Comparative Analysis** (`comparative_analysis.py`)
   - Compare different model architectures
   - Compare hyperparameters
   - Statistical significance testing

2. **Sensitivity Analysis** (`sensitivity_analysis.py`)
   - Parameter sensitivity
   - Robustness testing
   - Stability analysis

**Commands**:
```bash
# Run full research suite
./start_research.sh

# Or individual analyses
cd research/
python comparative_analysis.py
python sensitivity_analysis.py
```

**Output**:
```
research_results/
├── comparative_analysis_results.json
├── sensitivity_analysis_results.json
├── comparison_plots/
└── sensitivity_plots/
```

**Duration**: ~30 minutes (full suite)

**When to use**:
- ✅ Research projects
- ✅ Paper/thesis work
- ✅ Deep analysis
- ✅ Hyperparameter optimization

---

## Innovation Features

### Accessing Innovation Models

All 5 innovation models are available for use in your own scripts:

```python
from src.models.attention_lstm import AttentionLSTMExtractor
from src.models.bayesian_lstm import BayesianLSTMExtractor
from src.models.hybrid_lstm import HybridLSTMExtractor
from src.training.active_learning_trainer import ActiveLearningTrainer
from src.evaluation.adversarial_tester import AdversarialTester

# Use like standard LSTM
model = AttentionLSTMExtractor(
    input_size=5,
    hidden_size=128,
    num_layers=2,
    dropout=0.2,
    attention_heads=4
)

# Or Bayesian with uncertainty
model = BayesianLSTMExtractor(
    input_size=5,
    hidden_size=128,
    mc_samples=100
)

# Make predictions with confidence intervals
mean, std, _ = model.predict_with_uncertainty(x, n_samples=50)
```

### Innovation Feature Matrix

| Innovation | File | Unique Feature | Use Case | Added Value |
|------------|------|----------------|----------|-------------|
| **Attention** | `attention_lstm.py` | Multi-head attention | Explainability | See what matters |
| **Uncertainty** | `bayesian_lstm.py` | MC Dropout | Confidence intervals | Know reliability |
| **Hybrid** | `hybrid_lstm.py` | Time + Frequency | Multi-modal learning | Better accuracy |
| **Active Learning** | `active_learning_trainer.py` | Smart sampling | Data efficiency | 50% less data |
| **Adversarial** | `adversarial_tester.py` | Attack testing | Robustness | Security analysis |

---

## Production Features

### Plugin System

**Create custom plugin**:
```python
# plugins/my_custom_plugin.py
from src.core.plugin import Plugin

class MyCustomPlugin(Plugin):
    def initialize(self, **kwargs):
        self.name = "my_custom"
        self.version = "1.0.0"
    
    def on_training_start(self, data):
        print("Custom logic here!")
    
    def cleanup(self):
        pass

# Register in main_production.py
plugins = [
    TensorBoardPlugin(),
    MyCustomPlugin(),  # Add yours here
    # ...
]
```

### Event System

**Available events**:
```python
EventManager.TRAINING_START
EventManager.TRAINING_END
EventManager.EPOCH_START
EventManager.EPOCH_END
EventManager.DATA_LOADED
EventManager.DATA_PREPROCESSED
EventManager.MODEL_CREATED
EventManager.EVALUATION_START
EventManager.EVALUATION_END
EventManager.PLOT_CREATED
```

**Subscribe to events**:
```python
def my_callback(event):
    print(f"Event received: {event.type}")

event_manager.subscribe(EventManager.EPOCH_END, my_callback)
```

---

## Quick Reference Matrix

### Execution Decision Tree

```
What do you want to do?

├─ Standard training for assignment
│  └─ Use: python main.py (Mode 1)
│     Duration: ~7 min
│     Output: Required graphs
│
├─ Live monitoring / Demo
│  └─ Use: python main_with_dashboard.py (Mode 2)
│     Duration: ~7 min + server
│     Output: Interactive dashboard
│
├─ Production deployment / Custom plugins
│  └─ Use: python main_production.py (Mode 3)
│     Duration: ~7 min
│     Output: Extended features
│
├─ Showcase advanced ML / Research
│  └─ Use: python demo_innovations.py (Mode 4)
│     Duration: ~15 min
│     Output: 5 innovation demos
│
├─ Cost analysis / Optimization
│  └─ Use: python cost_analysis_report.py (Mode 5)
│     Duration: ~1 min
│     Output: Cost reports
│
├─ Test specific model / Quick demo
│  └─ Use: python test_data_generation.py (Demo script)
│     Duration: ~30 sec
│     Output: Data visualizations
│
└─ Research / Deep analysis
   └─ Use: ./start_research.sh (Research mode)
      Duration: ~30 min
      Output: Comparative analysis
```

---

## Screenshot Guide by Feature

### Mode 1: Basic (10 screenshots minimum)
See: `QUICK_SCREENSHOT_REFERENCE.md`

### Mode 2: Dashboard (15-20 screenshots)
1. Dashboard home page
2. Training curves (live)
3. Interactive plot with zoom
4. Per-frequency analysis tabs
5. Metrics table
6. Export functionality
7. Dashboard settings
8. Multiple experiments comparison

### Mode 3: Production (12-15 screenshots)
1. Plugin initialization logs
2. Event system activity
3. Hook execution traces
4. Component registry status
5. Extended metrics output

### Mode 4: Innovations (25-30 screenshots)
**Per innovation (5 screenshots each)**:
1. Innovation initialization
2. Training progress
3. Primary visualization
4. Statistics/metrics
5. Comparison with baseline

### Mode 5: Cost Analysis (5-7 screenshots)
1. Cost dashboard
2. Cloud comparison chart
3. Recommendations table
4. Summary report (markdown)
5. Environmental impact

---

## Comprehensive Execution Checklist

### For Assignment Submission ✅
- [ ] Run Mode 1: `python main.py`
- [ ] Capture 10 essential screenshots
- [ ] Verify Graph 1 & Graph 2 generated
- [ ] Check metrics meet requirements (MSE < 0.01, R² > 0.95)
- [ ] Review generalization (<10% gap)

### For Impressive Demo 🌟
- [ ] Run Mode 2: `python main_with_dashboard.py`
- [ ] Open dashboard in browser
- [ ] Capture interactive features
- [ ] Run Mode 4: `python demo_innovations.py`
- [ ] Show 2-3 key innovations

### For Research/Publication 📚
- [ ] Run Mode 1 (baseline)
- [ ] Run Mode 4 (all innovations)
- [ ] Run Mode 5 (cost analysis)
- [ ] Run research suite: `./start_research.sh`
- [ ] Generate comprehensive report

### For Production Deployment 🏭
- [ ] Run Mode 3: `python main_production.py`
- [ ] Configure plugins
- [ ] Set up monitoring
- [ ] Run quality checks: `pytest tests/ -v`
- [ ] Generate cost analysis

---

## Time Estimates

| Task | Duration | When |
|------|----------|------|
| **Quick demo** | 5 min | Mode 1 (reduced epochs) |
| **Standard training** | 7 min | Mode 1 (full) |
| **With dashboard** | 7 min + server | Mode 2 |
| **All innovations** | 15 min | Mode 4 (all 5) |
| **Single innovation** | 3 min | Mode 4 (one) |
| **Cost analysis** | 1 min | Mode 5 |
| **Research suite** | 30 min | Research mode |
| **Full showcase** | 25 min | All modes sequentially |

---

## Advanced Combinations

### Combination 1: Training + Dashboard + Cost
```bash
# Terminal 1: Training with dashboard
python main_with_dashboard.py

# After training completes:
# Terminal 2: Cost analysis
python cost_analysis_report.py

# Result: Complete analysis with interactive exploration
```

### Combination 2: Baseline + Innovation Comparison
```bash
# Terminal 1: Baseline
python main.py

# Terminal 2: With attention
# (modify main.py to use AttentionLSTMExtractor)
python main.py

# Terminal 3: Compare results
python research/comparative_analysis.py
```

### Combination 3: Full Research Pipeline
```bash
# 1. Standard training
python main.py

# 2. All innovations
python demo_innovations.py

# 3. Cost analysis
python cost_analysis_report.py

# 4. Research analysis
./start_research.sh

# Result: Publication-ready analysis
```

---

## Environment Variables

Optional environment variables for advanced control:

```bash
# Set device explicitly
export DEVICE=cuda  # or mps, cpu

# Reduce logging verbosity
export LOG_LEVEL=WARNING

# Custom experiment directory
export EXPERIMENT_DIR=./my_experiments

# Disable TensorBoard
export DISABLE_TENSORBOARD=1

# Run
python main.py
```

---

## Troubleshooting by Mode

### Mode 1 Issues
See: `EXECUTION_AND_SCREENSHOT_GUIDE.md` Troubleshooting section

### Mode 2 (Dashboard) Issues
```bash
# Port already in use
python main_with_dashboard.py --port 8080

# Dashboard not loading
# Check firewall, try different browser

# Dashboard crashes
python main_with_dashboard.py --no-dashboard
# Then: python main_with_dashboard.py --dashboard-only
```

### Mode 3 (Production) Issues
```bash
# Plugin errors
# Check plugin configuration in config.yaml
# Disable problematic plugin

# Event system errors
# Check event subscriber implementations
```

### Mode 4 (Innovations) Issues
```bash
# Out of memory (CUDA/MPS)
# Edit demo_innovations.py
# Reduce: hidden_size=64 (from 128)
# Reduce: mc_samples=50 (from 100)

# Slow execution
# Run individual innovations instead of all
```

### Mode 5 (Cost Analysis) Issues
```bash
# Missing experiment
python cost_analysis_report.py --experiment-dir experiments/SPECIFIC_DIR

# Missing checkpoint
# Ensure training completed successfully
ls experiments/LATEST/checkpoints/best_model.pt
```

---

## Next Steps

1. **Start with Mode 1**: Get comfortable with basic execution
2. **Try Mode 2**: Experience interactive dashboard
3. **Explore Mode 4**: See cutting-edge innovations
4. **Use Mode 5**: Optimize your setup
5. **Experiment with Mode 3**: For production scenarios

---

## Summary

Your project supports:
- ✅ **5 execution modes**
- ✅ **24+ features**
- ✅ **7 core flows**
- ✅ **5 innovations**
- ✅ **Production-grade architecture**

**Total Execution Time** (trying everything): ~60 minutes

**Recommended for first-time users**: Start with Mode 1 (7 minutes)

**Most impressive for demos**: Mode 2 (Dashboard) + Mode 4 (Innovations)

**Most valuable for deployment**: Mode 3 (Production) + Mode 5 (Cost Analysis)

---

**Document Version**: 2.0 (Complete Feature Coverage)  
**Last Updated**: November 2025  
**Related Docs**:
- `EXECUTION_AND_SCREENSHOT_GUIDE.md` - Mode 1 details
- `QUICK_SCREENSHOT_REFERENCE.md` - Quick Mode 1 reference
- `EXECUTION_FLOWS_INDEX.md` - Navigation hub
- `INNOVATIONS_SUMMARY.md` - Innovation details

**Status**: ✅ All features documented and tested

