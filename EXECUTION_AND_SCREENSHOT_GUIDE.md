# Execution and Screenshot Guide
## LSTM Frequency Extraction System

**Authors**: Fouad Azem (040830861) & Tal Goldengorn (207042573)  
**Course**: LLM And Multi Agent Orchestration
**Instructor**: Dr. Yoram Segal  
**Date**: November 2025

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [System Flows](#system-flows)
3. [Execution Methods](#execution-methods)
4. [Screenshot Guide](#screenshot-guide)
5. [Flow-by-Flow Execution](#flow-by-flow-execution)
6. [Advanced Execution Options](#advanced-execution-options)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

---

## Overview

This guide documents how to execute all supported flows in the LSTM Frequency Extraction system and how to capture screenshots of live execution for documentation, presentations, and assignment submissions.

### Supported Flows

The system supports 7 major execution flows:

| Flow # | Flow Name | Purpose | Duration |
|--------|-----------|---------|----------|
| 1 | Data Generation | Create train/test datasets | ~5 seconds |
| 2 | Dataset Creation | Prepare dataloaders | ~3 seconds |
| 3 | Model Creation | Initialize LSTM model | ~1 second |
| 4 | Model Training | Train the LSTM | ~5-7 minutes |
| 5 | Model Evaluation | Compute metrics | ~30 seconds |
| 6 | Visualization | Generate plots | ~10 seconds |
| 7 | Cost Analysis | Analyze computational costs | ~15 seconds |

**Total Runtime**: ~7-10 minutes (depending on hardware)

---

## System Flows

### Flow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Configuration Loading                     │
│                      (config.yaml)                           │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┴───────────────────────┐
        │                                       │
        ▼                                       ▼
┌──────────────────┐                   ┌──────────────────┐
│   Flow 1 & 2:    │                   │   Flow 3:        │
│   Data Pipeline  │──────────────────▶│   Model Init     │
└──────────────────┘                   └──────────────────┘
                                               │
                                               ▼
                                       ┌──────────────────┐
                                       │   Flow 4:        │
                                       │   Training       │
                                       └──────────────────┘
                                               │
                        ┌──────────────────────┴───────────────┐
                        │                                      │
                        ▼                                      ▼
                ┌──────────────────┐                  ┌──────────────────┐
                │   Flow 5:        │                  │   Flow 6:        │
                │   Evaluation     │──────────────────│   Visualization  │
                └──────────────────┘                  └──────────────────┘
                        │
                        ▼
                ┌──────────────────┐
                │   Flow 7:        │
                │   Cost Analysis  │
                └──────────────────┘
```

### Flow Details

#### Flow 1: Data Generation
**Purpose**: Generate noisy mixed signals and pure target frequencies  
**Key Operations**:
- Create SignalGenerator instances (train_seed=1, test_seed=2)
- Generate 10,000 time samples for 4 frequencies
- Add random amplitude and phase noise
- Create pure sine wave targets

**What to Screenshot**:
- Configuration parameters being loaded
- Data generation log messages
- Dataset size confirmation

#### Flow 2: Dataset Creation
**Purpose**: Prepare PyTorch datasets and dataloaders  
**Key Operations**:
- Create FrequencyExtractionDataset instances
- Setup StatefulDataLoader with sequential sampling
- Normalize signals
- Create one-hot frequency encodings

**What to Screenshot**:
- Dataset creation logs
- Total samples confirmation (40,000)
- Dataloader batch information

#### Flow 3: Model Creation
**Purpose**: Initialize the StatefulLSTMExtractor model  
**Key Operations**:
- Initialize LSTM layers (2 layers, 128 hidden units)
- Setup Layer Normalization
- Configure dropout (0.2)
- Initialize hidden and cell states

**What to Screenshot**:
- Model architecture summary
- Parameter count
- Device allocation (CPU/GPU/MPS)

#### Flow 4: Model Training
**Purpose**: Train the LSTM to extract frequencies  
**Key Operations**:
- Iterate through epochs (up to 50)
- Manage LSTM states (reset at frequency boundaries)
- Compute loss and backpropagate
- Update weights with Adam optimizer
- Apply gradient clipping
- Track validation loss
- Save best model checkpoint

**What to Screenshot**:
- Training progress (epoch logs)
- Loss curves in real-time (if using TensorBoard)
- Best model checkpoint save confirmation
- Training completion time

#### Flow 5: Model Evaluation
**Purpose**: Evaluate model performance on train and test sets  
**Key Operations**:
- Load best model checkpoint
- Compute metrics on training set (seed=1)
- Compute metrics on test set (seed=2)
- Calculate generalization gap
- Per-frequency analysis

**What to Screenshot**:
- Train set metrics (MSE, R², MAE, etc.)
- Test set metrics
- Generalization analysis results
- Per-frequency performance breakdown

#### Flow 6: Visualization
**Purpose**: Generate publication-quality plots  
**Key Operations**:
- Extract predictions per frequency
- Create Graph 1: Single frequency (f2=3Hz)
- Create Graph 2: All frequencies (2×2 grid)
- Training history plots
- Error distribution analysis
- Metrics comparison (train vs test)

**What to Screenshot**:
- Plot generation confirmation
- Sample of generated plots (inline if possible)
- File paths where plots are saved

#### Flow 7: Cost Analysis (Optional)
**Purpose**: Analyze computational costs and provide optimization recommendations  
**Key Operations**:
- Calculate training costs
- Benchmark inference speed
- Estimate cloud provider costs
- Generate optimization recommendations
- Create cost dashboards

**What to Screenshot**:
- Cost breakdown summary
- Cloud cost comparison
- Optimization recommendations
- Cost dashboard plots

---

## Execution Methods

### Method 1: Full Pipeline (Recommended)

**Command**:
```bash
cd Assignment2_LSTM_extracting_frequences
python main.py
```

**When to Use**: 
- First run
- Complete training from scratch
- Generating all outputs
- Assignment submission

**Output**:
- All 7 flows execute sequentially
- Results in `experiments/lstm_frequency_extraction_<timestamp>/`

**Screenshot Opportunities**: All flows (comprehensive documentation)

---

### Method 2: UV Package Manager (Faster)

**Command**:
```bash
cd Assignment2_LSTM_extracting_frequences
uv run main.py
```

**When to Use**:
- Have UV installed
- Faster dependency management
- Automatic virtual environment

**Advantages**:
- No manual venv setup
- Faster package resolution
- Cleaner execution

---

### Method 3: With Virtual Environment

**Commands**:
```bash
cd Assignment2_LSTM_extracting_frequences

# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate  # macOS/Linux
# OR
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run
python main.py
```

**When to Use**:
- Clean isolated environment
- First-time setup
- Avoiding conflicts with system packages

---

### Method 4: With Configuration Override

**Command**:
```bash
# Edit config.yaml first, then run
python main.py

# Or create custom config
python main.py --config custom_config.yaml  # (requires implementation)
```

**When to Use**:
- Testing different hyperparameters
- Shorter training for demos
- Different frequencies

**Common Modifications**:
```yaml
# Quick demo (in config/config.yaml)
training:
  epochs: 10  # Instead of 50
  batch_size: 64  # Faster training

# Disable cost analysis for speed
cost_analysis:
  enabled: false
```

---

### Method 5: Interactive Jupyter Notebook (Optional)

**Setup**:
```bash
# Install Jupyter
pip install jupyter notebook

# Create notebook
jupyter notebook
```

**Notebook Content**:
```python
# Cell 1: Imports
from main import *
import yaml

# Cell 2: Load config
config = load_config('config/config.yaml')

# Cell 3: Run specific flow
# ... (see Flow-by-Flow Execution section)
```

**When to Use**:
- Step-by-step exploration
- Debugging
- Educational presentations

---

## Screenshot Guide

### Equipment Setup

**Recommended Tools**:

| Platform | Screenshot Tool | Shortcut | Notes |
|----------|----------------|----------|-------|
| macOS | Built-in | `Cmd + Shift + 4` | Select region |
| macOS | Built-in | `Cmd + Shift + 5` | Screen recording |
| Windows | Snipping Tool | `Win + Shift + S` | Modern Windows |
| Linux | Flameshot | `PrtScn` | Install via package manager |
| All | Terminal recording | `asciinema record` | Record terminal sessions |

**Screen Recording Tools**:
- **QuickTime** (macOS): For video tutorials
- **OBS Studio** (All): Professional recording
- **asciinema**: Terminal-specific recording

---

### What to Screenshot

#### 1. Pre-Execution Screenshots

**Before running any code**:

✅ **Screenshot 1**: Project structure
```bash
cd Assignment2_LSTM_extracting_frequences
tree -L 2  # or ls -R for 2 levels

# Screenshot the output showing folder structure
```

✅ **Screenshot 2**: Configuration file
```bash
cat config/config.yaml

# Screenshot the full configuration
```

✅ **Screenshot 3**: Requirements
```bash
cat requirements.txt

# Screenshot dependencies list
```

---

#### 2. Execution Start Screenshots

✅ **Screenshot 4**: Command execution
```bash
# Screenshot your terminal showing:
# - Current directory
# - Command being executed
# - First few lines of output

python main.py
```

Example of what to capture:
```
fouad@macbook Assignment2_LSTM_extracting_frequences % python main.py
================================================================================
LSTM Frequency Extraction - Professional Implementation
================================================================================
2025-11-18 10:30:45 - __main__ - INFO - Configuration loaded successfully
2025-11-18 10:30:45 - __main__ - INFO - Random seed set to 42
2025-11-18 10:30:45 - __main__ - INFO - Using device: mps
```

---

#### 3. Flow-Specific Screenshots

**Flow 1-2: Data Generation & Dataset Creation**

✅ **Screenshot 5**: Data generation logs
```
================================================================================
STEP 1: Data Generation
================================================================================
2025-11-18 10:30:46 - signal_generator - INFO - Creating training generator (seed=1)
2025-11-18 10:30:46 - signal_generator - INFO - Creating test generator (seed=2)
2025-11-18 10:30:46 - signal_generator - INFO - Generated 10000 time samples
2025-11-18 10:30:46 - signal_generator - INFO - Frequencies: [1.0, 3.0, 5.0, 7.0] Hz
```

✅ **Screenshot 6**: Dataset creation logs
```
================================================================================
STEP 2: Dataset Creation
================================================================================
2025-11-18 10:30:47 - dataset - INFO - Created training dataset: 40000 samples
2025-11-18 10:30:47 - dataset - INFO - Created test dataset: 40000 samples
2025-11-18 10:30:47 - dataset - INFO - Batch size: 32
```

**Flow 3: Model Creation**

✅ **Screenshot 7**: Model architecture
```
================================================================================
STEP 3: Model Creation
================================================================================
2025-11-18 10:30:48 - lstm_extractor - INFO - Model architecture:
StatefulLSTMExtractor(
  (input_norm): LayerNorm((5,), eps=1e-05, elementwise_affine=True)
  (lstm): LSTM(5, 128, num_layers=2, batch_first=True, dropout=0.2)
  (output_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
  (fc1): Linear(in_features=128, out_features=64, bias=True)
  (relu): ReLU()
  (dropout): Dropout(p=0.2, inplace=False)
  (fc2): Linear(in_features=64, out_features=1, bias=True)
)
Total parameters: 215,041
```

**Flow 4: Training**

✅ **Screenshot 8**: Training progress (early epochs)
```
================================================================================
STEP 4: Model Training
================================================================================
Epoch [1/50] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:12
Train Loss: 0.1234 | Val Loss: 0.1156 | LR: 0.001000
```

✅ **Screenshot 9**: Training progress (mid-training)
```
Epoch [25/50] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:08
Train Loss: 0.0045 | Val Loss: 0.0048 | LR: 0.000250
```

✅ **Screenshot 10**: Training completion
```
Epoch [43/50] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:07
Train Loss: 0.0012 | Val Loss: 0.0013 | LR: 0.000125

Early stopping triggered! No improvement for 10 epochs.
Best model saved at: experiments/.../checkpoints/best_model.pt
Training completed in 387.52 seconds (6.46 minutes)
```

**Flow 5: Evaluation**

✅ **Screenshot 11**: Train set metrics
```
================================================================================
STEP 5: Model Evaluation
================================================================================

Evaluating on TRAIN set...
--------------------------------------------------------------------------------
TRAIN SET METRICS:
  mse: 0.001234
  rmse: 0.035128
  mae: 0.026734
  r2_score: 0.991245
  correlation: 0.995612
  snr_db: 41.234567
```

✅ **Screenshot 12**: Test set metrics and per-frequency analysis
```
Evaluating on TEST set...
--------------------------------------------------------------------------------
TEST SET METRICS:
  mse: 0.001331
  rmse: 0.036485
  mae: 0.027812
  r2_score: 0.990512
  correlation: 0.995234
  snr_db: 40.123456

--------------------------------------------------------------------------------
GENERALIZATION ANALYSIS
--------------------------------------------------------------------------------
Generalization gap: 8.13%
Status: ✅ GOOD - Model generalizes well!

PER-FREQUENCY METRICS (TEST SET):

  Frequency 1 (1.0 Hz):
    mse: 0.001123
    r2_score: 0.991234

  Frequency 2 (3.0 Hz):
    mse: 0.001289
    r2_score: 0.990567
    
  Frequency 3 (5.0 Hz):
    mse: 0.001401
    r2_score: 0.989123
    
  Frequency 4 (7.0 Hz):
    mse: 0.001512
    r2_score: 0.988456
```

**Flow 6: Visualization**

✅ **Screenshot 13**: Plot generation confirmation
```
================================================================================
STEP 6: Creating Visualizations
================================================================================
2025-11-18 10:37:35 - plotter - INFO - Creating graph 1: Single frequency (f2=3Hz)
2025-11-18 10:37:36 - plotter - INFO - Saved: experiments/.../plots/graph1_single_frequency_f2.png
2025-11-18 10:37:37 - plotter - INFO - Creating graph 2: All frequencies
2025-11-18 10:37:39 - plotter - INFO - Saved: experiments/.../plots/graph2_all_frequencies.png
2025-11-18 10:37:40 - plotter - INFO - Creating training history plot
2025-11-18 10:37:41 - plotter - INFO - All visualizations created successfully!
```

**Flow 7: Cost Analysis**

✅ **Screenshot 14**: Cost analysis summary
```
================================================================================
STEP 7: Cost Analysis & Optimization Recommendations
================================================================================

COST BREAKDOWN:
  Training cost: $0.0234
  Inference cost (per 1000 samples): $0.0001
  Total model parameters: 215,041
  Model size: 3.2 MB
  
CLOUD PROVIDER COMPARISON:
  AWS (p3.2xlarge): $3.06/hour → $0.51 for this training
  Azure (NC6): $0.90/hour → $0.15 for this training
  GCP (n1-standard-8 + T4): $0.65/hour → $0.11 for this training
  
OPTIMIZATION RECOMMENDATIONS:
  [HIGH] Reduce epochs to 30 (save ~40% training time)
  [MEDIUM] Increase batch size to 64 (save ~15% time)
  [LOW] Use mixed precision training (save ~25% memory)
```

---

#### 4. Generated Outputs Screenshots

✅ **Screenshot 15**: Experiment directory structure
```bash
ls -lR experiments/lstm_frequency_extraction_20251118_103045/

# Screenshot showing:
# - plots/ directory with all PNG files
# - checkpoints/ directory with model files
# - config.yaml copy
```

✅ **Screenshot 16-17**: Generated Plots (capture the actual images)
- **Graph 1**: `graph1_single_frequency_f2.png` (REQUIRED)
- **Graph 2**: `graph2_all_frequencies.png` (REQUIRED)

✅ **Screenshot 18-20**: Additional plots
- Training history
- Error distribution
- Metrics comparison

✅ **Screenshot 21**: Cost analysis dashboard (if enabled)
- Cost breakdown visualization
- Cloud comparison chart

---

#### 5. Post-Execution Screenshots

✅ **Screenshot 22**: Final summary
```
================================================================================
EXPERIMENT COMPLETED SUCCESSFULLY!
================================================================================

Results saved to: experiments/lstm_frequency_extraction_20251118_103045
- Plots: experiments/.../plots
- Checkpoints: experiments/.../checkpoints
- Cost Analysis: experiments/.../cost_analysis
- Tensorboard logs: experiments/.../checkpoints/tensorboard

================================================================================
FINAL METRICS SUMMARY
================================================================================
Train MSE: 0.001234
Test MSE:  0.001331
Train R²:  0.9912
Test R²:   0.9905

Generalization Status: ✅ GOOD
✅ SUCCESS: Model generalizes well to unseen noise!

================================================================================
💡 TIP: Run 'python cost_analysis_report.py' for detailed cost insights!
================================================================================
```

---

### Screenshot Organization

**Recommended folder structure for screenshots**:

```
screenshots/
├── 01_pre_execution/
│   ├── 01_project_structure.png
│   ├── 02_configuration.png
│   └── 03_requirements.png
│
├── 02_execution_start/
│   └── 04_command_execution.png
│
├── 03_data_pipeline/
│   ├── 05_data_generation.png
│   └── 06_dataset_creation.png
│
├── 04_model_initialization/
│   └── 07_model_architecture.png
│
├── 05_training_progress/
│   ├── 08_early_epochs.png
│   ├── 09_mid_training.png
│   └── 10_training_completion.png
│
├── 06_evaluation/
│   ├── 11_train_metrics.png
│   └── 12_test_metrics_and_per_frequency.png
│
├── 07_visualization/
│   └── 13_plot_generation.png
│
├── 08_cost_analysis/
│   └── 14_cost_summary.png
│
├── 09_generated_outputs/
│   ├── 15_directory_structure.png
│   ├── 16_graph1_single_frequency.png  ⭐ REQUIRED
│   ├── 17_graph2_all_frequencies.png   ⭐ REQUIRED
│   ├── 18_training_history.png
│   ├── 19_error_distribution.png
│   ├── 20_metrics_comparison.png
│   └── 21_cost_dashboard.png
│
└── 10_final_summary/
    └── 22_execution_summary.png
```

---

## Flow-by-Flow Execution

### Running Individual Flows (For Debugging/Demonstration)

While `main.py` runs all flows sequentially, you can execute individual flows for:
- Debugging specific components
- Quick demonstrations
- Educational purposes
- Testing configuration changes

**Note**: Most flows depend on previous flows, so ensure prerequisites are met.

---

### Flow 1 & 2: Data Generation (Standalone)

**Script**: `test_data_generation.py` (create this for standalone testing)

```python
"""
Standalone Data Generation Flow
Demonstrates signal generation without full training
"""
import yaml
import numpy as np
import matplotlib.pyplot as plt
from src.data.signal_generator import create_train_test_generators

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Generate data
train_gen, test_gen = create_train_test_generators(
    frequencies=config['data']['frequencies'],
    sampling_rate=config['data']['sampling_rate'],
    duration=config['data']['duration'],
    amplitude_range=config['data']['amplitude_range'],
    phase_range=config['data']['phase_range'],
    train_seed=config['data']['train_seed'],
    test_seed=config['data']['test_seed']
)

# Print info
print(f"✅ Train generator created (seed={config['data']['train_seed']})")
print(f"✅ Test generator created (seed={config['data']['test_seed']})")
print(f"✅ Mixed signal shape: {train_gen.mixed_signal.shape}")
print(f"✅ Frequencies: {train_gen.frequencies}")

# Quick visualization
fig, axes = plt.subplots(2, 1, figsize=(12, 6))

# Plot mixed signal
axes[0].plot(train_gen.time[:1000], train_gen.mixed_signal[:1000])
axes[0].set_title("Mixed Signal (first 1000 samples)")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Amplitude")
axes[0].grid(True)

# Plot pure frequency
axes[1].plot(train_gen.time[:1000], train_gen.pure_targets[1, :1000])
axes[1].set_title("Pure Frequency f2=3Hz (first 1000 samples)")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Amplitude")
axes[1].grid(True)

plt.tight_layout()
plt.savefig("data_generation_demo.png", dpi=150)
print("✅ Saved: data_generation_demo.png")
```

**Run**:
```bash
python test_data_generation.py
```

**Screenshot**:
- Terminal output showing data shapes
- Generated `data_generation_demo.png`

---

### Flow 3: Model Initialization (Standalone)

**Script**: `test_model_creation.py`

```python
"""
Standalone Model Creation Flow
Demonstrates model initialization and architecture
"""
import yaml
import torch
from src.models.lstm_extractor import create_model

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create model
model = create_model(config['model'])

# Print architecture
print("="*80)
print("MODEL ARCHITECTURE")
print("="*80)
print(model)
print("\n" + "="*80)
print("MODEL SUMMARY")
print("="*80)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")

# Test forward pass
print("\n" + "="*80)
print("TESTING FORWARD PASS")
print("="*80)

# Create dummy input: (batch=8, seq_len=1, input_size=5)
dummy_input = torch.randn(8, 1, 5)
print(f"Input shape: {dummy_input.shape}")

# Forward pass
output = model(dummy_input, reset_state=True)
print(f"Output shape: {output.shape}")
print("✅ Forward pass successful!")

# Test state management
print("\n" + "="*80)
print("TESTING STATE MANAGEMENT")
print("="*80)

model.reset_state()
print("✅ State reset successful")

# Process sequence
for t in range(5):
    out = model(dummy_input, reset_state=False)
    print(f"  Time step {t+1}: Output shape = {out.shape}")

model.detach_state()
print("✅ State detachment successful")

print("\n" + "="*80)
print("MODEL CREATION TEST COMPLETE")
print("="*80)
```

**Run**:
```bash
python test_model_creation.py
```

**Screenshot**:
- Complete model architecture
- Parameter count
- Forward pass test results

---

### Flow 4: Training (Requires Flows 1-3)

Training is integrated into `main.py`, but you can modify it for quick testing:

**Quick Training Config**:
```yaml
# config/config_quick.yaml
training:
  epochs: 5  # Reduced from 50
  batch_size: 64
  early_stopping_patience: 3
```

**Run**:
```bash
# Copy config
cp config/config.yaml config/config_backup.yaml

# Edit config.yaml to reduce epochs
# Then run
python main.py

# Restore
mv config/config_backup.yaml config/config.yaml
```

**Screenshot**:
- Quick training progress (5 epochs)
- Loss curves

---

### Flow 5 & 6: Evaluation and Visualization (Requires trained model)

**Script**: `test_evaluation.py`

```python
"""
Standalone Evaluation Flow
Load existing model and evaluate
"""
import yaml
import torch
from pathlib import Path
from src.models.lstm_extractor import create_model
from src.data.signal_generator import create_train_test_generators
from src.data.dataset import create_dataloaders
from src.evaluation.metrics import evaluate_model, compare_train_test_performance

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Setup device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model
model = create_model(config['model']).to(device)

# Find latest experiment
exp_dir = Path('experiments')
latest_exp = max(exp_dir.glob('lstm_frequency_extraction_*'))
checkpoint_path = latest_exp / 'checkpoints' / 'best_model.pt'

print(f"Loading model from: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
print("✅ Model loaded successfully")

# Generate data
train_gen, test_gen = create_train_test_generators(
    frequencies=config['data']['frequencies'],
    sampling_rate=config['data']['sampling_rate'],
    duration=config['data']['duration'],
    amplitude_range=config['data']['amplitude_range'],
    phase_range=config['data']['phase_range'],
    train_seed=config['data']['train_seed'],
    test_seed=config['data']['test_seed']
)

# Create dataloaders
train_loader, test_loader = create_dataloaders(
    train_generator=train_gen,
    test_generator=test_gen,
    batch_size=config['training']['batch_size'],
    normalize=True,
    device='cpu'
)

# Evaluate
print("\n" + "="*80)
print("EVALUATING MODEL")
print("="*80)

print("\nTrain set:")
train_results = evaluate_model(model, train_loader, device, compute_per_frequency=True)
for metric, value in train_results['overall'].items():
    print(f"  {metric}: {value:.6f}")

print("\nTest set:")
test_results = evaluate_model(model, test_loader, device, compute_per_frequency=True)
for metric, value in test_results['overall'].items():
    print(f"  {metric}: {value:.6f}")

print("\nGeneralization:")
comparison = compare_train_test_performance(train_results, test_results)
print(f"  Gap: {comparison['overall_generalization']['percentage_difference']:.2f}%")
print(f"  Status: {comparison['overall_generalization']['status']}")
```

**Run**:
```bash
python test_evaluation.py
```

**Screenshot**:
- Evaluation metrics from pre-trained model

---

### Flow 7: Cost Analysis (Standalone)

**Script**: `test_cost_analysis.py`

```python
"""
Standalone Cost Analysis Flow
Analyze computational costs without training
"""
import yaml
import torch
import time
from pathlib import Path
from src.models.lstm_extractor import create_model
from src.evaluation.cost_analysis import create_cost_analyzer

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Setup
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = create_model(config['model']).to(device)

# Create analyzer
analyzer = create_cost_analyzer(model, device)

# Simulate training time (or load from logs)
training_time_seconds = 420.0  # 7 minutes

# Create sample input
sample_input = torch.randn(1, 1, config['model']['input_size']).to(device)

# Analyze
print("="*80)
print("COST ANALYSIS")
print("="*80)

breakdown = analyzer.analyze_costs(
    training_time_seconds=training_time_seconds,
    sample_input=sample_input,
    final_mse=0.00133
)

# Generate recommendations
recommendations = analyzer.generate_recommendations(
    breakdown=breakdown,
    current_config=config
)

# Print
analyzer.print_recommendations(recommendations)

# Export
output_dir = Path('cost_analysis_standalone')
output_dir.mkdir(exist_ok=True)

analyzer.export_analysis(
    breakdown=breakdown,
    recommendations=recommendations,
    save_path=output_dir / 'cost_analysis.json'
)

print(f"\n✅ Analysis saved to: {output_dir}")
```

**Run**:
```bash
python test_cost_analysis.py
```

**Screenshot**:
- Cost breakdown
- Recommendations

---

## Advanced Execution Options

### Option 1: TensorBoard Monitoring

**During Training**:

```bash
# Terminal 1: Start training
python main.py

# Terminal 2: Start TensorBoard (in parallel)
tensorboard --logdir experiments/
```

**Open**: `http://localhost:6006`

**Screenshot**:
- TensorBoard dashboard
- Live training curves
- Model graph

---

### Option 2: Jupyter Notebook Execution

**Create**: `demo.ipynb`

```python
# Cell 1: Setup
import sys
sys.path.append('.')

from main import *
import yaml

# Cell 2: Config
config = load_config('config/config.yaml')
device = setup_device(config)

# Cell 3: Data
train_gen, test_gen = create_train_test_generators(
    frequencies=config['data']['frequencies'],
    sampling_rate=config['data']['sampling_rate'],
    duration=config['data']['duration'],
    amplitude_range=config['data']['amplitude_range'],
    phase_range=config['data']['phase_range'],
    train_seed=config['data']['train_seed'],
    test_seed=config['data']['test_seed']
)

# Cell 4: Visualize data
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(train_gen.time[:1000], train_gen.mixed_signal[:1000])
ax.set_title("Mixed Signal")
plt.show()

# Cell 5: Model
model = create_model(config['model']).to(device)
print(model)

# Cell 6: Training
# ... continue with other flows
```

**Screenshot**:
- Jupyter notebook interface
- Inline plots
- Interactive outputs

---

### Option 3: Automated Screenshot Script

**Create**: `auto_screenshot.py`

```python
"""
Automated Screenshot Capture During Execution
Captures key moments automatically
"""
import subprocess
import time
import os
from datetime import datetime

def capture_screenshot(name):
    """Capture screenshot on macOS"""
    timestamp = datetime.now().strftime("%H%M%S")
    filename = f"screenshots/{timestamp}_{name}.png"
    os.makedirs("screenshots", exist_ok=True)
    
    # macOS screenshot command
    subprocess.run(['screencapture', '-x', filename])
    print(f"📸 Screenshot saved: {filename}")

def run_with_screenshots():
    """Run main.py and capture screenshots at key points"""
    
    # Start execution
    print("Starting execution with automated screenshots...")
    
    # Capture initial state
    capture_screenshot("00_start")
    
    # Run main.py in subprocess
    process = subprocess.Popen(
        ['python', 'main.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Monitor output and capture screenshots
    screenshots_taken = {
        'data_gen': False,
        'model_init': False,
        'training_start': False,
        'training_mid': False,
        'evaluation': False,
        'visualization': False
    }
    
    for line in process.stdout:
        print(line, end='')
        
        # Trigger screenshots based on log messages
        if 'STEP 1: Data Generation' in line and not screenshots_taken['data_gen']:
            time.sleep(2)
            capture_screenshot("01_data_generation")
            screenshots_taken['data_gen'] = True
            
        elif 'STEP 3: Model Creation' in line and not screenshots_taken['model_init']:
            time.sleep(2)
            capture_screenshot("02_model_init")
            screenshots_taken['model_init'] = True
            
        elif 'STEP 4: Model Training' in line and not screenshots_taken['training_start']:
            time.sleep(2)
            capture_screenshot("03_training_start")
            screenshots_taken['training_start'] = True
            
        elif 'Epoch [25/' in line and not screenshots_taken['training_mid']:
            time.sleep(2)
            capture_screenshot("04_training_mid")
            screenshots_taken['training_mid'] = True
            
        elif 'STEP 5: Model Evaluation' in line and not screenshots_taken['evaluation']:
            time.sleep(2)
            capture_screenshot("05_evaluation")
            screenshots_taken['evaluation'] = True
            
        elif 'STEP 6: Creating Visualizations' in line and not screenshots_taken['visualization']:
            time.sleep(2)
            capture_screenshot("06_visualization")
            screenshots_taken['visualization'] = True
    
    # Wait for completion
    process.wait()
    
    # Final screenshot
    capture_screenshot("07_completion")
    
    print("\n✅ Execution complete with automated screenshots!")

if __name__ == '__main__':
    run_with_screenshots()
```

**Run**:
```bash
python auto_screenshot.py
```

---

### Option 4: Screen Recording

**For macOS**:
```bash
# Start recording
screencapture -v execution_recording.mov

# Then run
python main.py

# Stop recording: Ctrl+C in screencapture terminal
```

**For All Platforms (asciinema)**:
```bash
# Install
pip install asciinema

# Record terminal session
asciinema rec execution.cast

# Run
python main.py

# Stop: Ctrl+D

# Play back
asciinema play execution.cast

# Upload
asciinema upload execution.cast
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Module Not Found

**Error**:
```
ModuleNotFoundError: No module named 'src'
```

**Solution**:
```bash
# Ensure you're in the correct directory
cd Assignment2_LSTM_extracting_frequences

# Verify structure
ls src/

# Install dependencies
pip install -r requirements.txt
```

---

#### Issue 2: CUDA/MPS Not Available

**Error**:
```
RuntimeError: MPS backend is not available
```

**Solution**:
```yaml
# Edit config/config.yaml
compute:
  device: "cpu"  # Change from "auto"
```

---

#### Issue 3: Out of Memory

**Error**:
```
RuntimeError: CUDA out of memory
```

**Solution**:
```yaml
# Reduce batch size in config/config.yaml
training:
  batch_size: 16  # Reduce from 32
```

---

#### Issue 4: Plots Not Generating

**Error**:
```
FileNotFoundError: plots directory not found
```

**Solution**:
```bash
# Manually create directories
mkdir -p experiments/test/plots
mkdir -p experiments/test/checkpoints

# Or ensure main.py runs completely
python main.py 2>&1 | tee execution.log
```

---

#### Issue 5: Screenshot Tools Not Working

**macOS**:
```bash
# Verify screencapture
which screencapture

# Test
screencapture -x test.png
```

**Linux**:
```bash
# Install flameshot
sudo apt install flameshot  # Ubuntu/Debian
sudo dnf install flameshot  # Fedora

# Or use GNOME screenshot
gnome-screenshot -f screenshot.png
```

---

## Best Practices

### For Documentation

1. **Organized Screenshots**
   - Use sequential numbering
   - Include timestamps
   - Add descriptive filenames
   - Organize in folders by flow

2. **Clear Visibility**
   - Use high resolution (150-300 DPI)
   - Ensure text is readable
   - Highlight important sections
   - Use annotations if needed

3. **Comprehensive Coverage**
   - Capture all 7 flows
   - Include error states (if any)
   - Show before/after states
   - Document configuration changes

---

### For Presentations

1. **Key Screenshots Only**
   - Focus on most important outputs
   - Highlight exceptional results
   - Show required graphs (Graph 1 & 2)
   - Include final metrics

2. **Visual Quality**
   - Clean terminal (clear screen first)
   - Consistent theme
   - Readable fonts
   - Professional appearance

3. **Storytelling**
   - Show progression
   - Highlight achievements
   - Demonstrate understanding
   - Connect to theory

---

### For Assignment Submission

1. **Required Screenshots**
   - ✅ Project structure
   - ✅ Configuration file
   - ✅ Execution command
   - ✅ Training progress
   - ✅ Final metrics (train & test)
   - ✅ Generalization analysis
   - ✅ **Graph 1** (single frequency)
   - ✅ **Graph 2** (all frequencies)
   - ✅ Experiment directory

2. **Optional but Valuable**
   - Model architecture
   - TensorBoard screenshots
   - Cost analysis
   - Additional plots
   - Testing results

3. **Documentation**
   - Add captions to each screenshot
   - Explain what each shows
   - Reference in main documentation
   - Create index/table of contents

---

## Quick Reference Card

### Essential Commands

```bash
# Full execution
python main.py

# With virtual environment
source venv/bin/activate && python main.py

# With UV
uv run main.py

# View TensorBoard
tensorboard --logdir experiments/

# Run tests
pytest tests/ -v

# Quick training (edit config first)
python main.py  # with epochs: 5
```

### Screenshot Checklist

- [ ] Pre-execution (3 screenshots)
- [ ] Execution start (1 screenshot)
- [ ] Data pipeline (2 screenshots)
- [ ] Model creation (1 screenshot)
- [ ] Training progress (3 screenshots)
- [ ] Evaluation (2 screenshots)
- [ ] Visualization (1 screenshot)
- [ ] Cost analysis (1 screenshot)
- [ ] Generated outputs (7 screenshots)
- [ ] Final summary (1 screenshot)

**Total**: ~22 screenshots for complete documentation

### Critical Outputs

1. **graph1_single_frequency_f2.png** ⭐ REQUIRED
2. **graph2_all_frequencies.png** ⭐ REQUIRED
3. Final metrics (MSE train vs test)
4. Generalization status

---

## Conclusion

This guide provides comprehensive instructions for:
- ✅ Executing all 7 system flows
- ✅ Capturing screenshots at key points
- ✅ Organizing documentation
- ✅ Troubleshooting common issues
- ✅ Following best practices

**For Assignment Submission**:
1. Run full pipeline: `python main.py`
2. Capture 22 screenshots using this guide
3. Organize in `screenshots/` directory
4. Reference in main documentation

**For Quick Demo**:
1. Use reduced epochs config
2. Focus on key screenshots (execution, graphs, metrics)
3. Use screen recording for live demo

---

## Appendix: Screenshot Templates

### Template 1: Terminal Screenshot

```
┌─────────────────────────────────────────────────────────┐
│ Terminal - python main.py                               │
├─────────────────────────────────────────────────────────┤
│                                                          │
│ $ cd Assignment2_LSTM_extracting_frequences            │
│ $ python main.py                                        │
│                                                          │
│ ====================================================    │
│ LSTM Frequency Extraction - Professional Implementation │
│ ====================================================    │
│                                                          │
│ [Capture output here]                                   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Template 2: Plot Screenshot

```
┌─────────────────────────────────────────────────────────┐
│ Graph 1: Single Frequency Extraction (f2=3Hz)          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│ [Capture full plot image]                              │
│                                                          │
│ Caption: LSTM prediction (red) vs target (blue) for    │
│          3Hz frequency. MSE=0.00133, R²=0.991           │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

**Document Version**: 1.0  
**Last Updated**: November 2025  
**Authors**: Fouad Azem & Tal Goldengorn  
**Status**: ✅ Complete and Ready for Use

