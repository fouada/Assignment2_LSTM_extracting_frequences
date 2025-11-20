# 🚀 Complete M1 Pro Mac Guide - LSTM Frequency Extraction

**Optimized for Apple Silicon M1 Pro**

---

## 📋 Table of Contents

- [Quick Setup](#quick-setup)
- [Essential Commands](#essential-commands)
- [Running with Tensorboard](#running-with-tensorboard)
- [All Available Capabilities](#all-available-capabilities)
- [Advanced Features](#advanced-features)
- [Monitoring & Debugging](#monitoring--debugging)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)

---

## ⚡ Quick Setup

### Step 1: Check if UV is installed

```bash
uv --version
```

### Step 2: Install UV (if needed)

```bash
# Method 1: Homebrew (Recommended for M1 Mac)
brew install uv

# Method 2: Official installer
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH (if using Method 2)
export PATH="$HOME/.cargo/bin:$PATH"
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.zshrc
```

### Step 3: Verify Installation

```bash
uv --version
python3 --version
```

---

## 🎯 Essential Commands

### 1. **Run the Main Training Pipeline**

```bash
# Navigate to project directory
cd /Users/fouadaz/LearningFromUniversity/Learning/LLMSAndMultiAgentOrchestration/course-materials/assignments/Assignment2_LSTM_extracting_frequences

# Run with UV (handles everything automatically!)
uv run main.py
```

**What happens:**
- ✅ Auto-detects M1 MPS (Metal Performance Shaders) for GPU acceleration
- ✅ Creates virtual environment
- ✅ Installs all dependencies
- ✅ Trains LSTM model
- ✅ Generates visualizations
- ✅ Saves checkpoints and logs

### 2. **Run with Production Configuration**

```bash
uv run main_production.py
```

---

## 📊 Running with Tensorboard

### **Option 1: Run Tensorboard in Separate Terminal**

```bash
# Terminal 1: Start Training
uv run main.py

# Terminal 2: Start Tensorboard (while training)
uv run tensorboard --logdir experiments/
```

Then open: **http://localhost:6006**

### **Option 2: Run Tensorboard After Training**

```bash
# After training completes, view specific experiment
uv run tensorboard --logdir experiments/lstm_frequency_extraction_20251115_231209/checkpoints/tensorboard/

# Or view all experiments
uv run tensorboard --logdir experiments/
```

### **Option 3: Run Both Simultaneously (Background)**

```bash
# Start tensorboard in background
uv run tensorboard --logdir experiments/ &

# Run training
uv run main.py

# When done, kill tensorboard
pkill -f tensorboard
```

### **Option 4: Different Port (if 6006 is busy)**

```bash
uv run tensorboard --logdir experiments/ --port 6007
```

Open: **http://localhost:6007**

---

## 🔧 All Available Capabilities

### **A. Training & Execution**

```bash
# 1. Standard Training
uv run main.py

# 2. Production Training
uv run main_production.py

# 3. Run with specific Python version
uv run --python 3.11 main.py

# 4. Run with custom config (edit config/config.yaml first)
uv run main.py

# 5. Quick test run (modify epochs in config to 5)
uv run python -c "
import yaml
config = yaml.safe_load(open('config/config.yaml'))
config['training']['epochs'] = 5
config['data']['duration'] = 2.0
yaml.dump(config, open('config/config_test.yaml', 'w'))
"
```

### **B. Testing**

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_model.py -v
uv run pytest tests/test_data.py -v

# Run with coverage report
uv run pytest tests/ --cov=src --cov-report=html

# Run with detailed output
uv run pytest tests/ -vv --tb=long

# View coverage report
open htmlcov/index.html
```

### **C. Code Quality & Formatting**

```bash
# Format code with Black
uv run black src/ tests/ main.py

# Check code style
uv run flake8 src/ tests/ main.py

# Type checking
uv run mypy src/

# Sort imports
uv run isort src/ tests/ main.py

# Run all quality checks
uv run black src/ tests/ main.py && \
uv run isort src/ tests/ main.py && \
uv run flake8 src/ tests/ main.py && \
uv run mypy src/
```

### **D. Jupyter Notebooks**

```bash
# Start Jupyter Lab
uv run jupyter lab

# Start Jupyter Notebook
uv run jupyter notebook

# Run specific notebook
uv run jupyter notebook notebooks/experiment.ipynb
```

### **E. Tensorboard (All Options)**

```bash
# Basic tensorboard
uv run tensorboard --logdir experiments/

# Specific experiment
uv run tensorboard --logdir experiments/lstm_frequency_extraction_20251115_231209/checkpoints/tensorboard/

# Multiple experiment comparison
uv run tensorboard --logdir_spec=exp1:experiments/lstm_frequency_extraction_20251115_220133/checkpoints/tensorboard/,exp2:experiments/lstm_frequency_extraction_20251115_231209/checkpoints/tensorboard/

# Different port
uv run tensorboard --logdir experiments/ --port 8888

# Bind to all interfaces (access from other devices)
uv run tensorboard --logdir experiments/ --host 0.0.0.0

# Reload interval (seconds)
uv run tensorboard --logdir experiments/ --reload_interval 30

# With custom window title
uv run tensorboard --logdir experiments/ --window_title "LSTM Frequency Extraction"
```

### **F. Python Interactive Mode**

```bash
# Interactive Python with project imports
uv run python

# Then in Python:
>>> from src.data.signal_generator import create_train_test_generators
>>> from src.models.lstm_extractor import create_model
>>> import torch
>>> 
>>> # Check M1 GPU availability
>>> print(f"MPS Available: {torch.backends.mps.is_available()}")
>>> print(f"MPS Built: {torch.backends.mps.is_built()}")
>>> 
>>> # Create model
>>> config = {'input_size': 5, 'hidden_size': 128, 'num_layers': 2, 'output_size': 1, 'dropout': 0.2}
>>> model = create_model(config)
>>> print(model)
```

### **G. Package Management**

```bash
# Sync dependencies from pyproject.toml
uv sync

# Sync with all extras (dev, notebook)
uv sync --all-extras

# Sync only dev dependencies
uv sync --extra dev

# Add new package
uv add wandb

# Add dev dependency
uv add --dev black

# Remove package
uv remove wandb

# Update all packages
uv sync --upgrade

# Lock dependencies
uv lock

# Install from lock file (exact versions)
uv sync --frozen

# Clean cache
uv cache clean
```

### **H. Environment Management**

```bash
# Create new virtual environment
uv venv

# Create with specific Python version
uv venv --python 3.11

# Activate environment (if needed manually)
source .venv/bin/activate

# Deactivate
deactivate

# Remove and recreate environment
rm -rf .venv
uv venv
uv sync
```

---

## 🎨 Advanced Features

### **1. Custom Training with Modified Config**

Create a custom config file:

```bash
# Copy existing config
cp config/config.yaml config/my_config.yaml

# Edit it
nano config/my_config.yaml
```

Modify parameters in `my_config.yaml`:
```yaml
model:
  hidden_size: 256  # Increase capacity
  num_layers: 3     # Deeper network

training:
  batch_size: 64    # Larger batches for M1
  epochs: 100       # More training
  learning_rate: 0.0005
```

Then run with custom config (modify main.py line 125 to load your config).

### **2. View Results**

```bash
# View latest plots
open experiments/$(ls -t experiments/ | head -1)/plots/*.png

# View specific plots
open experiments/lstm_frequency_extraction_20251115_231209/plots/graph1_single_frequency_f2.png
open experiments/lstm_frequency_extraction_20251115_231209/plots/graph2_all_frequencies.png

# View training history
open experiments/lstm_frequency_extraction_20251115_231209/plots/training_history.png

# View metrics comparison
open experiments/lstm_frequency_extraction_20251115_231209/plots/metrics_comparison.png
```

### **3. Load and Evaluate Saved Model**

```bash
uv run python -c "
import torch
from src.models.lstm_extractor import create_model

# Load model
config = {'input_size': 5, 'hidden_size': 128, 'num_layers': 2, 'output_size': 1, 'dropout': 0.2}
model = create_model(config)

# Load checkpoint
checkpoint = torch.load('experiments/lstm_frequency_extraction_20251115_231209/checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

print('Model loaded successfully!')
print(f'Epoch: {checkpoint[\"epoch\"]}')
print(f'Loss: {checkpoint[\"loss\"]:.6f}')
"
```

### **4. Batch Processing Multiple Experiments**

```bash
# Run multiple experiments with different configs
for hidden_size in 64 128 256; do
  echo "Training with hidden_size=$hidden_size"
  # Modify config and run
  uv run python main.py
done
```

### **5. Export Environment**

```bash
# Export exact dependencies
uv pip freeze > requirements_frozen.txt

# Export conda-compatible environment
uv pip list --format=freeze > environment.txt
```

---

## 🔍 Monitoring & Debugging

### **1. Real-time Training Monitoring**

```bash
# Terminal 1: Training with verbose output
uv run main.py

# Terminal 2: Watch training log
tail -f training.log

# Terminal 3: Watch GPU usage (M1 specific)
sudo powermetrics --samplers gpu_power -i 1000
```

### **2. Check M1 GPU Utilization**

```bash
uv run python -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'MPS Available: {torch.backends.mps.is_available()}')
print(f'MPS Built: {torch.backends.mps.is_built()}')

if torch.backends.mps.is_available():
    device = torch.device('mps')
    x = torch.randn(1000, 1000, device=device)
    y = x @ x
    print('✅ MPS GPU is working!')
else:
    print('⚠️  MPS not available, using CPU')
"
```

### **3. Memory Profiling**

```bash
# Check memory usage during training
uv run python -c "
import torch
import psutil

process = psutil.Process()
print(f'Memory Usage: {process.memory_info().rss / 1024 / 1024:.2f} MB')

# Check tensor memory on MPS
if torch.backends.mps.is_available():
    device = torch.device('mps')
    x = torch.randn(10000, 10000, device=device)
    print(f'Created large tensor on MPS')
"
```

### **4. Log Analysis**

```bash
# View training progress
grep "Epoch" training.log

# View loss values
grep "Loss" training.log

# View errors
grep "ERROR" training.log

# View warnings
grep "WARNING" training.log

# Count total epochs trained
grep "Epoch" training.log | wc -l
```

### **5. Experiment Comparison**

```bash
# List all experiments
ls -lh experiments/

# Compare sizes
du -sh experiments/*/

# View all configs
cat experiments/*/config.yaml

# Compare final metrics
grep "Test MSE" experiments/*/training.log
```

---

## ⚡ Performance Optimization (M1 Specific)

### **1. Enable M1 GPU Acceleration**

The project automatically detects and uses M1 MPS. Verify:

```bash
uv run python -c "
import torch
print(f'Using device: {\"mps\" if torch.backends.mps.is_available() else \"cpu\"}')
"
```

### **2. Optimize Batch Size for M1**

Edit `config/config.yaml`:

```yaml
training:
  batch_size: 64  # M1 Pro can handle larger batches
```

### **3. Use Multiple Workers**

```yaml
compute:
  num_workers: 8  # M1 Pro has 8+ cores
  pin_memory: false  # Not needed for MPS
```

### **4. Compile Model (PyTorch 2.0+)**

Add to training script:

```python
import torch
model = torch.compile(model)  # JIT compile for M1
```

### **5. Monitor Temperature & Performance**

```bash
# Install powermetrics (if not available)
# Watch CPU/GPU temperature
sudo powermetrics --samplers cpu_power,gpu_power -i 1000

# Install osx-cpu-temp for simple monitoring
brew install osx-cpu-temp
osx-cpu-temp
```

---

## 🐛 Troubleshooting

### **Problem 1: UV not found**

```bash
# Add to PATH
export PATH="$HOME/.cargo/bin:$PATH"
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Verify
uv --version
```

### **Problem 2: MPS not available**

```bash
# Check PyTorch version
uv run python -c "import torch; print(torch.__version__)"

# Should be >= 2.0.0 for M1 support
# If not, upgrade:
uv sync --upgrade
```

### **Problem 3: Port already in use (Tensorboard)**

```bash
# Kill existing tensorboard
pkill -f tensorboard

# Or use different port
uv run tensorboard --logdir experiments/ --port 6007
```

### **Problem 4: Permission denied**

```bash
# Fix permissions
chmod +x main.py
chmod +x main_production.py
```

### **Problem 5: Module not found**

```bash
# Reinstall dependencies
uv sync --all-extras

# Or clean install
rm -rf .venv uv.lock
uv sync
```

### **Problem 6: Out of memory**

Edit `config/config.yaml`:

```yaml
training:
  batch_size: 16  # Reduce if OOM
```

### **Problem 7: Slow training**

```bash
# Verify MPS is being used
uv run python -c "
import torch
print(f'MPS: {torch.backends.mps.is_available()}')
"

# Check if using CPU instead of GPU
grep "Using device" training.log
```

---

## 🎯 Quick Reference Card

### **Most Common Commands**

```bash
# 1. Train model
uv run main.py

# 2. View tensorboard
uv run tensorboard --logdir experiments/ &
open http://localhost:6006

# 3. Run tests
uv run pytest tests/ -v

# 4. View results
open experiments/$(ls -t experiments/ | head -1)/plots/*.png

# 5. Check GPU
uv run python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"

# 6. Update dependencies
uv sync --upgrade

# 7. Format code
uv run black src/ tests/ main.py

# 8. Clean restart
rm -rf .venv && uv sync && uv run main.py
```

---

## 📊 Expected Output Example

When you run `uv run main.py`, you should see:

```
================================================================================
LSTM Frequency Extraction - Professional Implementation
================================================================================

STEP 1: Data Generation
--------------------------------------------------------------------------------
2025-11-17 15:30:00 - INFO - Train generator created (seed=1)
2025-11-17 15:30:00 - INFO - Test generator created (seed=2)

STEP 2: Dataset Creation
--------------------------------------------------------------------------------
2025-11-17 15:30:01 - INFO - Training dataset: 40,000 samples
2025-11-17 15:30:01 - INFO - Test dataset: 40,000 samples

STEP 3: Model Creation
--------------------------------------------------------------------------------
2025-11-17 15:30:01 - INFO - Using device: mps
2025-11-17 15:30:02 - INFO - Model parameters: 215,041

STEP 4: Model Training
--------------------------------------------------------------------------------
Epoch 1/50: 100%|████████████| 1250/1250 [00:15<00:00, 83.33it/s]
2025-11-17 15:30:18 - INFO - Epoch 1/50 - Train Loss: 0.0245, Val Loss: 0.0238
...
Best model saved at epoch 25!

STEP 5: Model Evaluation
--------------------------------------------------------------------------------
Train MSE: 0.001234, Test MSE: 0.001256
Train R²: 0.9956, Test R²: 0.9954
✅ SUCCESS: Model generalizes well!

STEP 6: Creating Visualizations
--------------------------------------------------------------------------------
✅ graph1_single_frequency_f2.png
✅ graph2_all_frequencies.png
✅ training_history.png
✅ metrics_comparison.png
✅ error_distribution.png

================================================================================
EXPERIMENT COMPLETED SUCCESSFULLY!
================================================================================

Results saved to: experiments/lstm_frequency_extraction_20251117_153000/
- Plots: experiments/lstm_frequency_extraction_20251117_153000/plots/
- Checkpoints: experiments/lstm_frequency_extraction_20251117_153000/checkpoints/
- Tensorboard logs: experiments/lstm_frequency_extraction_20251117_153000/checkpoints/tensorboard/
```

---

## 🚀 Next Steps

1. **Run your first training:**
   ```bash
   uv run main.py
   ```

2. **Monitor with Tensorboard:**
   ```bash
   uv run tensorboard --logdir experiments/ &
   open http://localhost:6006
   ```

3. **View results:**
   ```bash
   open experiments/$(ls -t experiments/ | head -1)/plots/*.png
   ```

4. **Experiment with different configurations:**
   - Edit `config/config.yaml`
   - Try different `hidden_size`: 64, 128, 256, 512
   - Try different `num_layers`: 1, 2, 3, 4
   - Try different `learning_rate`: 0.0001, 0.001, 0.01

5. **Compare experiments in Tensorboard:**
   ```bash
   uv run tensorboard --logdir experiments/
   ```

---

## 🎓 Pro Tips for M1 Mac

1. **M1 GPU Acceleration**: The project auto-detects MPS (Metal Performance Shaders)
2. **Larger Batches**: M1 Pro can handle larger batch sizes (try 64-128)
3. **Parallel Workers**: Use 8+ workers for data loading on M1 Pro
4. **Low Power Mode**: Disable low power mode for faster training
5. **Temperature**: Monitor temperature to avoid thermal throttling
6. **Memory**: M1 Pro has unified memory - adjust batch size if needed

---

## 📚 Documentation Files

- `README.md` - Project overview
- `UV_QUICKSTART.md` - UV basics
- `USAGE_GUIDE.md` - Detailed usage
- `EXECUTION_GUIDE.md` - Step-by-step execution
- `ARCHITECTURE.md` - System architecture
- `Quick_Reference_Guide.md` - Quick commands
- `M1_MAC_COMPLETE_GUIDE.md` - This file!

---

**🎉 You're all set! Happy training on your M1 Pro Mac!**

**Quick Start:** Just run `uv run main.py` and you're good to go! 🚀

