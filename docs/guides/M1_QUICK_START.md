# 🚀 M1 Pro Mac - Quick Start Guide

**Your system is ready! Everything is configured and working!** ✅

---

## ⚡ Instant Commands (Copy & Paste)

### **Navigate to Project**
```bash
cd /Users/fouadaz/LearningFromUniversity/Learning/LLMSAndMultiAgentOrchestration/course-materials/assignments/Assignment2_LSTM_extracting_frequences
```

---

## 🎯 Most Common Actions

### **1. Run Training (M1 GPU Accelerated)** 🏃
```bash
uv run main.py
```
- Uses M1 GPU automatically
- ~15-30 minutes for 50 epochs
- Saves plots and checkpoints

### **2. View TensorBoard** 📊
```bash
uv run tensorboard --logdir experiments/
```
**Then open:** http://localhost:6006

### **3. Use Interactive Menu** 🎮
```bash
./quick_commands.sh
```
This gives you a menu with all options!

### **4. View Latest Results** 👀
```bash
open experiments/lstm_frequency_extraction_20251115_231209/plots/*.png
```

### **5. Run Tests** 🧪
```bash
uv run pytest tests/ -v
```

---

## 📊 What You Already Have

### ✅ **Completed Experiments:** 5
- `lstm_frequency_extraction_20251115_220133`
- `lstm_frequency_extraction_20251115_220852`
- `lstm_frequency_extraction_20251115_221136`
- `lstm_frequency_extraction_20251115_221832`
- `lstm_frequency_extraction_20251115_231209` (latest)

### ✅ **Latest Experiment Has:**
1. ✅ **graph1_single_frequency_f2.png** - Assignment requirement #1
2. ✅ **graph2_all_frequencies.png** - Assignment requirement #2
3. ✅ **training_history.png** - Loss curves
4. ✅ **metrics_comparison.png** - Train vs Test
5. ✅ **error_distribution.png** - Error analysis
6. ✅ **Checkpoints** - Trained model weights
7. ✅ **TensorBoard logs** - Training metrics

---

## 🎮 Interactive Menu Script

I've created a convenient script for you!

```bash
./quick_commands.sh
```

**Menu Options:**
1. 🏃 Run Training (uses M1 GPU)
2. 📊 Launch TensorBoard (all experiments)
3. 📈 Launch TensorBoard (latest experiment only)
4. 👀 View Latest Results (plots)
5. 🧪 Run Tests
6. 🔍 Check M1 GPU Status
7. 📦 Update Dependencies
8. 🧹 Clean & Fresh Install
9. 📋 List All Experiments
10. 💻 Interactive Python Shell

---

## 🔥 Advanced Commands

### **Run with Different Configurations**

```bash
# Edit config first
nano config/config.yaml

# Then run
uv run main.py
```

### **Compare Multiple Experiments in TensorBoard**

```bash
uv run tensorboard --logdir experiments/
```
TensorBoard will automatically compare all experiments!

### **Run Tests with Coverage**

```bash
uv run pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

### **Interactive Python with Project**

```bash
uv run python
```

Then try:
```python
from src.data.signal_generator import create_train_test_generators
from src.models.lstm_extractor import create_model
import torch

# Check M1 GPU
print(f"MPS Available: {torch.backends.mps.is_available()}")

# Create generators
train_gen, test_gen = create_train_test_generators(
    frequencies=[1.0, 3.0, 5.0, 7.0],
    sampling_rate=1000,
    duration=10.0,
    train_seed=1,
    test_seed=2
)

# Generate data
mixed_signal, targets = train_gen.generate_complete_dataset()
print(f"Mixed signal shape: {mixed_signal.shape}")
print(f"Targets shape: {targets.shape}")
```

### **Code Quality Check**

```bash
# Format code
uv run black src/ tests/ main.py

# Check style
uv run flake8 src/ tests/ main.py

# Type check
uv run mypy src/
```

---

## 🎯 Quick Experiments

### **Fast Training (5 epochs)**

Edit `config/config.yaml`:
```yaml
training:
  epochs: 5  # Change from 50 to 5
```

Then run:
```bash
uv run main.py
```

### **Larger Model**

Edit `config/config.yaml`:
```yaml
model:
  hidden_size: 256  # Change from 128
  num_layers: 3     # Change from 2
```

Then run:
```bash
uv run main.py
```

### **Larger Batch Size (M1 Pro can handle it!)**

Edit `config/config.yaml`:
```yaml
training:
  batch_size: 64  # Change from 32
```

Then run:
```bash
uv run main.py
```

---

## 🔍 Monitoring

### **Watch Training Log in Real-time**

```bash
# Terminal 1: Run training
uv run main.py

# Terminal 2: Watch log
tail -f training.log
```

### **Check M1 GPU Usage**

```bash
# Simple check
uv run python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"

# Detailed check
sudo powermetrics --samplers gpu_power -i 1000
```

### **View Experiment History**

```bash
# List all experiments
ls -lh experiments/

# View configs
cat experiments/*/config.yaml

# Compare final metrics
grep "Test MSE" experiments/*/training.log 2>/dev/null || echo "No logs found"
```

---

## 📚 Documentation Files

All guides are in the project root:

- **M1_QUICK_START.md** ← You are here!
- **M1_MAC_COMPLETE_GUIDE.md** - Full detailed guide
- **UV_QUICKSTART.md** - UV basics
- **README.md** - Project overview
- **USAGE_GUIDE.md** - Detailed usage
- **EXECUTION_GUIDE.md** - Step-by-step
- **Quick_Reference_Guide.md** - Quick reference

---

## 🎓 Your System Status

### ✅ **Installed & Working:**
- ✅ UV 0.9.9
- ✅ PyTorch 2.9.1
- ✅ M1 GPU (MPS) Available
- ✅ All Dependencies
- ✅ 5 Completed Experiments

### 🚀 **Ready to:**
- ✅ Train new models
- ✅ Run experiments
- ✅ View TensorBoard
- ✅ Run tests
- ✅ Use M1 GPU acceleration

---

## 💡 Pro Tips for M1 Pro

1. **GPU Acceleration:** Your M1 GPU is automatically detected! Look for "Using device: mps" in logs
2. **Larger Batches:** M1 Pro can handle batch_size of 64-128
3. **Parallel Loading:** Set `num_workers: 8` in config for faster data loading
4. **Background TensorBoard:** Run `uv run tensorboard --logdir experiments/ &` to keep it running
5. **Quick Testing:** Set `epochs: 5` for quick experiments
6. **Multiple Terminals:** Use one for training, one for TensorBoard, one for monitoring

---

## 🎬 Recommended Next Steps

### **Option 1: View Your Existing Results** (1 minute)
```bash
open experiments/lstm_frequency_extraction_20251115_231209/plots/*.png
```

### **Option 2: Launch TensorBoard** (2 minutes)
```bash
uv run tensorboard --logdir experiments/
# Open: http://localhost:6006
```

### **Option 3: Run New Training** (15-30 minutes)
```bash
uv run main.py
```

### **Option 4: Interactive Menu** (easiest!)
```bash
./quick_commands.sh
```

---

## 🆘 Troubleshooting

### **Problem: Command not found**
```bash
# Make sure you're in the right directory
cd /Users/fouadaz/LearningFromUniversity/Learning/LLMSAndMultiAgentOrchestration/course-materials/assignments/Assignment2_LSTM_extracting_frequences
```

### **Problem: UV not found**
```bash
# Reinstall UV
brew install uv
```

### **Problem: Port already in use**
```bash
# Kill existing tensorboard
pkill -f tensorboard

# Or use different port
uv run tensorboard --logdir experiments/ --port 8888
```

### **Problem: Out of memory**
Edit `config/config.yaml`:
```yaml
training:
  batch_size: 16  # Reduce from 32
```

---

## 🎉 You're All Set!

Everything is configured and ready to go!

**Simplest Start:**
```bash
cd /Users/fouadaz/LearningFromUniversity/Learning/LLMSAndMultiAgentOrchestration/course-materials/assignments/Assignment2_LSTM_extracting_frequences
./quick_commands.sh
```

**Or just run:**
```bash
cd /Users/fouadaz/LearningFromUniversity/Learning/LLMSAndMultiAgentOrchestration/course-materials/assignments/Assignment2_LSTM_extracting_frequences
uv run main.py
```

---

**Questions? Check:**
- `M1_MAC_COMPLETE_GUIDE.md` for full details
- `UV_QUICKSTART.md` for UV commands
- `README.md` for project overview

**Happy Training! 🚀**

