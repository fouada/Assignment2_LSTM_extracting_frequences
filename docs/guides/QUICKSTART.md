# Quick Start Guide
## LSTM Frequency Extraction System

Get started in under 5 minutes! 🚀

---

## Installation

### Option 1: UV (Fastest - Recommended)

```bash
# Install UV (one-time setup)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run directly (UV handles dependencies automatically)
uv run main.py
```

### Option 2: Traditional (pip + venv)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run training
python main.py
```

---

## Running the System

### 1. Basic Training (Static Visualizations)

```bash
python main.py
```

**Output:**
- Training logs in terminal
- Static plots in `experiments/*/plots/`
- Model checkpoints in `experiments/*/checkpoints/`
- Results logged to `training.log`

---

### 2. Interactive Dashboard (Real-time Monitoring)

```bash
# Install dashboard dependencies (first time)
pip install dash dash-bootstrap-components plotly

# Train with dashboard
python main_with_dashboard.py
```

**Access:** http://localhost:8050

**Features:**
- Real-time training monitoring
- Interactive frequency extraction plots
- Comprehensive error analysis
- Performance metrics dashboard
- Model architecture viewer

---

### 3. Research Mode (Comparative Analysis)

```bash
# Run sensitivity analysis
python research/sensitivity_analysis.py

# Run comparative analysis  
python research/comparative_analysis.py

# Run all research experiments
./start_research.sh
```

---

## Configuration

Edit `config/config.yaml` to customize:

```yaml
# Signal parameters
data:
  frequencies: [1.0, 3.0, 5.0, 7.0]  # Hz
  sampling_rate: 1000                 # Hz
  duration: 10.0                      # seconds

# Model architecture
model:
  hidden_size: 128
  num_layers: 2
  dropout: 0.2

# Training settings
training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
```

---

## Apple Silicon (M1/M2/M3) Users

```bash
# System will automatically use MPS (Metal Performance Shaders)
python main.py  # Will detect and use MPS automatically

# Force specific device
# Edit config/config.yaml:
compute:
  device: "mps"  # or "cpu" or "cuda"
```

**Note:** Full M1 guide available in `docs/M1_MAC_COMPLETE_GUIDE.md`

---

## Expected Output

```
================================================================================
LSTM Frequency Extraction - Professional Implementation
================================================================================

STEP 1: Data Generation
  ✅ Train generator created (seed=1)
  ✅ Test generator created (seed=2)
  
STEP 2: Dataset Creation
  ✅ Training dataset: 40,000 samples
  ✅ Test dataset: 40,000 samples
  
STEP 3: Model Creation
  ✅ Parameters: 215,041
  
STEP 4: Model Training
  📈 Epoch 1/50: Train Loss: 0.0245, Val Loss: 0.0238
  ...
  ✅ Best model saved!
  
STEP 5: Model Evaluation
  ✅ Train MSE: 0.001234
  ✅ Test MSE: 0.001256
  ✅ Generalization: Good
  
STEP 6: Visualizations Created
  ✅ All plots saved

================================================================================
✅ EXPERIMENT COMPLETED SUCCESSFULLY!
================================================================================
```

---

## Viewing Results

### Static Plots

```bash
# Open plots directory
open experiments/lstm_frequency_extraction_*/plots/

# Key plots:
# - graph1_single_frequency_f2.png
# - graph2_all_frequencies.png
# - training_history.png
# - error_distribution.png
# - metrics_comparison.png
```

### Interactive Dashboard

```bash
# View latest experiment
python dashboard.py

# View specific experiment
python dashboard.py --experiment experiments/lstm_frequency_extraction_20251118_002838

# Custom port
python dashboard.py --port 8080
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_model.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

---

## Troubleshooting

### Issue: Import errors

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: GPU not detected

**Solution:**
Check `config/config.yaml`:
```yaml
compute:
  device: "auto"  # Will detect best available: cuda -> mps -> cpu
```

### Issue: Out of memory

**Solution:**
Reduce batch size in `config/config.yaml`:
```yaml
training:
  batch_size: 16  # Reduce from 32
```

### Issue: Dashboard won't start

**Solution:**
```bash
# Install dashboard dependencies
pip install dash dash-bootstrap-components plotly kaleido

# Use different port if 8050 is busy
python dashboard.py --port 8080
```

---

## Next Steps

1. ✅ **Read the Architecture:** [ARCHITECTURE.md](ARCHITECTURE.md)
2. ✅ **Explore Usage:** [USAGE_GUIDE.md](USAGE_GUIDE.md)
3. ✅ **Try Dashboard:** [DASHBOARD.md](DASHBOARD.md)
4. ✅ **Run Tests:** [TESTING.md](TESTING.md)
5. ✅ **Research Mode:** [RESEARCH_QUICKSTART.md](RESEARCH_QUICKSTART.md)

---

## Quick Commands Reference

```bash
# Basic training
python main.py

# With dashboard
python main_with_dashboard.py

# View dashboard
python dashboard.py

# Run tests
pytest tests/ -v

# Research mode
./start_research.sh

# Clean experiments
rm -rf experiments/lstm_frequency_extraction_*
```

---

## Support

- **Assignment Details:** [Assignment_English_Translation.md](Assignment_English_Translation.md)
- **Complete Documentation:** [README.md](../README.md)
- **Issue?** Check [USAGE_GUIDE.md](USAGE_GUIDE.md) troubleshooting section

---

**Ready to start!** Run: `python main.py` or `python main_with_dashboard.py`

