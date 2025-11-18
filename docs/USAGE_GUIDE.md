# Usage Guide
## Quick Start to Advanced Usage

---

## 🚀 Quick Start (3 Steps)

### Step 1: Install Dependencies

```bash
cd Assignment2_LSTM_extracting_frequences
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Run the Complete Pipeline

```bash
python main.py
```

### Step 3: View Results

Results are saved in `experiments/lstm_frequency_extraction_TIMESTAMP/`:
- **Plots**: `plots/` directory
- **Model**: `checkpoints/best_model.pt`
- **Logs**: `checkpoints/tensorboard/`

---

## 📊 View Training Progress

### Tensorboard

```bash
# In a new terminal (keep venv activated)
tensorboard --logdir experiments/
```

Open browser: http://localhost:6006

You'll see:
- Training/validation loss curves
- Learning rate schedule
- Real-time metric updates

---

## ⚙️ Configuration

### Edit `config/config.yaml` Before Running

#### Change Model Size:

```yaml
model:
  hidden_size: 256      # Default: 128 (more = more capacity)
  num_layers: 3         # Default: 2 (more = deeper network)
  dropout: 0.3          # Default: 0.2 (higher = more regularization)
```

#### Adjust Training:

```yaml
training:
  batch_size: 64        # Default: 32 (higher = faster, more memory)
  epochs: 100           # Default: 50 (more = longer training)
  learning_rate: 0.0005 # Default: 0.001 (lower = more stable)
```

#### Change Frequencies:

```yaml
data:
  frequencies: [2.0, 4.0, 6.0, 8.0]  # Must be 4 frequencies
  sampling_rate: 2000                # Higher = more samples
  duration: 5.0                      # Shorter = less data
```

---

## 🔬 Advanced Usage

### 1. Use as a Library

```python
from src.data import create_train_test_generators, create_dataloaders
from src.models import create_model
from src.evaluation import evaluate_model

# Generate data
train_gen, test_gen = create_train_test_generators(
    frequencies=[1.0, 3.0, 5.0, 7.0],
    sampling_rate=1000,
    duration=10.0
)

# Create loaders
train_loader, test_loader = create_dataloaders(
    train_gen, test_gen, batch_size=32
)

# Create model
model = create_model({
    'input_size': 5,
    'hidden_size': 128,
    'num_layers': 2
})

# Train (your custom loop)
# ...

# Evaluate
results = evaluate_model(model, test_loader, device)
print(f"MSE: {results['overall']['mse']:.6f}")
```

### 2. Custom Training Loop

```python
from src.training import LSTMTrainer

trainer = LSTMTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=test_loader,
    config={
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 0.001,
        'optimizer': 'adam'
    },
    device=torch.device('cuda'),
    experiment_dir=Path('my_experiment')
)

history = trainer.train()
```

### 3. Generate Custom Visualizations

```python
from src.visualization import FrequencyExtractionVisualizer

visualizer = FrequencyExtractionVisualizer(save_dir='my_plots')

visualizer.plot_single_frequency_comparison(
    time=time_vector,
    mixed_signal=mixed,
    target=target_f2,
    prediction=pred_f2,
    frequency=3.0,
    freq_idx=1,
    save_name='my_custom_plot'
)
```

---

## 🧪 Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test Module

```bash
pytest tests/test_data.py -v
```

### With Coverage Report

```bash
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html  # View coverage
```

### Run Single Test

```bash
pytest tests/test_model.py::TestStatefulLSTMExtractor::test_forward_single_sample -v
```

---

## 📈 Interpreting Results

### Training Output

```
Epoch 1/50 - Train Loss: 0.0245, Val Loss: 0.0238, LR: 1.00e-03
```

**What to look for:**
- ✅ Loss decreasing over epochs
- ✅ Val loss ≈ Train loss (good generalization)
- ⚠️ Val loss >> Train loss (overfitting)
- ⚠️ Both losses stuck high (underfitting)

### Final Metrics

```
Train MSE: 0.001234
Test MSE:  0.001256
Generalization: ✅ Good
```

**Success Criteria:**
- MSE < 0.01 (both train and test)
- Test MSE ≈ Train MSE (within 10%)
- R² > 0.95

### Visualizations

**Graph 1** (Single Frequency):
- Red dots should follow blue line closely
- Gray background is the noisy input
- MSE annotation shows fit quality

**Graph 2** (All Frequencies):
- Check all 4 subplots
- Each should show good fit
- Compare MSE across frequencies

---

## 🔧 Troubleshooting

### Issue: CUDA Out of Memory

**Solution:**
```yaml
# config/config.yaml
training:
  batch_size: 16  # Reduce from 32
```

Or force CPU:
```yaml
compute:
  device: "cpu"
```

### Issue: Loss Not Decreasing

**Possible causes:**
1. Learning rate too high
   ```yaml
   learning_rate: 0.0001  # Reduce
   ```

2. Model too small
   ```yaml
   hidden_size: 256  # Increase
   num_layers: 3
   ```

3. State management issue
   - Check logs for "State reset" messages
   - Should reset at start of each frequency

### Issue: Overfitting (Test Loss >> Train Loss)

**Solutions:**
```yaml
model:
  dropout: 0.4  # Increase regularization

training:
  weight_decay: 1e-4  # Add weight decay
  early_stopping_patience: 5  # Stop earlier
```

### Issue: Imports Not Working

```bash
# Make sure you're in the project root
cd Assignment2_LSTM_extracting_frequences

# Activate venv
source venv/bin/activate

# Install in development mode
pip install -e .
```

---

## 💻 Development Workflow

### 1. Make Changes to Code

```bash
# Edit files in src/
vim src/models/lstm_extractor.py
```

### 2. Format Code

```bash
black src/ tests/ main.py
```

### 3. Check Style

```bash
flake8 src/ tests/ main.py
```

### 4. Run Tests

```bash
pytest tests/ -v
```

### 5. Test Your Changes

```bash
python main.py
```

---

## 📊 Experiment Tracking

### Structure

```
experiments/
└── lstm_frequency_extraction_20251115_143022/
    ├── config.yaml                  # Config used
    ├── plots/
    │   ├── graph1_single_frequency_f2.png
    │   ├── graph2_all_frequencies.png
    │   ├── training_history.png
    │   ├── error_distribution.png
    │   └── metrics_comparison.png
    └── checkpoints/
        ├── best_model.pt            # Best model
        ├── checkpoint_epoch_9.pt
        ├── checkpoint_epoch_19.pt
        └── tensorboard/
            └── events.out.tfevents.*
```

### Load a Saved Model

```python
import torch
from src.models import StatefulLSTMExtractor

# Load checkpoint
checkpoint = torch.load('experiments/.../checkpoints/best_model.pt')

# Recreate model
model = StatefulLSTMExtractor(**checkpoint['config']['model'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use for inference
with torch.no_grad():
    output = model(input_tensor)
```

---

## 🎓 Learning Exercises

### Exercise 1: Change Frequencies

Modify `config.yaml`:
```yaml
frequencies: [0.5, 2.0, 4.5, 8.0]
```

**Question**: How does this affect:
- Training time?
- Model performance?
- Visualization patterns?

### Exercise 2: Implement L=10

Modify dataset to use sequences:
```python
# In src/data/dataset.py
def __getitem__(self, idx):
    # Return sequences of L=10 instead of single samples
    return self.get_sequence_batch(idx, sequence_length=10)
```

**Compare**: L=1 vs L=10 performance

### Exercise 3: Add More Frequencies

```yaml
frequencies: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # 6 instead of 4
```

**Modify**:
- `input_size: 7` (S[t] + 6 one-hot values)
- Dataset structure
- Visualization (2×3 grid)

### Exercise 4: Bidirectional LSTM

```yaml
model:
  bidirectional: true
```

**Analyze**:
- Does it improve performance?
- What are the trade-offs?
- Why might it not be suitable for real-time?

---

## 📚 Common Patterns

### Pattern 1: Batch Inference

```python
model.eval()
all_predictions = []

with torch.no_grad():
    model.reset_state()  # Start fresh
    
    for t in range(num_samples):
        input_t = create_input(t)
        pred = model(input_t, reset_state=False)
        all_predictions.append(pred.item())
        # State carries forward automatically
```

### Pattern 2: Multi-Frequency Prediction

```python
predictions_per_freq = {}

for freq_idx in range(4):
    model.reset_state()  # New frequency sequence
    predictions = []
    
    for t in range(num_samples):
        input_t = create_input(t, freq_idx)
        pred = model(input_t, reset_state=False)
        predictions.append(pred.item())
    
    predictions_per_freq[freq_idx] = np.array(predictions)
```

### Pattern 3: Custom Metric

```python
from src.evaluation import FrequencyExtractionMetrics

metrics = FrequencyExtractionMetrics()

# Accumulate predictions
for batch in dataloader:
    preds = model(batch['input'])
    metrics.update(preds, batch['target'], batch['freq_idx'])

# Compute all metrics
results = metrics.compute_metrics()
print(f"R²: {results['r2_score']:.4f}")
```

---

## 🎯 Assignment Submission Checklist

- [ ] Code runs without errors: `python main.py`
- [ ] Tests pass: `pytest tests/ -v`
- [ ] Training converges (loss decreases)
- [ ] Test MSE < 0.01
- [ ] Generalization is good (test ≈ train)
- [ ] Graph 1 generated and looks correct
- [ ] Graph 2 generated with all 4 frequencies
- [ ] Results saved in `experiments/`
- [ ] Code is well-documented
- [ ] README.md explains the project

### Submission Files

```
submission/
├── src/                    # All source code
├── config/config.yaml      # Your configuration
├── experiments/best_run/   # Your best results
│   └── plots/             # Required visualizations
├── main.py                # Entry point
├── requirements.txt       # Dependencies
├── README.md              # Documentation
└── report.pdf             # Your analysis (if required)
```

---

## 💡 Tips & Best Practices

### 1. Start Small, Scale Up

```yaml
# For quick testing
data:
  duration: 1.0        # Instead of 10.0
training:
  epochs: 5            # Instead of 50
```

### 2. Monitor Tensorboard

- Always run tensorboard during training
- Watch for divergence or plateaus
- Helps debug training issues early

### 3. Save Intermediate Results

```python
# In main.py, save after each step
torch.save(train_dataset, 'train_dataset.pt')
torch.save(model, 'model_checkpoint.pt')
```

### 4. Use Logging

```python
import logging
logger = logging.getLogger(__name__)

logger.info("Starting training...")
logger.debug(f"Batch shape: {batch.shape}")
```

### 5. Version Your Experiments

```bash
# Use descriptive experiment names
experiments/
├── baseline_128hidden/
├── deep_256hidden_3layers/
└── bidirectional_dropout0.3/
```

---

## 🆘 Getting Help

### Check Logs

```bash
cat training.log | grep ERROR
```

### Debug Mode

```python
# Add to main.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Common Commands

```bash
# Check GPU usage
nvidia-smi

# Monitor memory
htop

# Find process using port (tensorboard)
lsof -i :6006
```

---

## 🎉 Success!

When everything works:
- ✅ Training completes
- ✅ Metrics look good
- ✅ Plots are generated
- ✅ Generalization is confirmed

**You have successfully implemented a professional MIT-level LSTM frequency extraction system!** 🚀

---

**Need more help?** Check:
- `README.md` - Project overview
- `ARCHITECTURE.md` - System design
- `Assignment_English_Translation.md` - Requirements
- `Quick_Reference_Guide.md` - Key concepts

