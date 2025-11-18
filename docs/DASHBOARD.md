# Interactive Dashboard Guide
## Professional Visualization for LSTM Frequency Extraction

---

## Overview

The Interactive Dashboard provides real-time, professional visualization for monitoring training and analyzing results. Built with Plotly Dash, it offers:

- **5 Interactive Tabs:** Frequency extraction, training progress, error analysis, metrics, architecture
- **Real-time Monitoring:** Watch training as it happens
- **Export Capabilities:** Save plots as PNG/SVG/PDF
- **Network Access:** Share dashboard on local network
- **Mobile Friendly:** Responsive design for all devices

---

## Installation

```bash
# Install dashboard dependencies
pip install dash dash-bootstrap-components plotly kaleido

# Or update all requirements
pip install -r requirements.txt
```

---

## Usage

### Launch with Training (Real-time Monitoring)

```bash
python main_with_dashboard.py
```

**What happens:**
1. Training starts
2. Dashboard launches (port 8050)
3. Browser opens automatically
4. Watch real-time updates every 2 seconds

**Access:** http://localhost:8050

---

### Launch for Existing Experiment

```bash
# Latest experiment
python dashboard.py

# Specific experiment
python dashboard.py --experiment experiments/lstm_frequency_extraction_20251118_002838

# Custom port
python dashboard.py --port 8080
```

---

## Dashboard Features

### Tab 1: Frequency Extraction 📈

**Interactive 2×2 grid showing all 4 frequencies:**
- Mixed signal (gray background)
- Target signal (blue line)
- LSTM predictions (red scatter)
- Real-time MSE per frequency

**Controls:**
- Select frequency (1Hz, 3Hz, 5Hz, 7Hz)
- Adjust time range slider (0-1000 samples)
- Zoom, pan, hover for details
- Export as PNG/SVG

---

### Tab 2: Training Progress 📊

**Real-time training monitoring:**
- Loss curves (train & validation)
- Learning rate schedule
- Gradient norms over time
- Per-epoch training duration

**Features:**
- Logarithmic scales for better visualization
- Updates every 2 seconds during training
- Early stopping indicators
- Interactive legends

---

### Tab 3: Error Analysis 🔍

**Comprehensive error analysis:**
- Error distribution histogram
- Prediction vs Target scatter plot
- Residual plot with zero-error line
- Error by frequency box plots

**Insights:**
- Bias detection
- Variance analysis
- Outlier identification
- Frequency-specific performance

---

### Tab 4: Performance Metrics 📉

**Quantitative evaluation:**
- Train vs Test comparison bar chart
- All metrics: MSE, MAE, RMSE, R², SNR, Correlation
- Per-frequency metrics table
- Generalization analysis

**Features:**
- Interactive charts
- Sortable tables
- Export to CSV
- Copy to clipboard

---

### Tab 5: Model Architecture 🏗️

**Complete model details:**
- Network structure
- Training configuration
- Model statistics (215,041 parameters)
- Layer-by-layer breakdown table

**Information:**
- Input/output shapes
- Parameter counts per layer
- Model size (~0.82 MB)
- Inference speed estimates

---

## Control Panel

### Top Dashboard Controls

- **Experiment Selector:** Browse available experiments
- **Frequency Selector:** Choose f₁, f₂, f₃, or f₄
- **Time Range Slider:** Adjust visible time window
- **Refresh Button:** Reload experiment data
- **Export Button:** Generate comprehensive report

### Metric Cards (Real-time)

Four cards displaying:
1. **Train MSE** - Training set performance
2. **Test MSE** - Test set performance  
3. **R² Score** - Model fit quality (0-1)
4. **Status** - Training state & current epoch

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl + R` | Refresh data |
| `Ctrl + E` | Export current view |
| `← / →` | Navigate between tabs |
| `+ / -` | Zoom in/out on plots |
| `Home` | Reset plot view |
| `Ctrl + C` | Stop dashboard server |

---

## Export Options

### Individual Plots
- Click camera icon (📷) on any plot
- Select format: PNG (300 DPI) or SVG (vector)
- Saves to downloads folder

### Complete Report
- Click "Export Report" button
- Generates comprehensive PDF
- Includes all visualizations and metrics

---

## Network Access

### Share with Team

```bash
# Find your IP address
ifconfig | grep "inet "  # Mac/Linux
ipconfig                  # Windows

# Example: 192.168.1.100

# Start dashboard
python dashboard.py

# Share this URL with team:
# http://192.168.1.100:8050
```

**Works on:**
- Desktop browsers
- Mobile phones
- Tablets
- Any device on same network

---

## Advanced Features

### Real-time Training Integration

```python
# In your training script
from src.visualization.live_monitor import create_live_monitor

# Create monitor
monitor = create_live_monitor(experiment_dir, auto_start=True)
monitor.set_total_epochs(50)

# Update during training
for epoch in range(50):
    train_loss = train_one_epoch()
    val_loss = validate()
    
    monitor.update_epoch(
        epoch=epoch,
        train_loss=train_loss,
        val_loss=val_loss,
        learning_rate=current_lr
    )

# Stop when done
monitor.stop()
```

### Compare Multiple Experiments

```bash
# Terminal 1: Experiment A
python dashboard.py --experiment experiments/exp_A --port 8050

# Terminal 2: Experiment B
python dashboard.py --experiment experiments/exp_B --port 8051

# Open both in browser tabs for side-by-side comparison
```

---

## Troubleshooting

### Port Already in Use

```bash
# Use different port
python dashboard.py --port 8080

# Or kill existing process
lsof -ti:8050 | xargs kill -9
```

### No Experiments Found

```bash
# Run training first
python main.py

# Then launch dashboard
python dashboard.py
```

### Dashboard Shows No Data

**Check that experiment has required files:**
```bash
ls experiments/lstm_frequency_extraction_*/
# Should contain:
# - config.yaml
# - training_history.json
# - test_results.json
# - predictions_f*.npz
```

### Slow Performance

**Solutions:**
- Reduce time range slider window
- Close other browser tabs
- Use Chrome or Firefox
- Check CPU/RAM usage

---

## Best Practices

### During Training
1. Start dashboard before training for real-time monitoring
2. Keep update interval at 1-2 seconds
3. Monitor metric cards for training health
4. Check loss curves for convergence

### After Training
1. Review all 5 tabs systematically
2. Export visualizations for documentation
3. Compare with previous experiments
4. Document insights

### For Presentations
1. Pre-load experiment data
2. Use full screen mode (F11)
3. Prepare key screenshots beforehand
4. Test on target display

---

## Quick Reference

### Launch Commands

```bash
# Train with dashboard
python main_with_dashboard.py

# View latest
python dashboard.py

# View specific
python dashboard.py --experiment <path>

# Custom port
python dashboard.py --port 8080

# Test dashboard
python test_dashboard.py
```

### URLs

- **Local:** http://localhost:8050
- **Network:** http://YOUR_IP:8050

---

## API Reference

### LiveTrainingMonitor

```python
from src.visualization.live_monitor import LiveTrainingMonitor

monitor = LiveTrainingMonitor(experiment_dir, update_interval=1.0)
monitor.start()
monitor.update_epoch(epoch, train_loss, val_loss, ...)
monitor.update_test_results(overall_metrics, per_frequency_metrics)
monitor.stop()
```

### LSTMFrequencyDashboard

```python
from src.visualization.interactive_dashboard import create_dashboard

dashboard = create_dashboard(experiment_dir, port=8050)
dashboard.run(debug=False)
```

---

## Support

- **Quick Commands:** See "Quick Reference" above
- **Troubleshooting:** See troubleshooting section
- **Testing:** Run `python test_dashboard.py`
- **Issues:** Check experiment data files exist

---

**Ready to visualize!** Run: `python main_with_dashboard.py`

