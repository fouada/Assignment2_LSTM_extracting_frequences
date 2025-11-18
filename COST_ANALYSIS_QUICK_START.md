# 💰 Cost Analysis - Quick Start Guide

**Get cost insights and optimization recommendations in under 5 minutes!**

---

## ⚡ Super Quick Start

### 1. Run Training (Automatic Cost Analysis)

```bash
python main.py
```

**Output includes:**
- Training metrics
- Test metrics  
- **➕ Cost analysis dashboard**
- **➕ Optimization recommendations**

### 2. View Results

```bash
cd experiments/lstm_frequency_extraction_*/cost_analysis/
ls -lh
```

**You'll find:**
```
📊 cost_dashboard.png          # Beautiful visual dashboard
📈 cost_comparison.png         # Detailed cost charts
📄 cost_analysis.json          # Complete data (machine-readable)
```

### 3. Open the Dashboard

**On Mac:**
```bash
open experiments/lstm_frequency_extraction_*/cost_analysis/cost_dashboard.png
```

**On Linux:**
```bash
xdg-open experiments/lstm_frequency_extraction_*/cost_analysis/cost_dashboard.png
```

**On Windows:**
```bash
start experiments\lstm_frequency_extraction_*\cost_analysis\cost_dashboard.png
```

---

## 📊 What You'll See

### Cost Dashboard Components

```
┌─────────────────────────────────────────────────────────┐
│  Cost Breakdown  │  Cloud Comparison │ Efficiency Score │
│    (Pie Chart)   │   (Bar Chart)     │     (Gauge)      │
├─────────────────────────────────────────────────────────┤
│         Resource Usage         │  Environmental Impact │
│         (Comparison)           │    (Equivalents)      │
├─────────────────────────────────────────────────────────┤
│          Optimization Recommendations Matrix            │
│          (Priority × Effort with Cost Savings)          │
└─────────────────────────────────────────────────────────┘
```

### Key Metrics at a Glance

| Metric | What It Tells You | Good Value |
|--------|-------------------|------------|
| **Training Cost** | $ to train locally | < $0.01 ✅ |
| **Cloud Cost** | $ if you used AWS/Azure/GCP | Compare options |
| **Inference Time** | ms per prediction | < 1ms ✅ |
| **Efficiency Score** | Overall optimization (0-100) | > 80 🏆 |
| **Carbon Footprint** | kg CO₂ emissions | < 0.1kg 🌍 |

---

## 🎯 Top 3 Quick Wins

### 1. Enable Mixed Precision (GPU Only) - 50% Cost Reduction

```python
# Add to training loop
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)
scaler.scale(loss).backward()
```

### 2. Batch Your Inference - 40% Faster

```python
# Instead of one-by-one:
for sample in samples:
    model(sample)

# Do batches:
batch_size = 32
for i in range(0, len(samples), batch_size):
    model(samples[i:i+batch_size])
```

### 3. Use Spot Instances (Cloud) - 70% Savings

```bash
# AWS example
aws ec2 request-spot-instances --instance-type g4dn.xlarge --spot-price "0.20"
```

---

## 📈 Generate Detailed Report

For a comprehensive markdown report with recommendations:

```bash
python cost_analysis_report.py
```

**This creates:**
```
experiments/lstm_frequency_extraction_*/cost_analysis/
└── COST_ANALYSIS_SUMMARY.md  ← Read this!
```

**The report includes:**
- ✅ Detailed cost breakdown by category
- ✅ Cloud provider comparison table
- ✅ Top 5 optimization recommendations with code
- ✅ Environmental impact analysis
- ✅ ROI projections
- ✅ Next steps

---

## 🔧 Customize Analysis

### Provide Accurate Training Time

```bash
# Time your training first
time python main.py

# Then provide exact time (in seconds) for accurate analysis
python cost_analysis_report.py --training-time 420  # 7 minutes
```

### Analyze Specific Experiment

```bash
python cost_analysis_report.py --experiment-dir experiments/lstm_frequency_extraction_20251118_002838
```

### Disable Cost Analysis

**Edit `config/config.yaml`:**
```yaml
cost_analysis:
  enabled: false
```

---

## 💡 Understanding Your Results

### Training Costs

```
Your cost: $0.008
AWS cost:  $0.36 (45x more expensive!)
Azure:     $0.36
GCP:       $0.29

💡 Insight: Local training is most cost-effective for this model
```

### Inference Performance

```
Latency: 0.15 ms
Throughput: 6,667 samples/sec

💡 Insight: Fast enough for real-time applications
```

### Efficiency Score

```
Score: 85/100 🏆

Breakdown:
├── Training Speed:  90/100 ✅
├── Model Size:      88/100 ✅
├── Accuracy:        99/100 ✅
└── Inference Speed: 75/100 ⚡ (room for improvement)

💡 Insight: Excellent overall, focus on inference optimization
```

---

## 🚀 Common Use Cases

### Use Case 1: Budget Planning

**Question:** "How much will it cost to train 100 models?"

**Answer:**
```bash
python cost_analysis_report.py
# Check: Training Cost = $0.008

100 models × $0.008 = $0.80 total
```

### Use Case 2: Production Scaling

**Question:** "What if I need 1M predictions per day?"

**Answer:**
```python
# From cost analysis JSON:
cost_per_1k = 0.0000167  # $
daily_predictions = 1_000_000

daily_cost = (daily_predictions / 1000) * cost_per_1k
monthly_cost = daily_cost * 30

# Result: ~$0.50/day = $15/month
```

### Use Case 3: Cloud vs Local Decision

**Question:** "Should I train in the cloud?"

**Decision Matrix:**
```
If training time < 1 hour:     Use LOCAL  (cheaper)
If training time > 10 hours:   Use CLOUD  (faster)
If need GPU immediately:       Use CLOUD  (spot instances)
If experimenting:              Use LOCAL  (full control)
```

### Use Case 4: Reducing Carbon Footprint

**Question:** "How can I be more environmentally friendly?"

**Actions:**
1. Check current footprint in cost analysis
2. Use efficiency recommendations  
3. Consider green cloud regions
4. Optimize model to use less compute

---

## 📚 Learn More

**Comprehensive Guide:** See `docs/COST_ANALYSIS_GUIDE.md` for:
- Detailed explanations of all metrics
- Complete optimization strategies
- Cloud provider deep-dive
- Environmental impact details
- Advanced cost modeling
- FAQ and troubleshooting

**Quick Links:**
- [Full Documentation](docs/COST_ANALYSIS_GUIDE.md)
- [Architecture Details](docs/ARCHITECTURE.md)
- [Main README](README.md)

---

## 🎓 5-Minute Exercise

**Try this to see cost analysis in action:**

```bash
# Step 1: Run baseline
python main.py

# Step 2: Note efficiency score
# Check: experiments/*/cost_analysis/cost_dashboard.png

# Step 3: Implement top recommendation
# Example: Reduce batch size for memory efficiency

# Step 4: Run again
python main.py

# Step 5: Compare results
# Compare old and new dashboards
```

---

## ❓ Quick FAQ

**Q: Is this free?**  
A: Yes! Cost analysis is built-in and free.

**Q: Does it slow down training?**  
A: No! Analysis runs after training completes. Impact: ~5 seconds.

**Q: What if I don't want it?**  
A: Set `cost_analysis.enabled: false` in `config/config.yaml`

**Q: Can I export data?**  
A: Yes! Check `cost_analysis.json` for all data in JSON format.

**Q: How accurate is it?**  
A: Very accurate for local costs (±10%), cloud costs based on 2025 pricing.

---

## 🎉 You're Ready!

That's it! You now know how to:
- ✅ Run automatic cost analysis
- ✅ Read the dashboard
- ✅ Understand key metrics  
- ✅ Apply quick optimizations
- ✅ Make informed decisions

**Next step:** Run `python main.py` and explore your first cost report!

---

**Questions?** See the [comprehensive guide](docs/COST_ANALYSIS_GUIDE.md) or check the logs in `cost_analysis.log`.

**Happy optimizing! 💰⚡🌍**

