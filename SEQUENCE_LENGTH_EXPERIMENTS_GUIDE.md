# Sequence Length (L) Experiments Guide

## Overview

This guide explains the comprehensive experiments designed to understand how different sequence lengths (L) affect LSTM performance in frequency extraction tasks.

## What is Sequence Length (L)?

**Sequence Length (L)** determines how many consecutive time samples the LSTM processes in a single forward pass:

- **L = 1**: Single sample mode (current assignment default)
- **L > 1**: Sequence mode (e.g., L=10, 50, 100, 500)

## Theoretical Impact of L

### L = 1 (Single Sample Mode)

**Temporal Processing:**
- LSTM sees ONE time point at a time
- Must rely entirely on hidden state (h_t, c_t) for temporal memory
- Requires explicit state preservation between samples
- Learns temporal patterns incrementally through state propagation

**Advantages:**
- Memory efficient
- Online learning capability
- Demonstrates pure state-based learning

**Challenges:**
- Slower convergence
- Longer gradient paths
- No direct access to temporal context

### L > 1 (Sequence Mode)

**Temporal Processing:**
- LSTM sees MULTIPLE consecutive time points
- Can learn patterns within the sequence directly
- Backpropagation Through Time (BPTT) captures L time steps naturally
- Both internal gates AND sequence structure contribute to learning

**Example with L = 50 (0.05 seconds at 1000 Hz):**
- For 1 Hz frequency: sees ~5% of one cycle
- For 7 Hz frequency: sees ~35% of one cycle
- Can detect local curvature, trends, and phase information

**Advantages:**
- Richer temporal context
- Better gradient flow
- Faster convergence
- More natural for LSTM architecture

**Challenges:**
- Higher memory usage
- Less suitable for streaming/online scenarios

## Frequency-Specific Analysis

Given our frequencies (1, 3, 5, 7 Hz) at 1000 Hz sampling:

| Frequency | Period (samples) | L=1 | L=10 | L=50 | L=100 | L=500 |
|-----------|------------------|-----|------|------|-------|-------|
| 1 Hz | 1000 | 0.1% | 1% | 5% | 10% | 50% |
| 3 Hz | 333 | 0.3% | 3% | 15% | 30% | 150% (1.5 cycles) |
| 5 Hz | 200 | 0.5% | 5% | 25% | 50% | 250% (2.5 cycles) |
| 7 Hz | 143 | 0.7% | 7% | 35% | 70% | 350% (3.5 cycles) |

**Key Insight:** Larger L values provide increasingly complete cycle information, especially for higher frequencies.

## Experiment Setup

### Tested Configurations

We test 5 different L values:
1. **L = 1**: Baseline (pure state-based)
2. **L = 10**: Minimal temporal context
3. **L = 50**: Moderate context (partial cycles)
4. **L = 100**: Good context (near-complete cycles for higher frequencies)
5. **L = 500**: Rich context (multiple complete cycles)

### Evaluation Metrics

For each L value, we measure:

1. **Performance Metrics:**
   - Final Training MSE
   - Final Test MSE
   - Generalization Gap (Test - Train MSE)

2. **Efficiency Metrics:**
   - Training time (seconds)
   - Convergence speed (epochs to 90% performance)
   - Throughput (samples/second)
   - Memory usage

3. **Learning Dynamics:**
   - Training loss curves
   - Test loss curves
   - Best epoch achieved

## Running the Experiments

### Quick Start

```bash
# Make script executable
chmod +x run_sequence_experiments.sh

# Run with default settings (30 epochs)
./run_sequence_experiments.sh

# Run with custom epochs
./run_sequence_experiments.sh 50

# Run with custom config
./run_sequence_experiments.sh 50 config/config_production.yaml
```

### Manual Execution

```bash
# Full command with all options
python experiments_sequence_length.py \
    --sequence-lengths 1 10 50 100 500 \
    --epochs 30 \
    --config config/config.yaml \
    --output-dir experiments/sequence_length_comparison
```

### Custom Experiments

Test specific L values:
```bash
python experiments_sequence_length.py \
    --sequence-lengths 1 20 100 \
    --epochs 40
```

## Output Files

After running experiments, you'll find:

```
experiments/sequence_length_comparison/
├── results_summary.json          # Detailed metrics for all experiments
├── comparative_analysis.png      # Visualizations (6 plots)
├── analysis_report.txt          # Comprehensive text report
├── best_model_L1.pt            # Best model for L=1
├── best_model_L10.pt           # Best model for L=10
├── best_model_L50.pt           # Best model for L=50
├── best_model_L100.pt          # Best model for L=100
└── best_model_L500.pt          # Best model for L=500
```

### Visualizations

The `comparative_analysis.png` includes:

1. **Training Loss Convergence**: How quickly each L converges
2. **Test Loss Convergence**: Generalization over training
3. **Final Performance**: Bar chart of final MSE values
4. **Training Time**: Computational cost comparison
5. **Convergence Speed**: Epochs needed to reach good performance
6. **Generalization Gap**: Test vs Train MSE difference

## Interpreting Results

### What to Look For

1. **Performance vs Complexity Trade-off:**
   - Does larger L improve accuracy?
   - At what point do diminishing returns occur?

2. **Convergence Behavior:**
   - Do larger L values converge faster?
   - Is training more stable with larger L?

3. **Generalization:**
   - Which L generalizes best to test data?
   - Is there overfitting with very large L?

4. **Computational Efficiency:**
   - How does training time scale with L?
   - Is the performance gain worth the computational cost?

### Expected Insights

Based on theory, we expect:

1. **L=1** will have:
   - Slowest convergence
   - Good generalization (if properly trained)
   - Lowest memory usage
   - Demonstrates pure state-based learning

2. **L=10-50** will likely show:
   - Improved convergence speed
   - Better gradient flow
   - Balanced performance/efficiency

3. **L=100-500** may show:
   - Best final accuracy
   - Fastest convergence
   - Highest memory usage
   - Potential over-reliance on direct pattern matching vs state memory

## Assignment Justification

If using **L ≠ 1** in your assignment, include:

### Required Elements

1. **Choice Justification:**
   ```
   Example: "I chose L=50 because it provides sufficient temporal context 
   to capture partial cycles of all frequencies (5%-35% of a cycle), 
   enabling the LSTM to learn both local curvature patterns and 
   temporal dependencies through its hidden state."
   ```

2. **Temporal Learning Explanation:**
   ```
   Example: "With L=50, the LSTM learns through two mechanisms:
   - Direct pattern recognition within the 50-sample window
   - State-based memory for patterns spanning beyond L
   This hybrid approach leverages LSTM's full capabilities."
   ```

3. **Output Handling:**
   ```
   Example: "For L=50 sequences, the model outputs predictions for all 
   50 time steps. During training, we compute loss across the entire 
   sequence, allowing gradient flow through time. For inference, we can 
   use either the full sequence output or focus on specific positions."
   ```

### Comparison with L=1

Include results showing:
- Convergence speed improvement
- Final performance comparison
- Training efficiency gains
- Any trade-offs observed

## Advanced Experiments

### Stride Analysis

Test overlapping vs non-overlapping sequences:

```python
# Non-overlapping (default)
stride = sequence_length  # No overlap

# 50% overlap
stride = sequence_length // 2

# Heavy overlap (for more training samples)
stride = 1
```

### Bidirectional LSTM with Sequences

```yaml
# In config.yaml
model:
  bidirectional: true
  sequence_length: 50
```

This allows the LSTM to see future context as well!

### Variable Length Testing

Test if optimal L depends on frequency:
```bash
python experiments_sequence_length.py \
    --sequence-lengths 1 10 20 50 75 100 150 200 300 500 \
    --epochs 50
```

## Troubleshooting

### Out of Memory

If you get OOM errors with large L:

```bash
# Reduce batch size for large L
# Edit config.yaml:
training:
  batch_size: 16  # or 8 for very large L
```

### Slow Training

For faster experimentation:

```bash
# Use fewer epochs
./run_sequence_experiments.sh 20

# Test fewer L values
python experiments_sequence_length.py --sequence-lengths 1 50 100
```

### Poor Convergence

If results are poor:

1. Check learning rate (may need adjustment for different L)
2. Increase gradient clipping for large L
3. Try longer training (more epochs)
4. Adjust hidden size if needed

## Integration with Assignment

### Using Experiment Results

1. **Choose Best L** based on your criteria:
   - Best accuracy: Use L with lowest test MSE
   - Best efficiency: Use L with best time/performance ratio
   - Balanced: Use mid-range L (e.g., 50-100)

2. **Update Config:**
   ```yaml
   model:
     sequence_length: 50  # Your chosen L
   ```

3. **Update Data Loading:**
   The experiment script automatically handles this, but for main training:
   ```python
   if config['model']['sequence_length'] > 1:
       from src.data.sequence_dataset import create_sequence_dataloaders
       train_loader, test_loader = create_sequence_dataloaders(...)
   else:
       from src.data.dataset import create_dataloaders
       train_loader, test_loader = create_dataloaders(...)
   ```

## Questions to Answer

After running experiments, consider:

1. **Which L value performed best? Why?**
2. **How does temporal context affect learning speed?**
3. **Is there a point of diminishing returns?**
4. **How does L affect generalization?**
5. **What's the optimal L for this specific task?**
6. **How would you choose L for a different frequency extraction task?**

## References

- Assignment Section 4.3: Alternative and Justification
- LSTM Architecture: Understanding hidden state propagation
- BPTT: Backpropagation Through Time principles
- Sequence Modeling: Best practices for temporal data

## Summary

This experiment suite provides:
- ✅ Systematic comparison of L values
- ✅ Quantitative performance metrics
- ✅ Computational efficiency analysis
- ✅ Visual comparative analysis
- ✅ Comprehensive documentation
- ✅ Assignment-ready justification materials

Run the experiments, analyze the results, and make an informed decision about the optimal sequence length for your frequency extraction task!

---

**Note:** For assignment submission, include the generated `comparative_analysis.png` and key findings from `analysis_report.txt` to demonstrate thorough investigation of the sequence length parameter.

