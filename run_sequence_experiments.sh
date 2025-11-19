#!/bin/bash
# Script to run sequence length experiments
# Tests different L values and compares their performance

echo "========================================="
echo "Sequence Length Experiment Suite"
echo "========================================="
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check Python availability
if ! command -v python &> /dev/null; then
    echo "Error: Python not found"
    exit 1
fi

echo "Python version: $(python --version)"
echo ""

# Set default parameters
EPOCHS=${1:-30}  # Default 30 epochs (faster for comparison)
CONFIG=${2:-config/config.yaml}

echo "Configuration:"
echo "  Epochs: $EPOCHS"
echo "  Config: $CONFIG"
echo ""

# Run experiments
echo "Starting experiments..."
echo "Testing L = 1, 10, 50, 100, 500"
echo ""

python experiments_sequence_length.py \
    --sequence-lengths 1 10 50 100 500 \
    --epochs $EPOCHS \
    --config $CONFIG \
    --output-dir experiments/sequence_length_comparison

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Experiments completed successfully!"
    echo "========================================="
    echo ""
    echo "Results saved to: experiments/sequence_length_comparison/"
    echo ""
    echo "Generated files:"
    echo "  - results_summary.json (detailed metrics)"
    echo "  - comparative_analysis.png (visualizations)"
    echo "  - analysis_report.txt (text report)"
    echo "  - best_model_L*.pt (trained models)"
    echo ""
else
    echo ""
    echo "========================================="
    echo "Experiments failed with exit code: $EXIT_CODE"
    echo "========================================="
    exit $EXIT_CODE
fi

