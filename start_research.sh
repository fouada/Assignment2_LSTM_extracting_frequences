#!/bin/bash
# Quick Start Script for Research Framework
# Makes it easy to run research with one command

echo "=========================================="
echo "LSTM Frequency Extraction Research"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"
echo ""

# Check if in virtual environment or offer to activate
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Virtual environment not active"
    
    if [ -d "venv" ]; then
        echo "📦 Found venv directory. Activating..."
        source venv/bin/activate
        echo "✓ Virtual environment activated"
    else
        echo "❌ No venv found. Please run: python3 -m venv venv && source venv/bin/activate"
        exit 1
    fi
else
    echo "✓ Virtual environment active: $VIRTUAL_ENV"
fi

echo ""
echo "=========================================="
echo "Step 1: Testing Research Module"
echo "=========================================="
echo ""

python3 research/test_research.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Tests failed. Please fix issues before running research."
    echo "   Common fixes:"
    echo "   - Install dependencies: pip install -r requirements.txt"
    echo "   - Activate venv: source venv/bin/activate"
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 2: Choose Research Mode"
echo "=========================================="
echo ""
echo "Select mode:"
echo "  1) Quick mode (~30 minutes, for testing)"
echo "  2) Full mode (~2-4 hours, comprehensive)"
echo "  3) Cancel"
echo ""
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        MODE="quick"
        echo ""
        echo "Starting QUICK mode research..."
        echo "Expected time: ~30 minutes"
        ;;
    2)
        MODE="full"
        echo ""
        echo "Starting FULL mode research..."
        echo "Expected time: ~2-4 hours"
        ;;
    3)
        echo "Cancelled."
        exit 0
        ;;
    *)
        echo "Invalid choice."
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Step 3: Running Research Pipeline"
echo "=========================================="
echo ""

python3 research/run_full_research.py --mode $MODE --output-dir ./research/full_study

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ RESEARCH COMPLETED SUCCESSFULLY!"
    echo "=========================================="
    echo ""
    echo "Results saved to: research/full_study/"
    echo ""
    echo "Next steps:"
    echo "  1. Read the report:"
    echo "     open research/full_study/research_report_*.md"
    echo ""
    echo "  2. View visualizations:"
    echo "     open research/full_study/sensitivity/"
    echo "     open research/full_study/comparison/"
    echo ""
    echo "  3. Study the theory:"
    echo "     open research/MATHEMATICAL_ANALYSIS.md"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ RESEARCH FAILED"
    echo "=========================================="
    echo ""
    echo "Please check the error messages above."
    echo "For help, see: RESEARCH_QUICKSTART.md"
    exit 1
fi

