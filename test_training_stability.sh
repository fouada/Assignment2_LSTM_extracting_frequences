#!/bin/bash
# Quick test script for training stability fixes

echo "=================================================="
echo "Training Stability Test - Three Options"
echo "=================================================="
echo ""

echo "Choose an option:"
echo "  1) Quick Fix (faster scheduler + gradient clipping)"
echo "  2) Cosine Schedule (RECOMMENDED - proactive LR reduction)"
echo "  3) Lower Learning Rate (most conservative)"
echo ""
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo ""
        echo "Running Option 1: Quick Fix"
        echo "- Scheduler patience: 5 (was 10)"
        echo "- Gradient clip: 0.5 (was 1.0)"
        echo "- Early stopping: 7 (was 10)"
        echo ""
        python main.py
        ;;
    2)
        echo ""
        echo "Running Option 2: Cosine Annealing (RECOMMENDED)"
        echo "- Proactive LR reduction"
        echo "- Smooth decay throughout training"
        echo "- Best for preventing spikes"
        echo ""
        python main.py --config config/config_cosine_schedule.yaml
        ;;
    3)
        echo ""
        echo "Running Option 3: Lower Learning Rate"
        echo "- LR: 0.0003 (most conservative)"
        echo ""
        echo "Note: Manually edit config/config.yaml first:"
        echo "  Change: learning_rate: 0.0005"
        echo "  To:     learning_rate: 0.0003"
        echo ""
        read -p "Have you made this change? (y/n): " confirm
        if [ "$confirm" = "y" ]; then
            python main.py
        else
            echo "Please edit config/config.yaml first, then run: python main.py"
        fi
        ;;
    *)
        echo "Invalid choice. Please run again and select 1, 2, or 3."
        exit 1
        ;;
esac

echo ""
echo "=================================================="
echo "Training Complete!"
echo "=================================================="
echo ""
echo "Check the plots in: experiments/[latest]/plots/training_history.png"
echo ""
echo "Look for:"
echo "  ✅ No spike at epoch 13-14"
echo "  ✅ Smooth LR reduction curve"
echo "  ✅ Continuous loss decrease"
echo ""

