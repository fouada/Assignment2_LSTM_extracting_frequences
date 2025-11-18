#!/bin/bash
# Quick Commands for LSTM Frequency Extraction
# M1 Pro Mac Optimized

PROJECT_DIR="/Users/fouadaz/LearningFromUniversity/Learning/LLMSAndMultiAgentOrchestration/course-materials/assignments/Assignment2_LSTM_extracting_frequences"

cd "$PROJECT_DIR" || exit

echo "🚀 LSTM Frequency Extraction - Quick Commands"
echo "=============================================="
echo ""
echo "Select an option:"
echo ""
echo "1. 🏃 Run Training (uses M1 GPU)"
echo "2. 📊 Launch TensorBoard (all experiments)"
echo "3. 📈 Launch TensorBoard (latest experiment only)"
echo "4. 👀 View Latest Results (plots)"
echo "5. 🧪 Run Tests"
echo "6. 🔍 Check M1 GPU Status"
echo "7. 📦 Update Dependencies"
echo "8. 🧹 Clean & Fresh Install"
echo "9. 📋 List All Experiments"
echo "10. 💻 Interactive Python Shell"
echo ""
read -p "Enter choice [1-10]: " choice

case $choice in
    1)
        echo "🏃 Starting training..."
        uv run main.py
        ;;
    2)
        echo "📊 Launching TensorBoard (all experiments)..."
        echo "Open: http://localhost:6006"
        uv run tensorboard --logdir experiments/
        ;;
    3)
        LATEST_EXP=$(ls -t experiments/ | head -1)
        echo "📈 Launching TensorBoard (latest: $LATEST_EXP)..."
        echo "Open: http://localhost:6006"
        uv run tensorboard --logdir "experiments/$LATEST_EXP/checkpoints/tensorboard/"
        ;;
    4)
        echo "👀 Opening latest plots..."
        LATEST_EXP=$(ls -t experiments/ | head -1)
        open "experiments/$LATEST_EXP/plots/"*.png
        ;;
    5)
        echo "🧪 Running tests..."
        uv run pytest tests/ -v
        ;;
    6)
        echo "🔍 Checking M1 GPU status..."
        uv run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS Available: {torch.backends.mps.is_available()}'); print(f'MPS Built: {torch.backends.mps.is_built()}'); device = torch.device('mps'); x = torch.randn(1000, 1000, device=device); print('✅ M1 GPU is working!')"
        ;;
    7)
        echo "📦 Updating dependencies..."
        uv sync --upgrade
        ;;
    8)
        echo "🧹 Clean & fresh install..."
        read -p "This will remove .venv and uv.lock. Continue? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf .venv uv.lock
            uv sync
            echo "✅ Fresh install complete!"
        fi
        ;;
    9)
        echo "📋 All experiments:"
        echo ""
        ls -lh experiments/
        echo ""
        echo "Total experiments: $(ls -1 experiments/ | wc -l)"
        ;;
    10)
        echo "💻 Starting interactive Python shell..."
        echo "Try: from src.data.signal_generator import create_train_test_generators"
        uv run python
        ;;
    *)
        echo "Invalid choice. Please run again and select 1-10."
        ;;
esac

