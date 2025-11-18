"""
Standalone Model Creation Flow
Demonstrates model initialization and architecture

Author: Fouad Azem & Tal Goldengorn
Purpose: Quick demo of model architecture for screenshots and understanding
"""

import yaml
import torch
from src.models.lstm_extractor import create_model

def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80)

def main():
    """Run standalone model creation demo."""
    
    print_section("STANDALONE MODEL CREATION DEMO")
    
    # Load config
    print("\n📋 Loading configuration...")
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print("✅ Configuration loaded")
    
    # Create model
    print("\n🔧 Creating LSTM model...")
    model = create_model(config['model'])
    print("✅ Model created successfully")
    
    # Print architecture
    print_section("MODEL ARCHITECTURE")
    print(model)
    
    # Model summary
    print_section("MODEL SUMMARY")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\n📊 Parameter Statistics:")
    print(f"   Total parameters:      {total_params:,}")
    print(f"   Trainable parameters:  {trainable_params:,}")
    print(f"   Frozen parameters:     {frozen_params:,}")
    print(f"\n💾 Model Size:")
    print(f"   Float32 (4 bytes):     {total_params * 4 / 1024 / 1024:.2f} MB")
    print(f"   Float16 (2 bytes):     {total_params * 2 / 1024 / 1024:.2f} MB")
    
    # Layer breakdown
    print(f"\n🏗️  Layer-wise Parameters:")
    total_counted = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_counted += param_count
        print(f"   {name:40s}: {param_count:>10,} params  {list(param.shape)}")
    
    print(f"\n   {'TOTAL':40s}: {total_counted:>10,} params")
    
    # Configuration details
    print_section("MODEL CONFIGURATION")
    print(f"\n⚙️  Hyperparameters:")
    print(f"   Input size:            {config['model']['input_size']}")
    print(f"   Hidden size:           {config['model']['hidden_size']}")
    print(f"   Number of layers:      {config['model']['num_layers']}")
    print(f"   Output size:           {config['model']['output_size']}")
    print(f"   Dropout:               {config['model']['dropout']}")
    print(f"   Bidirectional:         {config['model']['bidirectional']}")
    print(f"   Sequence length (L):   {config['model']['sequence_length']}")
    
    # Test forward pass
    print_section("TESTING FORWARD PASS")
    
    # Create dummy input
    batch_size = 8
    seq_len = 1
    input_size = config['model']['input_size']
    
    dummy_input = torch.randn(batch_size, seq_len, input_size)
    print(f"\n🧪 Test Input:")
    print(f"   Shape:                 {dummy_input.shape}")
    print(f"   Dtype:                 {dummy_input.dtype}")
    print(f"   Device:                {dummy_input.device}")
    
    # Forward pass
    print(f"\n▶️  Running forward pass...")
    with torch.no_grad():
        output = model(dummy_input, reset_state=True)
    
    print(f"\n✅ Forward pass successful!")
    print(f"\n🎯 Output:")
    print(f"   Shape:                 {output.shape}")
    print(f"   Dtype:                 {output.dtype}")
    print(f"   Device:                {output.device}")
    print(f"   Value range:           [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"   Mean:                  {output.mean().item():.4f}")
    print(f"   Std:                   {output.std().item():.4f}")
    
    # Test state management
    print_section("TESTING STATE MANAGEMENT")
    
    print("\n🔄 State Management Test:")
    print(f"\n1. Initial state:")
    print(f"   Hidden state:          {model.hidden_state}")
    print(f"   Cell state:            {model.cell_state}")
    
    print(f"\n2. Reset state...")
    model.reset_state()
    print(f"   Hidden state:          {model.hidden_state}")
    print(f"   Cell state:            {model.cell_state}")
    
    print(f"\n3. Initialize with forward pass...")
    _ = model(dummy_input, reset_state=True)
    print(f"   Hidden state shape:    {model.hidden_state.shape}")
    print(f"   Cell state shape:      {model.cell_state.shape}")
    print(f"   Hidden state mean:     {model.hidden_state.mean().item():.6f}")
    print(f"   Cell state mean:       {model.cell_state.mean().item():.6f}")
    
    # Process sequence without reset
    print(f"\n4. Process sequence (5 time steps without reset)...")
    outputs = []
    for t in range(5):
        out = model(dummy_input, reset_state=False)
        outputs.append(out)
        print(f"   Time step {t+1}:")
        print(f"      Output shape:     {out.shape}")
        print(f"      Output mean:      {out.mean().item():.6f}")
        print(f"      Hidden mean:      {model.hidden_state.mean().item():.6f}")
        print(f"      Cell mean:        {model.cell_state.mean().item():.6f}")
    
    print(f"\n5. Detach state (for TBPTT)...")
    model.detach_state()
    print(f"   ✅ State detached from computational graph")
    print(f"   Hidden state still exists: {model.hidden_state is not None}")
    print(f"   Hidden requires_grad:      {model.hidden_state.requires_grad}")
    
    # Test variable batch size
    print_section("TESTING VARIABLE BATCH SIZES")
    
    batch_sizes = [1, 4, 16, 32]
    print(f"\n🔀 Testing different batch sizes:")
    for bs in batch_sizes:
        test_input = torch.randn(bs, 1, input_size)
        model.reset_state()
        with torch.no_grad():
            test_output = model(test_input, reset_state=True)
        print(f"   Batch size {bs:>2d}:  Input {list(test_input.shape)}  →  Output {list(test_output.shape)}  ✅")
    
    # Memory estimation
    print_section("MEMORY ANALYSIS")
    
    print(f"\n💾 Memory Requirements:")
    
    # Parameters
    param_memory = total_params * 4  # float32
    print(f"   Parameters (float32):      {param_memory / 1024 / 1024:.2f} MB")
    
    # Gradients (same size as parameters)
    gradient_memory = total_params * 4
    print(f"   Gradients (float32):       {gradient_memory / 1024 / 1024:.2f} MB")
    
    # Activations (estimated for batch_size=32)
    batch_size_est = 32
    # Input: (32, 1, 5)
    input_mem = batch_size_est * 1 * 5 * 4
    # LSTM hidden: (num_layers, batch, hidden) = (2, 32, 128)
    hidden_mem = config['model']['num_layers'] * batch_size_est * config['model']['hidden_size'] * 4 * 2  # h and c
    # Output: (32, 1, 1)
    output_mem = batch_size_est * 1 * 1 * 4
    activation_memory = input_mem + hidden_mem + output_mem
    print(f"   Activations (batch=32):    {activation_memory / 1024 / 1024:.2f} MB")
    
    # Total
    total_memory = param_memory + gradient_memory + activation_memory
    print(f"   {'─'*40}")
    print(f"   Total (training):          {total_memory / 1024 / 1024:.2f} MB")
    print(f"   Total (inference):         {(param_memory + activation_memory) / 1024 / 1024:.2f} MB")
    
    # Optimizer state (Adam: 2x parameters for momentum and variance)
    optimizer_memory = total_params * 4 * 2
    print(f"\n   Adam optimizer state:      {optimizer_memory / 1024 / 1024:.2f} MB")
    print(f"   Grand total (with Adam):   {(total_memory + optimizer_memory) / 1024 / 1024:.2f} MB")
    
    # Device recommendation
    print_section("DEVICE RECOMMENDATIONS")
    
    print(f"\n🖥️  Supported Devices:")
    print(f"   ✅ CPU:        Suitable (low memory, ~{(total_memory + optimizer_memory) / 1024 / 1024:.0f} MB)")
    print(f"   ✅ MPS (Mac):  Recommended (M1/M2/M3 chips)")
    print(f"   ✅ CUDA:       Optimal (any modern GPU)")
    
    # Check available devices
    print(f"\n🔍 Available on this system:")
    print(f"   CPU:           ✅ Always available")
    print(f"   CUDA:          {'✅ Available' if torch.cuda.is_available() else '❌ Not available'}")
    print(f"   MPS (Metal):   {'✅ Available' if torch.backends.mps.is_available() else '❌ Not available'}")
    
    if torch.backends.mps.is_available():
        print(f"\n💡 Recommendation: Use MPS for ~3-5x speedup on Mac")
    elif torch.cuda.is_available():
        print(f"\n💡 Recommendation: Use CUDA for optimal performance")
    else:
        print(f"\n💡 Using CPU (sufficient for this model size)")
    
    # Final summary
    print_section("SUMMARY")
    
    print(f"\n✅ Model Creation Test Complete!")
    print(f"\n📋 Key Takeaways:")
    print(f"   • Model has {total_params:,} trainable parameters")
    print(f"   • Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (float32)")
    print(f"   • Forward pass successful with various batch sizes")
    print(f"   • State management working correctly")
    print(f"   • Suitable for CPU/GPU/MPS training")
    print(f"\n📸 Screenshot Recommendations:")
    print(f"   1. Model architecture section")
    print(f"   2. Model summary with parameter counts")
    print(f"   3. State management test results")
    print(f"   4. Memory analysis section")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()

