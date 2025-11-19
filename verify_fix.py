"""
Verify that the fix is actually in the code and working
"""

import torch
import yaml

from src.data.signal_generator import create_train_test_generators
from src.data.dataset import create_dataloaders
from src.models.lstm_extractor import create_model

def verify_fix():
    print("="*80)
    print("VERIFYING THE FIX IS APPLIED")
    print("="*80)
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create data
    train_generator, _ = create_train_test_generators(
        frequencies=config['data']['frequencies'],
        duration=config['data']['duration'],
        sampling_rate=config['data']['sampling_rate'],
        train_seed=config['data']['train_seed'],
        test_seed=config['data']['test_seed']
    )
    
    train_loader, _ = create_dataloaders(
        train_generator=train_generator,
        test_generator=train_generator,
        batch_size=64,
        normalize=False,
        device='cpu'
    )
    
    # Create fresh model
    model = create_model(config['model'])
    model.eval()
    
    # Read the source code to verify
    print("\n🔍 Checking source code...")
    with open('src/models/lstm_extractor.py', 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines[168:178], start=169):
            print(f"  Line {i}: {line.rstrip()}")
    
    # Test with a batch
    for batch in train_loader:
        inputs = batch['input']  # [64, 5]
        targets = batch['target']
        break
    
    print(f"\n📊 TEST 1: Batch processing")
    print(f"Input shape: {inputs.shape}")
    
    model.reset_state()
    outputs = model(inputs, reset_state=False)
    
    print(f"Output shape: {outputs.shape}")
    print(f"Output std: {outputs.std():.8f}")
    
    if outputs.std() < 0.001:
        print("❌ FAIL: Outputs are still constant! Fix not working!")
        print(f"   All outputs: {outputs[:10].flatten()}")
    else:
        print("✅ PASS: Outputs vary correctly!")
        print(f"   First 5: {outputs[:5].flatten()}")
        print(f"   Last 5: {outputs[-5:].flatten()}")
    
    # Test sequential processing to compare
    print(f"\n📊 TEST 2: Sequential processing (for comparison)")
    model.reset_state()
    outputs_seq = []
    for i in range(len(inputs)):
        out = model(inputs[i:i+1], reset_state=False)
        outputs_seq.append(out.item())
    outputs_seq = torch.tensor(outputs_seq).unsqueeze(1)
    
    print(f"Sequential output std: {outputs_seq.std():.8f}")
    print(f"Difference from batch: {(outputs - outputs_seq).abs().max():.8f}")
    
    if (outputs - outputs_seq).abs().max() < 1e-5:
        print("✅ PERFECT: Batch and sequential match!")
    else:
        print("❌ MISMATCH: Batch and sequential don't match!")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    verify_fix()


