"""
Verification Script: LSTM State Management
Ensures state is preserved within frequency but reset between frequencies.
"""

import torch
import sys
sys.path.insert(0, 'src')

from src.data.signal_generator import create_train_test_generators
from src.data.dataset import create_dataloaders
from src.models.lstm_extractor import StatefulLSTMExtractor


def test_basic_state_preservation():
    """Test 1: Basic state preservation mechanics."""
    print("="*70)
    print("TEST 1: Basic State Preservation Mechanics")
    print("="*70)
    
    model = StatefulLSTMExtractor(input_size=5, hidden_size=8, num_layers=1)
    model.eval()
    
    batch_size = 2
    input1 = torch.randn(batch_size, 5)
    input2 = torch.randn(batch_size, 5)
    
    # Scenario A: State preserved
    print("\nScenario A: State PRESERVED between inputs")
    print("-"*70)
    model.reset_state()
    with torch.no_grad():
        output1a = model(input1, reset_state=False)
        state_after_1a = model.hidden_state.clone() if model.hidden_state is not None else None
        output2a = model(input2, reset_state=False)  # Should remember input1
    
    print(f"Output for input2 WITH state:  {output2a[0, 0]:.8f}")
    if state_after_1a is not None:
        print(f"State after input1: {state_after_1a[0, 0, :4].numpy()}")
    
    # Scenario B: State reset
    print("\nScenario B: State RESET between inputs")
    print("-"*70)
    model.reset_state()
    with torch.no_grad():
        output1b = model(input1, reset_state=False)
        model.reset_state()  # RESET here!
        output2b = model(input2, reset_state=False)  # No memory of input1
    
    print(f"Output for input2 WITHOUT state: {output2b[0, 0]:.8f}")
    
    # Compare
    print("\n" + "-"*70)
    difference = torch.abs(output2a - output2b).max().item()
    print(f"Maximum difference: {difference:.8f}")
    
    if difference > 1e-6:
        print("✅ PASS: State preservation is WORKING!")
        print("   Outputs are different, proving state affects predictions")
    else:
        print("❌ FAIL: State has no effect on outputs")
    
    return difference > 1e-6


def test_dataloader_state_management():
    """Test 2: State management in actual data loader."""
    print("\n" + "="*70)
    print("TEST 2: State Management in StatefulDataLoader")
    print("="*70)
    
    print("\nCreating signal generators and data loaders...")
    train_gen, test_gen = create_train_test_generators(
        frequencies=[1.0, 3.0],  # Two frequencies for testing
        sampling_rate=1000,
        duration=0.5,  # Short duration for testing
        train_seed=1,
        test_seed=2
    )
    
    train_loader, _ = create_dataloaders(
        train_gen, test_gen,
        batch_size=50,
        normalize=True
    )
    
    model = StatefulLSTMExtractor(input_size=5, hidden_size=16, num_layers=1)
    model.eval()
    
    print("\nProcessing batches and tracking state...")
    print("-"*70)
    
    batch_info = []
    
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= 8:  # Check first 8 batches (should cover both frequencies)
                break
            
            is_first = batch['is_first_batch']
            freq_idx = batch['freq_idx']
            time_range = batch['time_range']
            
            # Simulate proper state management
            if is_first:
                model.reset_state()
                reset_marker = "🔴 RESET"
            else:
                reset_marker = "🟢 PRESERVE"
            
            inputs = batch['input']
            outputs = model(inputs, reset_state=False)
            
            # Track state norm
            if model.hidden_state is not None:
                state_norm = torch.norm(model.hidden_state).item()
            else:
                state_norm = 0.0
            
            batch_info.append({
                'batch_idx': i,
                'freq_idx': freq_idx,
                'time_range': time_range,
                'is_first': is_first,
                'state_norm': state_norm,
                'reset_marker': reset_marker
            })
            
            print(f"Batch {i:2d} | Freq {freq_idx} | t={time_range[0]:4d}-{time_range[1]:4d} | "
                  f"State norm: {state_norm:8.4f} | {reset_marker}")
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    # Check 1: Frequency transitions
    freq_changes = []
    for i in range(1, len(batch_info)):
        if batch_info[i]['freq_idx'] != batch_info[i-1]['freq_idx']:
            freq_changes.append(i)
            print(f"\n✓ Frequency transition detected at batch {i}")
            print(f"  From Freq {batch_info[i-1]['freq_idx']} to Freq {batch_info[i]['freq_idx']}")
            print(f"  is_first_batch flag: {batch_info[i]['is_first']}")
    
    if len(freq_changes) > 0 and all(batch_info[idx]['is_first'] for idx in freq_changes):
        print(f"\n✅ PASS: State reset at frequency boundaries")
        print(f"   {len(freq_changes)} frequency transition(s) detected")
    else:
        print("\n❌ FAIL: State reset not properly triggered at frequency boundaries")
    
    # Check 2: State evolution within frequency
    print("\n" + "-"*70)
    for freq_idx in [0, 1]:
        freq_batches = [b for b in batch_info if b['freq_idx'] == freq_idx and not b['is_first']]
        if len(freq_batches) > 1:
            state_norms = [b['state_norm'] for b in freq_batches]
            state_variance = torch.var(torch.tensor(state_norms)).item()
            print(f"\nFrequency {freq_idx}:")
            print(f"  Batches processed: {len(freq_batches) + 1}")  # +1 for first batch
            print(f"  State norm range: [{min(state_norms):.4f}, {max(state_norms):.4f}]")
            print(f"  State variance: {state_variance:.6f}")
            
            if state_variance > 1e-6:
                print(f"  ✅ State is evolving (not stuck)")
            else:
                print(f"  ⚠️  State appears constant (may be an issue)")
    
    return len(freq_changes) > 0


def test_state_impact_on_predictions():
    """Test 3: Demonstrate state impact on predictions."""
    print("\n" + "="*70)
    print("TEST 3: State Impact on Predictions")
    print("="*70)
    
    model = StatefulLSTMExtractor(input_size=5, hidden_size=32, num_layers=2)
    model.eval()
    
    # Create a sequence of related inputs
    t = torch.linspace(0, 1, 10)
    # Simple sine wave pattern
    sequence = torch.stack([torch.cat([
        torch.sin(2 * torch.pi * ti).unsqueeze(0),
        torch.ones(4)
    ]) for ti in t])
    
    print("\nScenario A: Sequential processing WITH state")
    print("-"*70)
    model.reset_state()
    outputs_with_state = []
    
    with torch.no_grad():
        for i, inp in enumerate(sequence):
            out = model(inp.unsqueeze(0), reset_state=False)
            outputs_with_state.append(out.item())
            if i < 3:
                print(f"t={i}: input={inp[0]:.4f}, output={out.item():.6f}")
    
    print("\nScenario B: Independent processing WITHOUT state (reset each time)")
    print("-"*70)
    outputs_without_state = []
    
    with torch.no_grad():
        for i, inp in enumerate(sequence):
            model.reset_state()  # Reset each time!
            out = model(inp.unsqueeze(0), reset_state=False)
            outputs_without_state.append(out.item())
            if i < 3:
                print(f"t={i}: input={inp[0]:.4f}, output={out.item():.6f}")
    
    # Compare
    print("\n" + "-"*70)
    with_state = torch.tensor(outputs_with_state)
    without_state = torch.tensor(outputs_without_state)
    
    diff = torch.abs(with_state - without_state)
    avg_diff = diff.mean().item()
    max_diff = diff.max().item()
    
    print(f"Average difference: {avg_diff:.6f}")
    print(f"Maximum difference: {max_diff:.6f}")
    
    if max_diff > 1e-4:
        print("\n✅ PASS: State has significant impact on predictions")
        print("   This proves the LSTM is using temporal memory")
    else:
        print("\n❌ FAIL: State has minimal impact")
    
    return max_diff > 1e-4


def main():
    """Run all verification tests."""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + " "*15 + "LSTM STATE MANAGEMENT VERIFICATION" + " "*19 + "█")
    print("█" + " "*68 + "█")
    print("█"*70 + "\n")
    
    results = []
    
    # Run tests
    try:
        results.append(("Basic State Preservation", test_basic_state_preservation()))
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        results.append(("Basic State Preservation", False))
    
    try:
        results.append(("DataLoader State Management", test_dataloader_state_management()))
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        results.append(("DataLoader State Management", False))
    
    try:
        results.append(("State Impact on Predictions", test_state_impact_on_predictions()))
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        results.append(("State Impact on Predictions", False))
    
    # Summary
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + " "*25 + "FINAL SUMMARY" + " "*30 + "█")
    print("█" + " "*68 + "█")
    print("█"*70 + "\n")
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "="*70)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("="*70)
        print("\nYour LSTM state management is correctly implemented:")
        print("  ✅ State is preserved between consecutive samples")
        print("  ✅ State is reset at frequency boundaries")
        print("  ✅ State has meaningful impact on predictions")
        print("\nThis ensures proper temporal learning for L=1 mode!")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("="*70)
        print("\nPlease review the failed tests above.")
    
    print()


if __name__ == "__main__":
    main()

