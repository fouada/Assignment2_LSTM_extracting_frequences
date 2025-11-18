"""
Performance and Stress Tests
Tests performance characteristics, scalability, and stress conditions.
Target: Ensure system performs within acceptable bounds.
"""

import pytest
import torch
import numpy as np
import time
import psutil
import os
from pathlib import Path

from src.data import SignalGenerator, SignalConfig, FrequencyExtractionDataset
from src.data.dataset import StatefulDataLoader
from src.models import StatefulLSTMExtractor, create_model
from src.training.trainer import LSTMTrainer
from src.evaluation.metrics import evaluate_model, FrequencyExtractionMetrics


@pytest.mark.performance
@pytest.mark.slow
class TestDataGenerationPerformance:
    """Test data generation performance."""
    
    def test_signal_generation_speed(self, test_seed):
        """Test that signal generation completes within acceptable time."""
        config = SignalConfig(
            frequencies=[1.0, 3.0, 5.0, 7.0],
            sampling_rate=1000,
            duration=10.0,
            amplitude_range=(0.8, 1.2),
            phase_range=(0, 2*np.pi),
            seed=test_seed
        )
        
        start_time = time.time()
        generator = SignalGenerator(config)
        mixed, targets = generator.generate_complete_dataset()
        elapsed_time = time.time() - start_time
        
        # Should complete in under 5 seconds
        assert elapsed_time < 5.0, f"Signal generation took {elapsed_time:.2f}s, expected < 5s"
        
        # Verify data was generated
        assert len(mixed) == 10000
        assert len(targets) == 4
    
    def test_large_dataset_generation(self, test_seed):
        """Test generation of large dataset."""
        config = SignalConfig(
            frequencies=[1.0, 3.0, 5.0, 7.0],
            sampling_rate=1000,
            duration=60.0,  # 60 seconds = 60k samples
            amplitude_range=(0.8, 1.2),
            phase_range=(0, 2*np.pi),
            seed=test_seed
        )
        
        start_time = time.time()
        generator = SignalGenerator(config)
        dataset = FrequencyExtractionDataset(generator, normalize=True)
        elapsed_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert elapsed_time < 30.0, f"Large dataset generation took {elapsed_time:.2f}s"
        assert len(dataset) == 60000 * 4  # 60k samples * 4 frequencies
    
    def test_parallel_dataset_creation(self, test_seed):
        """Test creating multiple datasets doesn't cause memory issues."""
        datasets = []
        
        for i in range(5):
            config = SignalConfig(
                frequencies=[1.0, 3.0],
                sampling_rate=100,
                duration=1.0,
                amplitude_range=(0.8, 1.2),
                phase_range=(0, 2*np.pi),
                seed=test_seed + i
            )
            generator = SignalGenerator(config)
            dataset = FrequencyExtractionDataset(generator, normalize=True)
            datasets.append(dataset)
        
        # All datasets should be valid
        assert len(datasets) == 5
        for ds in datasets:
            assert len(ds) > 0


@pytest.mark.performance
@pytest.mark.slow
class TestModelPerformance:
    """Test model inference and training performance."""
    
    def test_forward_pass_speed(self, standard_model, device):
        """Test forward pass speed."""
        standard_model.eval()
        standard_model.to(device)
        
        batch_size = 32
        input_size = 5
        num_iterations = 100
        
        inputs = torch.randn(batch_size, input_size).to(device)
        
        # Warm up
        with torch.no_grad():
            _ = standard_model(inputs)
        
        # Time forward passes
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = standard_model(inputs)
        elapsed_time = time.time() - start_time
        
        avg_time = elapsed_time / num_iterations
        throughput = batch_size / avg_time
        
        # Should process at reasonable speed (adjust based on hardware)
        assert throughput > 100, f"Throughput: {throughput:.0f} samples/sec, expected > 100"
    
    def test_large_batch_inference(self, standard_model, device):
        """Test inference with large batches."""
        standard_model.eval()
        standard_model.to(device)
        
        large_batch_size = 1024
        input_size = 5
        
        inputs = torch.randn(large_batch_size, input_size).to(device)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = standard_model(inputs)
        elapsed_time = time.time() - start_time
        
        assert outputs.shape == (large_batch_size, 1)
        assert elapsed_time < 5.0, f"Large batch inference took {elapsed_time:.2f}s"
    
    def test_long_sequence_processing(self, standard_model, device):
        """Test processing long sequences."""
        standard_model.eval()
        standard_model.to(device)
        
        batch_size = 8
        seq_length = 1000
        input_size = 5
        
        inputs = torch.randn(batch_size, seq_length, input_size).to(device)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = standard_model(inputs)
        elapsed_time = time.time() - start_time
        
        assert outputs.shape == (batch_size, seq_length, 1)
        assert elapsed_time < 10.0, f"Long sequence processing took {elapsed_time:.2f}s"
    
    def test_training_epoch_speed(self, minimal_trainer):
        """Test training epoch completion time."""
        start_time = time.time()
        loss = minimal_trainer.train_epoch()
        elapsed_time = time.time() - start_time
        
        # Should complete epoch in reasonable time
        assert elapsed_time < 30.0, f"Training epoch took {elapsed_time:.2f}s, expected < 30s"
        assert isinstance(loss, float)


@pytest.mark.performance
class TestMemoryUsage:
    """Test memory usage characteristics."""
    
    def get_memory_mb(self):
        """Get current process memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def test_dataset_memory_footprint(self, test_seed):
        """Test dataset memory usage."""
        initial_memory = self.get_memory_mb()
        
        config = SignalConfig(
            frequencies=[1.0, 3.0, 5.0, 7.0],
            sampling_rate=1000,
            duration=10.0,
            amplitude_range=(0.8, 1.2),
            phase_range=(0, 2*np.pi),
            seed=test_seed
        )
        
        generator = SignalGenerator(config)
        dataset = FrequencyExtractionDataset(generator, normalize=True)
        
        final_memory = self.get_memory_mb()
        memory_used = final_memory - initial_memory
        
        # Dataset should use reasonable amount of memory (< 100 MB for this size)
        assert memory_used < 100, f"Dataset used {memory_used:.1f}MB, expected < 100MB"
    
    def test_model_memory_footprint(self, standard_model_config, device):
        """Test model memory usage."""
        initial_memory = self.get_memory_mb()
        
        model = create_model(standard_model_config)
        model.to(device)
        
        final_memory = self.get_memory_mb()
        memory_used = final_memory - initial_memory
        
        # Model should use reasonable memory
        param_count = model.count_parameters()
        # Rough estimate: 4 bytes per parameter + overhead
        expected_mb = (param_count * 4) / 1024 / 1024
        
        assert memory_used < expected_mb * 10, f"Model used {memory_used:.1f}MB"
    
    def test_training_memory_stability(self, minimal_trainer):
        """Test that training doesn't cause memory leaks."""
        initial_memory = self.get_memory_mb()
        
        # Train for a few steps
        for _ in range(3):
            minimal_trainer.train_epoch()
        
        final_memory = self.get_memory_mb()
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be bounded
        assert memory_growth < 200, f"Memory grew by {memory_growth:.1f}MB during training"
    
    def test_evaluation_memory_efficiency(self, minimal_model, minimal_dataloader, device):
        """Test evaluation memory efficiency."""
        minimal_model.eval()
        
        initial_memory = self.get_memory_mb()
        
        # Run evaluation multiple times
        for _ in range(5):
            _ = evaluate_model(
                model=minimal_model,
                data_loader=minimal_dataloader,
                device=device
            )
        
        final_memory = self.get_memory_mb()
        memory_growth = final_memory - initial_memory
        
        # Should not accumulate memory
        assert memory_growth < 100, f"Memory grew by {memory_growth:.1f}MB during evaluation"


@pytest.mark.performance
@pytest.mark.slow
class TestScalability:
    """Test system scalability."""
    
    def test_increasing_frequency_count(self, test_seed, device):
        """Test performance with increasing number of frequencies."""
        results = []
        
        for num_freqs in [2, 4, 8]:
            frequencies = list(range(1, num_freqs + 1))
            
            config = SignalConfig(
                frequencies=[float(f) for f in frequencies],
                sampling_rate=100,
                duration=1.0,
                amplitude_range=(0.8, 1.2),
                phase_range=(0, 2*np.pi),
                seed=test_seed
            )
            
            start_time = time.time()
            generator = SignalGenerator(config)
            dataset = FrequencyExtractionDataset(generator, normalize=True)
            elapsed_time = time.time() - start_time
            
            results.append({
                'num_freqs': num_freqs,
                'time': elapsed_time,
                'dataset_size': len(dataset)
            })
        
        # Time should scale roughly linearly
        for i in range(1, len(results)):
            ratio = results[i]['num_freqs'] / results[i-1]['num_freqs']
            time_ratio = results[i]['time'] / results[i-1]['time']
            
            # Allow 3x time growth for 2x frequency increase
            assert time_ratio < ratio * 3
    
    def test_increasing_duration(self, test_seed):
        """Test performance with increasing signal duration."""
        results = []
        
        for duration in [1.0, 5.0, 10.0]:
            config = SignalConfig(
                frequencies=[1.0, 3.0],
                sampling_rate=100,
                duration=duration,
                amplitude_range=(0.8, 1.2),
                phase_range=(0, 2*np.pi),
                seed=test_seed
            )
            
            start_time = time.time()
            generator = SignalGenerator(config)
            _ = generator.generate_complete_dataset()
            elapsed_time = time.time() - start_time
            
            results.append({
                'duration': duration,
                'time': elapsed_time
            })
        
        # Time should scale roughly linearly with duration
        for i in range(1, len(results)):
            duration_ratio = results[i]['duration'] / results[i-1]['duration']
            time_ratio = results[i]['time'] / results[i-1]['time']
            
            # Allow 2x time growth for duration increase
            assert time_ratio < duration_ratio * 2
    
    def test_increasing_model_size(self, device):
        """Test performance with increasing model sizes."""
        results = []
        
        for hidden_size in [32, 64, 128]:
            config = {
                'input_size': 5,
                'hidden_size': hidden_size,
                'num_layers': 2,
                'output_size': 1,
                'dropout': 0.2
            }
            
            model = create_model(config)
            model.to(device)
            model.eval()
            
            inputs = torch.randn(32, 5).to(device)
            
            # Warm up
            with torch.no_grad():
                _ = model(inputs)
            
            # Time inference
            start_time = time.time()
            with torch.no_grad():
                for _ in range(100):
                    _ = model(inputs)
            elapsed_time = time.time() - start_time
            
            results.append({
                'hidden_size': hidden_size,
                'time': elapsed_time,
                'params': model.count_parameters()
            })
        
        # Verify all models work
        assert all(r['time'] > 0 for r in results)


@pytest.mark.performance
class TestStressConditions:
    """Test system under stress conditions."""
    
    @pytest.mark.slow
    def test_continuous_training_stability(self, minimal_trainer):
        """Test stability during extended training."""
        # Train for many epochs
        minimal_trainer.config['epochs'] = 20
        
        history = minimal_trainer.train()
        
        # Should complete without errors
        assert len(history['train_loss']) <= 20
        
        # Losses should be finite
        assert all(np.isfinite(loss) for loss in history['train_loss'])
    
    @pytest.mark.slow
    def test_repeated_evaluation(self, minimal_model, minimal_dataloader, device):
        """Test repeated evaluation doesn't cause issues."""
        minimal_model.eval()
        
        results = []
        for _ in range(20):
            result = evaluate_model(
                model=minimal_model,
                data_loader=minimal_dataloader,
                device=device
            )
            results.append(result['overall']['mse'])
        
        # Results should be consistent
        assert np.std(results) < 0.01, "Evaluation results should be deterministic"
    
    def test_extreme_batch_sizes(self, minimal_dataset):
        """Test with extreme batch sizes."""
        # Very small batch
        small_loader = StatefulDataLoader(minimal_dataset, batch_size=1)
        for batch in small_loader:
            assert batch['input'].shape[0] == 1
            break
        
        # Very large batch
        large_loader = StatefulDataLoader(minimal_dataset, batch_size=10000)
        batch_count = 0
        for batch in large_loader:
            batch_count += 1
            assert batch['input'].shape[0] <= 10000
        
        assert batch_count > 0
    
    @pytest.mark.slow
    def test_many_forward_passes(self, minimal_model, device):
        """Test stability over many forward passes."""
        minimal_model.eval()
        
        for i in range(1000):
            inp = torch.randn(1, 3).to(device)
            with torch.no_grad():
                out = minimal_model(inp)
            
            # Output should remain finite
            assert torch.isfinite(out).all(), f"Non-finite output at iteration {i}"


@pytest.mark.performance
class TestConcurrency:
    """Test concurrent operations."""
    
    def test_concurrent_dataloaders(self, minimal_dataset):
        """Test multiple dataloaders can coexist."""
        loaders = [
            StatefulDataLoader(minimal_dataset, batch_size=16)
            for _ in range(3)
        ]
        
        # All loaders should work
        for loader in loaders:
            batch = next(iter(loader))
            assert 'input' in batch
    
    def test_model_state_isolation(self, minimal_model):
        """Test that model instances are properly isolated."""
        model1 = minimal_model
        
        # Create second model with same architecture
        config = {
            'input_size': 3,
            'hidden_size': 32,
            'num_layers': 1,
            'output_size': 1
        }
        model2 = create_model(config)
        
        # Forward pass through model1
        inp = torch.randn(1, 3)
        _ = model1(inp, reset_state=False)
        state1 = model1.hidden_state
        
        # Forward pass through model2
        _ = model2(inp, reset_state=False)
        state2 = model2.hidden_state
        
        # States should be independent
        assert not torch.allclose(state1, state2, atol=0.01)


@pytest.mark.performance
class TestResourceLimits:
    """Test behavior at resource limits."""
    
    @pytest.mark.edge_case
    def test_maximum_reasonable_dataset(self, test_seed):
        """Test with maximum reasonable dataset size."""
        config = SignalConfig(
            frequencies=[1.0, 3.0, 5.0, 7.0],
            sampling_rate=1000,
            duration=100.0,  # 100 seconds
            amplitude_range=(0.8, 1.2),
            phase_range=(0, 2*np.pi),
            seed=test_seed
        )
        
        start_time = time.time()
        generator = SignalGenerator(config)
        dataset = FrequencyExtractionDataset(generator, normalize=True)
        elapsed_time = time.time() - start_time
        
        # Should complete (may take time but should not crash)
        assert len(dataset) == 100000 * 4
        assert elapsed_time < 120.0, "Should complete in 2 minutes"
    
    @pytest.mark.edge_case
    def test_deep_model(self, device):
        """Test with very deep model."""
        config = {
            'input_size': 5,
            'hidden_size': 64,
            'num_layers': 10,  # Deep network
            'output_size': 1,
            'dropout': 0.2
        }
        
        model = create_model(config)
        model.to(device)
        
        # Should be able to do forward pass
        inp = torch.randn(16, 5).to(device)
        out = model(inp)
        
        assert out.shape == (16, 1)
        assert torch.isfinite(out).all()
    
    @pytest.mark.edge_case
    def test_wide_model(self, device):
        """Test with very wide model."""
        config = {
            'input_size': 5,
            'hidden_size': 1024,  # Very wide
            'num_layers': 2,
            'output_size': 1,
            'dropout': 0.2
        }
        
        model = create_model(config)
        model.to(device)
        
        # Should work
        inp = torch.randn(16, 5).to(device)
        out = model(inp)
        
        assert out.shape == (16, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

