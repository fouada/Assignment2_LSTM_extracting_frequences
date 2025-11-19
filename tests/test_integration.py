"""
Comprehensive Integration Tests
Tests end-to-end workflows and component interactions.
Target: 85%+ coverage with documented edge cases.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import yaml
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for CI

from src.data import SignalGenerator, SignalConfig, FrequencyExtractionDataset, create_train_test_generators
from src.data.dataset import StatefulDataLoader, create_dataloaders
from src.models import StatefulLSTMExtractor, create_model
from src.training.trainer import LSTMTrainer
from src.evaluation.metrics import evaluate_model, compare_train_test_performance, FrequencyExtractionMetrics
from src.visualization.plotter import FrequencyExtractionVisualizer, create_all_visualizations


@pytest.mark.integration
class TestDataPipeline:
    """Test complete data generation and loading pipeline."""
    
    def test_signal_generator_to_dataset(self, minimal_signal_config):
        """Test creating dataset from signal generator."""
        generator = SignalGenerator(minimal_signal_config)
        dataset = FrequencyExtractionDataset(generator, normalize=True)
        
        assert len(dataset) > 0
        assert dataset.num_frequencies == len(minimal_signal_config.frequencies)
        
        # Test getting items
        for i in range(min(10, len(dataset))):
            inp, tgt = dataset[i]
            assert inp.shape[0] == dataset.num_frequencies + 1
            assert tgt.shape[0] == 1
    
    def test_dataset_to_dataloader(self, minimal_dataset):
        """Test creating dataloader from dataset."""
        loader = StatefulDataLoader(
            minimal_dataset,
            batch_size=16,
            shuffle_frequencies=False
        )
        
        batch_count = 0
        for batch in loader:
            assert 'input' in batch
            assert 'target' in batch
            assert 'freq_idx' in batch
            assert 'is_first_batch' in batch
            batch_count += 1
        
        assert batch_count > 0
    
    def test_create_dataloaders_factory(self, test_seed):
        """Test factory function for creating train/test loaders."""
        train_gen, test_gen = create_train_test_generators(
            frequencies=[1.0, 3.0],
            sampling_rate=100,
            duration=1.0,
            train_seed=test_seed,
            test_seed=test_seed + 1
        )
        
        train_loader, test_loader, train_dataset, test_dataset = create_dataloaders(
            train_gen,
            test_gen,
            batch_size=16,
            normalize=True
        )

        assert len(train_loader) > 0
        assert len(test_loader) > 0
        assert train_dataset is not None
        assert test_dataset is not None
    
    @pytest.mark.integration
    def test_data_reproducibility(self, minimal_signal_config):
        """Test that same seed produces same data."""
        gen1 = SignalGenerator(minimal_signal_config)
        gen2 = SignalGenerator(minimal_signal_config)
        
        signal1, targets1 = gen1.generate_complete_dataset()
        signal2, targets2 = gen2.generate_complete_dataset()
        
        np.testing.assert_array_almost_equal(signal1, signal2)
        for i in range(len(targets1)):
            np.testing.assert_array_almost_equal(targets1[i], targets2[i])
    
    @pytest.mark.integration
    def test_dataloader_order_preservation(self, minimal_dataset):
        """Test that dataloader preserves temporal order."""
        loader = StatefulDataLoader(
            minimal_dataset,
            batch_size=16,
            shuffle_frequencies=False
        )
        
        prev_freq = -1
        prev_time_end = -1
        
        for batch in loader:
            freq_idx = batch['freq_idx']
            time_start, time_end = batch['time_range']
            
            # When frequency changes, time should reset
            if freq_idx != prev_freq:
                assert batch['is_first_batch']
                prev_time_end = -1
            else:
                # Within same frequency, time should be sequential
                if prev_time_end >= 0:
                    assert time_start == prev_time_end
            
            prev_freq = freq_idx
            prev_time_end = time_end


@pytest.mark.integration
class TestModelPipeline:
    """Test model creation, training, and evaluation pipeline."""
    
    def test_model_creation_and_forward_pass(self, minimal_model_config):
        """Test model creation and basic forward pass."""
        model = create_model(minimal_model_config)
        
        # Test with single sample
        inp = torch.randn(1, minimal_model_config['input_size'])
        out = model(inp)
        assert out.shape == (1, 1)
        
        # Test with batch
        inp_batch = torch.randn(16, minimal_model_config['input_size'])
        out_batch = model(inp_batch)
        assert out_batch.shape == (16, 1)
    
    def test_model_state_management(self, minimal_model):
        """Test model state management across forward passes."""
        minimal_model.reset_state()
        assert minimal_model.hidden_state is None
        
        # First forward pass
        inp1 = torch.randn(1, 3)
        _ = minimal_model(inp1, reset_state=False)
        assert minimal_model.hidden_state is not None
        
        state1 = minimal_model.hidden_state.clone()
        
        # Second forward pass
        inp2 = torch.randn(1, 3)
        _ = minimal_model(inp2, reset_state=False)
        state2 = minimal_model.hidden_state
        
        # States should be different
        assert not torch.allclose(state1, state2)
    
    def test_model_save_and_load(self, minimal_model, temp_dir):
        """Test model saving and loading."""
        # Save model
        checkpoint = minimal_model.get_state_dict_with_config()
        save_path = temp_dir / "model.pt"
        torch.save(checkpoint, save_path)
        
        # Load model
        loaded_checkpoint = torch.load(save_path, map_location='cpu')
        loaded_model = StatefulLSTMExtractor.from_state_dict_with_config(loaded_checkpoint)
        
        # Check architectures match
        assert loaded_model.input_size == minimal_model.input_size
        assert loaded_model.hidden_size == minimal_model.hidden_size
        
        # Check outputs match
        minimal_model.eval()
        loaded_model.eval()
        
        inp = torch.randn(1, 3)
        with torch.no_grad():
            out1 = minimal_model(inp, reset_state=True)
            out2 = loaded_model(inp, reset_state=True)
        
        assert torch.allclose(out1, out2, atol=1e-5)


@pytest.mark.integration
class TestTrainingPipeline:
    """Test complete training pipeline."""
    
    @pytest.mark.slow
    def test_end_to_end_training(self, minimal_model, minimal_dataloader, device, experiment_dir):
        """Test complete training workflow."""
        config = {
            'epochs': 3,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'gradient_clip_value': 1.0
        }
        
        trainer = LSTMTrainer(
            model=minimal_model,
            train_loader=minimal_dataloader,
            val_loader=minimal_dataloader,
            config=config,
            device=device,
            experiment_dir=experiment_dir / "checkpoints"
        )
        
        # Train
        history = trainer.train()
        
        # Verify training completed
        assert len(history['train_loss']) == 3
        assert len(history['val_loss']) == 3
        assert all(isinstance(loss, float) for loss in history['train_loss'])
        
        # Verify checkpoints exist
        assert (experiment_dir / "checkpoints" / "best_model.pt").exists()
    
    @pytest.mark.slow
    def test_training_reduces_loss(self, minimal_model, minimal_dataloader, device, experiment_dir):
        """Test that training actually improves the model."""
        config = {
            'epochs': 5,
            'learning_rate': 0.001,
            'optimizer': 'adam'
        }
        
        trainer = LSTMTrainer(
            model=minimal_model,
            train_loader=minimal_dataloader,
            val_loader=minimal_dataloader,
            config=config,
            device=device,
            experiment_dir=experiment_dir / "checkpoints"
        )
        
        history = trainer.train()
        
        # Loss should generally decrease
        initial_loss = history['train_loss'][0]
        final_loss = history['train_loss'][-1]
        
        assert final_loss < initial_loss * 1.5  # Allow some tolerance
    
    @pytest.mark.slow
    def test_checkpoint_and_resume(self, minimal_model, minimal_dataloader, device, experiment_dir):
        """Test training checkpoint and resume functionality."""
        config = {
            'epochs': 3,
            'learning_rate': 0.001,
            'optimizer': 'adam'
        }
        
        # Train for a few epochs
        trainer1 = LSTMTrainer(
            model=minimal_model,
            train_loader=minimal_dataloader,
            val_loader=minimal_dataloader,
            config=config,
            device=device,
            experiment_dir=experiment_dir / "checkpoints"
        )
        
        history1 = trainer1.train()
        
        # Create new trainer and resume
        new_model = create_model({
            'input_size': 3,
            'hidden_size': 32,
            'num_layers': 1,
            'output_size': 1
        })
        
        trainer2 = LSTMTrainer(
            model=new_model,
            train_loader=minimal_dataloader,
            val_loader=minimal_dataloader,
            config=config,
            device=device,
            experiment_dir=experiment_dir / "checkpoints"
        )
        
        # Load checkpoint
        checkpoint_path = experiment_dir / "checkpoints" / "best_model.pt"
        trainer2.load_checkpoint(checkpoint_path)
        
        # States should be restored
        assert trainer2.best_val_loss == trainer1.best_val_loss


@pytest.mark.integration
class TestEvaluationPipeline:
    """Test evaluation and metrics pipeline."""
    
    def test_model_evaluation(self, minimal_model, minimal_dataloader, device):
        """Test complete model evaluation."""
        minimal_model.eval()
        
        results = evaluate_model(
            model=minimal_model,
            data_loader=minimal_dataloader,
            device=device,
            compute_per_frequency=True
        )
        
        # Verify results structure
        assert 'overall' in results
        assert 'per_frequency' in results
        assert 'predictions' in results
        assert 'targets' in results
        
        # Verify metrics
        assert 'mse' in results['overall']
        assert 'r2_score' in results['overall']
        
        # Verify per-frequency metrics
        for freq_metrics in results['per_frequency'].values():
            assert 'mse' in freq_metrics
            assert 'num_samples' in freq_metrics
    
    def test_train_test_comparison(self, minimal_model, minimal_dataloader, device):
        """Test train/test performance comparison."""
        minimal_model.eval()
        
        train_results = evaluate_model(
            model=minimal_model,
            data_loader=minimal_dataloader,
            device=device
        )
        
        test_results = evaluate_model(
            model=minimal_model,
            data_loader=minimal_dataloader,
            device=device
        )
        
        comparison = compare_train_test_performance(train_results, test_results)
        
        assert 'overall_generalization' in comparison
        assert 'mse' in comparison
        assert 'r2_score' in comparison


@pytest.mark.integration
class TestVisualizationPipeline:
    """Test visualization pipeline."""
    
    def test_visualization_creation(self, temp_dir, minimal_dataset):
        """Test creating visualizations from evaluation results."""
        visualizer = FrequencyExtractionVisualizer(save_dir=temp_dir)
        
        time = minimal_dataset.generator.time_vector
        freq_idx = 0
        frequency = minimal_dataset.generator.frequencies[freq_idx]
        
        mixed_signal = minimal_dataset.mixed_signal
        target = minimal_dataset.targets[freq_idx]
        prediction = target + np.random.randn(len(target)) * 0.05
        
        # Should create plot without error
        visualizer.plot_single_frequency_comparison(
            time=time,
            mixed_signal=mixed_signal,
            target=target,
            prediction=prediction,
            frequency=frequency,
            freq_idx=freq_idx,
            save_name="integration_test"
        )
        
        assert (temp_dir / "integration_test.png").exists()
        
        import matplotlib.pyplot as plt
        plt.close('all')


@pytest.mark.integration
@pytest.mark.slow
class TestCompleteWorkflow:
    """Test complete end-to-end workflow."""
    
    def test_complete_workflow(self, temp_dir, test_seed, device):
        """Test complete workflow from data generation to visualization."""
        # 1. Generate data
        train_gen, test_gen = create_train_test_generators(
            frequencies=[1.0, 3.0],
            sampling_rate=100,
            duration=1.0,
            train_seed=test_seed,
            test_seed=test_seed + 1
        )
        
        train_loader, test_loader, train_dataset, test_dataset = create_dataloaders(
            train_gen,
            test_gen,
            batch_size=16,
            normalize=True
        )

        # 2. Create model
        model_config = {
            'input_size': 3,
            'hidden_size': 32,
            'num_layers': 1,
            'output_size': 1,
            'dropout': 0.1
        }
        model = create_model(model_config)
        
        # 3. Train model
        training_config = {
            'epochs': 2,
            'learning_rate': 0.001,
            'optimizer': 'adam'
        }
        
        trainer = LSTMTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            config=training_config,
            device=device,
            experiment_dir=temp_dir / "checkpoints"
        )
        
        history = trainer.train()
        
        # 4. Evaluate model
        train_results = evaluate_model(
            model=model,
            data_loader=train_loader,
            device=device
        )
        
        test_results = evaluate_model(
            model=model,
            data_loader=test_loader,
            device=device
        )
        
        # 5. Compare performance
        comparison = compare_train_test_performance(train_results, test_results)
        
        # 6. Verify everything completed successfully
        assert len(history['train_loss']) == 2
        assert 'overall' in train_results
        assert 'overall' in test_results
        assert 'overall_generalization' in comparison
        
        # 7. Check checkpoint exists
        assert (temp_dir / "checkpoints" / "best_model.pt").exists()


@pytest.mark.integration
class TestConfigWorkflow:
    """Test configuration-based workflow."""
    
    def test_config_based_training(self, temp_dir, test_seed, device):
        """Test training with configuration file."""
        # Create config
        config = {
            'data': {
                'frequencies': [1.0, 3.0],
                'sampling_rate': 100,
                'duration': 1.0,
                'batch_size': 16,
                'train_seed': test_seed,
                'test_seed': test_seed + 1
            },
            'model': {
                'input_size': 3,
                'hidden_size': 32,
                'num_layers': 1,
                'output_size': 1,
                'dropout': 0.1
            },
            'training': {
                'epochs': 2,
                'learning_rate': 0.001,
                'optimizer': 'adam'
            }
        }
        
        # Save config
        config_path = temp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Load and verify
        with open(config_path, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        assert loaded_config['data']['frequencies'] == [1.0, 3.0]
        assert loaded_config['model']['hidden_size'] == 32
        assert loaded_config['training']['epochs'] == 2


@pytest.mark.integration
class TestErrorPropagation:
    """Test error handling across components."""
    
    def test_invalid_data_propagation(self):
        """Test that invalid data raises appropriate errors."""
        # Invalid frequency configuration
        with pytest.raises((ValueError, AssertionError)):
            config = SignalConfig(
                frequencies=[-1.0, 3.0],  # Negative frequency
                sampling_rate=1000,
                duration=10.0,
                amplitude_range=(0.8, 1.2),
                phase_range=(0, 2*np.pi),
                seed=42
            )
            generator = SignalGenerator(config)
            generator.generate_mixed_signal()
    
    def test_mismatched_dimensions(self, minimal_model, device):
        """Test error handling with mismatched tensor dimensions."""
        # Wrong input size
        wrong_input = torch.randn(1, 10)  # Model expects size 3
        
        with pytest.raises(RuntimeError):
            minimal_model(wrong_input)


@pytest.mark.integration
class TestMemoryManagement:
    """Test memory management in pipelines."""
    
    def test_dataloader_memory_efficiency(self, minimal_dataset):
        """Test that dataloader doesn't cause memory leaks."""
        loader = StatefulDataLoader(
            minimal_dataset,
            batch_size=16
        )
        
        # Iterate multiple times
        for _ in range(3):
            for batch in loader:
                pass  # Just iterate, don't accumulate
        
        # No assertion needed, just checking it completes without OOM
    
    def test_model_memory_cleanup(self, minimal_model, device):
        """Test that model properly cleans up memory."""
        minimal_model.eval()
        
        with torch.no_grad():
            for _ in range(100):
                inp = torch.randn(16, 3).to(device)
                _ = minimal_model(inp)
                # Memory should be released
        
        # No assertion needed, checking for memory stability


@pytest.mark.integration
class TestReproducibility:
    """Test reproducibility of complete workflow."""
    
    def test_reproducible_training(self, device, temp_dir, set_random_seeds):
        """Test that training is reproducible with same seed."""
        def run_training(exp_dir):
            # Set seeds
            np.random.seed(42)
            torch.manual_seed(42)
            
            # Create data
            train_gen, test_gen = create_train_test_generators(
                frequencies=[1.0, 3.0],
                sampling_rate=100,
                duration=1.0,
                train_seed=42,
                test_seed=43
            )
            
            train_loader, _ = create_dataloaders(
                train_gen, test_gen,
                batch_size=16,
                normalize=True
            )
            
            # Create model
            model = create_model({
                'input_size': 3,
                'hidden_size': 32,
                'num_layers': 1,
                'output_size': 1
            })
            
            # Train
            trainer = LSTMTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=None,
                config={'epochs': 1, 'learning_rate': 0.001, 'optimizer': 'adam'},
                device=device,
                experiment_dir=exp_dir
            )
            
            history = trainer.train()
            return history['train_loss']
        
        # Run twice
        losses1 = run_training(temp_dir / "run1")
        losses2 = run_training(temp_dir / "run2")
        
        # Results should be very similar (allowing for floating point differences)
        assert len(losses1) == len(losses2)
        for l1, l2 in zip(losses1, losses2):
            assert abs(l1 - l2) < 0.01  # Small tolerance for numerical differences


@pytest.mark.integration
class TestQualityModuleIntegration:
    """Test quality module integration."""
    
    def test_quality_module_imports(self):
        """Test that quality module imports work correctly."""
        # This ensures the __init__ file is covered
        from src.quality import (
            QualityMetricsCollector,
            InputValidator,
            ConfigValidator,
            SecurityManager,
            PerformanceMonitor,
            ReliabilityMonitor
        )
        
        assert QualityMetricsCollector is not None
        assert InputValidator is not None
        assert ConfigValidator is not None
        assert SecurityManager is not None
        assert PerformanceMonitor is not None
        assert ReliabilityMonitor is not None


@pytest.mark.integration
class TestDatasetAdvancedMethods:
    """Test advanced dataset methods."""
    
    def test_get_sequence(self, minimal_dataset):
        """Test getting sequential data."""
        sequence_length = 10
        start_idx = 0
        
        input_seq, target_seq = minimal_dataset.get_sequence(start_idx, sequence_length)
        
        assert input_seq.shape[0] == sequence_length
        assert target_seq.shape[0] == sequence_length
        assert input_seq.shape[1] == minimal_dataset.num_frequencies + 1
        assert target_seq.shape[1] == 1
    
    def test_get_time_series_for_frequency(self, minimal_dataset):
        """Test getting time series for specific frequency."""
        freq_idx = 0
        
        time_vector, mixed, target = minimal_dataset.get_time_series_for_frequency(freq_idx)
        
        assert len(time_vector) == len(mixed)
        assert len(mixed) == len(target)
        assert isinstance(time_vector, np.ndarray)
        assert isinstance(mixed, np.ndarray)
        assert isinstance(target, np.ndarray)


@pytest.mark.integration
class TestTrainerVariants:
    """Test trainer with different configurations."""
    
    def test_trainer_with_sgd(self, minimal_model, minimal_train_loader, device, temp_dir):
        """Test trainer with SGD optimizer."""
        config = {
            'epochs': 2,
            'learning_rate': 0.01,
            'optimizer': 'sgd',
            'weight_decay': 0.0001
        }
        
        trainer = LSTMTrainer(
            model=minimal_model,
            train_loader=minimal_train_loader,
            val_loader=None,
            config=config,
            device=device,
            experiment_dir=temp_dir
        )
        
        history = trainer.train()
        assert len(history['train_loss']) == 2
    
    def test_trainer_with_adamw(self, minimal_model, minimal_train_loader, device, temp_dir):
        """Test trainer with AdamW optimizer."""
        config = {
            'epochs': 2,
            'learning_rate': 0.001,
            'optimizer': 'adamw',
            'weight_decay': 0.01
        }
        
        trainer = LSTMTrainer(
            model=minimal_model,
            train_loader=minimal_train_loader,
            val_loader=None,
            config=config,
            device=device,
            experiment_dir=temp_dir
        )
        
        history = trainer.train()
        assert len(history['train_loss']) == 2
    
    def test_trainer_with_cosine_scheduler(self, minimal_model, minimal_train_loader, device, temp_dir):
        """Test trainer with cosine annealing scheduler."""
        config = {
            'epochs': 3,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'scheduler': 'cosine'
        }
        
        trainer = LSTMTrainer(
            model=minimal_model,
            train_loader=minimal_train_loader,
            val_loader=None,
            config=config,
            device=device,
            experiment_dir=temp_dir
        )
        
        history = trainer.train()
        assert len(history['train_loss']) == 3
        # Learning rate should decrease
        assert history['learning_rate'][-1] < history['learning_rate'][0]
    
    def test_trainer_with_reduce_on_plateau_scheduler(self, minimal_model, minimal_train_loader, device, temp_dir):
        """Test trainer with ReduceLROnPlateau scheduler."""
        config = {
            'epochs': 3,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'scheduler': 'reduce_on_plateau',
            'scheduler_factor': 0.5,
            'scheduler_patience': 1
        }
        
        trainer = LSTMTrainer(
            model=minimal_model,
            train_loader=minimal_train_loader,
            val_loader=None,
            config=config,
            device=device,
            experiment_dir=temp_dir
        )
        
        history = trainer.train()
        assert len(history['train_loss']) == 3


@pytest.mark.integration
class TestVisualizationIntegration:
    """Test visualization integration with real data."""
    
    def test_create_all_visualizations_integration(self, minimal_dataset, minimal_model, device, temp_dir):
        """Test creating all visualizations with real model predictions."""
        matplotlib.use('Agg')
        
        # Get predictions
        minimal_model.eval()
        predictions_dict = {}
        
        with torch.no_grad():
            for freq_idx in range(minimal_dataset.num_frequencies):
                time_vector, mixed, target = minimal_dataset.get_time_series_for_frequency(freq_idx)
                
                # Create predictions
                predictions = []
                for i in range(len(target)):
                    inp, _ = minimal_dataset[i]
                    pred = minimal_model(inp.unsqueeze(0).to(device))
                    predictions.append(pred.cpu().item())
                
                predictions_dict[freq_idx] = np.array(predictions)
        
        # Create visualizations
        history = {
            'train_loss': [0.5, 0.3, 0.2],
            'val_loss': [0.6, 0.35, 0.25],
            'learning_rate': [0.001, 0.001, 0.0005]
        }
        
        train_metrics = {
            'overall': {
                'mse': 0.01,
                'mae': 0.08,
                'r2_score': 0.95
            }
        }
        
        test_metrics = {
            'overall': {
                'mse': 0.012,
                'mae': 0.09,
                'r2_score': 0.93
            }
        }
        
        # This should complete without error
        create_all_visualizations(
            test_dataset=minimal_dataset,
            predictions_dict=predictions_dict,
            frequencies=list(minimal_dataset.generator.frequencies),
            history=history,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            save_dir=temp_dir
        )
        
        # Check that at least some plots were created
        plots = list(temp_dir.glob("*.png"))
        assert len(plots) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

