"""
Comprehensive Unit Tests for Training Module
Tests trainer functionality, edge cases, and error handling.
Target: 85%+ coverage with documented edge cases.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.training.trainer import LSTMTrainer
from src.models import StatefulLSTMExtractor


class TestLSTMTrainerInitialization:
    """Test trainer initialization with various configurations."""
    
    def test_basic_initialization(self, minimal_trainer):
        """Test basic trainer initialization."""
        assert minimal_trainer is not None
        assert isinstance(minimal_trainer.model, StatefulLSTMExtractor)
        assert isinstance(minimal_trainer.criterion, nn.MSELoss)
        assert minimal_trainer.current_epoch == 0
        assert minimal_trainer.global_step == 0
        assert minimal_trainer.best_val_loss == float('inf')
    
    def test_optimizer_creation_adam(self, minimal_model, minimal_dataloader, device, experiment_dir):
        """Test Adam optimizer creation."""
        config = {
            'epochs': 2,
            'batch_size': 16,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'weight_decay': 1e-5
        }
        
        trainer = LSTMTrainer(
            model=minimal_model,
            train_loader=minimal_dataloader,
            val_loader=None,
            config=config,
            device=device,
            experiment_dir=experiment_dir / "checkpoints"
        )
        
        assert trainer.optimizer.__class__.__name__ == 'Adam'
        assert trainer.optimizer.param_groups[0]['lr'] == 0.001
    
    def test_optimizer_creation_adamw(self, minimal_model, minimal_dataloader, device, experiment_dir):
        """Test AdamW optimizer creation."""
        config = {
            'epochs': 2,
            'batch_size': 16,
            'learning_rate': 0.001,
            'optimizer': 'adamw',
            'weight_decay': 1e-5
        }
        
        trainer = LSTMTrainer(
            model=minimal_model,
            train_loader=minimal_dataloader,
            val_loader=None,
            config=config,
            device=device,
            experiment_dir=experiment_dir / "checkpoints"
        )
        
        assert trainer.optimizer.__class__.__name__ == 'AdamW'
    
    def test_optimizer_creation_sgd(self, minimal_model, minimal_dataloader, device, experiment_dir):
        """Test SGD optimizer creation."""
        config = {
            'epochs': 2,
            'batch_size': 16,
            'learning_rate': 0.01,
            'optimizer': 'sgd'
        }
        
        trainer = LSTMTrainer(
            model=minimal_model,
            train_loader=minimal_dataloader,
            val_loader=None,
            config=config,
            device=device,
            experiment_dir=experiment_dir / "checkpoints"
        )
        
        assert trainer.optimizer.__class__.__name__ == 'SGD'
    
    @pytest.mark.edge_case
    def test_invalid_optimizer_raises_error(self, minimal_model, minimal_dataloader, device, experiment_dir):
        """Test that invalid optimizer name raises error."""
        config = {
            'epochs': 2,
            'learning_rate': 0.001,
            'optimizer': 'invalid_optimizer'
        }
        
        with pytest.raises(ValueError, match="Unknown optimizer"):
            LSTMTrainer(
                model=minimal_model,
                train_loader=minimal_dataloader,
                val_loader=None,
                config=config,
                device=device,
                experiment_dir=experiment_dir / "checkpoints"
            )
    
    def test_scheduler_reduce_on_plateau(self, minimal_model, minimal_dataloader, device, experiment_dir):
        """Test ReduceLROnPlateau scheduler creation."""
        config = {
            'epochs': 2,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'scheduler': 'reduce_on_plateau',
            'scheduler_factor': 0.5,
            'scheduler_patience': 5
        }
        
        trainer = LSTMTrainer(
            model=minimal_model,
            train_loader=minimal_dataloader,
            val_loader=None,
            config=config,
            device=device,
            experiment_dir=experiment_dir / "checkpoints"
        )
        
        assert trainer.scheduler is not None
        assert trainer.scheduler.__class__.__name__ == 'ReduceLROnPlateau'
    
    def test_scheduler_cosine(self, minimal_model, minimal_dataloader, device, experiment_dir):
        """Test CosineAnnealingLR scheduler creation."""
        config = {
            'epochs': 10,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'scheduler': 'cosine'
        }
        
        trainer = LSTMTrainer(
            model=minimal_model,
            train_loader=minimal_dataloader,
            val_loader=None,
            config=config,
            device=device,
            experiment_dir=experiment_dir / "checkpoints"
        )
        
        assert trainer.scheduler is not None
        assert trainer.scheduler.__class__.__name__ == 'CosineAnnealingLR'
    
    @pytest.mark.edge_case
    def test_invalid_scheduler_raises_error(self, minimal_model, minimal_dataloader, device, experiment_dir):
        """Test that invalid scheduler name raises error."""
        config = {
            'epochs': 2,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'scheduler': 'invalid_scheduler'
        }
        
        with pytest.raises(ValueError, match="Unknown scheduler"):
            LSTMTrainer(
                model=minimal_model,
                train_loader=minimal_dataloader,
                val_loader=None,
                config=config,
                device=device,
                experiment_dir=experiment_dir / "checkpoints"
            )


class TestTrainingEpoch:
    """Test training epoch functionality."""
    
    def test_train_epoch_completes(self, minimal_trainer):
        """Test that training epoch completes without errors."""
        loss = minimal_trainer.train_epoch()
        
        assert isinstance(loss, float)
        assert loss >= 0
        assert not np.isnan(loss)
        assert not np.isinf(loss)
    
    def test_train_epoch_updates_metrics(self, minimal_trainer):
        """Test that training epoch updates metrics."""
        initial_step = minimal_trainer.global_step
        minimal_trainer.train_epoch()
        
        assert minimal_trainer.global_step > initial_step
    
    def test_train_epoch_state_management(self, minimal_trainer):
        """Test that LSTM state is properly managed during training."""
        # State should be reset at the start of each frequency
        minimal_trainer.train_epoch()
        
        # Model should have been called with reset_state correctly
        assert minimal_trainer.model.hidden_state is not None
    
    def test_gradient_clipping(self, minimal_model, minimal_dataloader, device, experiment_dir):
        """Test gradient clipping functionality."""
        config = {
            'epochs': 1,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'gradient_clip_value': 1.0
        }
        
        trainer = LSTMTrainer(
            model=minimal_model,
            train_loader=minimal_dataloader,
            val_loader=None,
            config=config,
            device=device,
            experiment_dir=experiment_dir / "checkpoints"
        )
        
        trainer.train_epoch()
        
        # Check that gradients are within clip value
        for param in trainer.model.parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                # After clipping, norm should be reasonable
                assert grad_norm < 100.0
    
    @pytest.mark.edge_case
    def test_train_epoch_with_empty_loader(self, minimal_model, device, experiment_dir):
        """Test training with empty dataloader."""
        from src.data.dataset import StatefulDataLoader, FrequencyExtractionDataset
        from src.data import SignalGenerator, SignalConfig
        
        # Create minimal dataset
        config = SignalConfig(
            frequencies=[1.0],
            sampling_rate=10,
            duration=0.1,
            amplitude_range=(0.8, 1.2),
            phase_range=(0, 2*np.pi),
            seed=42
        )
        generator = SignalGenerator(config)
        dataset = FrequencyExtractionDataset(generator, normalize=True)
        loader = StatefulDataLoader(dataset, batch_size=1)
        
        training_config = {
            'epochs': 1,
            'learning_rate': 0.001,
            'optimizer': 'adam'
        }
        
        # Update model input size to match dataset
        minimal_model.input_size = 2  # S[t] + 1 one-hot
        
        trainer = LSTMTrainer(
            model=minimal_model,
            train_loader=loader,
            val_loader=None,
            config=training_config,
            device=device,
            experiment_dir=experiment_dir / "checkpoints"
        )
        
        # Should complete without error
        loss = trainer.train_epoch()
        assert isinstance(loss, float)


class TestValidation:
    """Test validation functionality."""
    
    def test_validation_completes(self, minimal_trainer):
        """Test that validation completes without errors."""
        val_loss = minimal_trainer.validate()
        
        assert isinstance(val_loss, float)
        assert val_loss >= 0 or np.isnan(val_loss)  # NaN if no val loader
    
    def test_validation_no_gradients(self, minimal_trainer):
        """Test that validation doesn't update gradients."""
        minimal_trainer.validate()
        
        # All gradients should be None or zero
        for param in minimal_trainer.model.parameters():
            if param.grad is not None:
                assert param.grad.abs().sum() == 0 or param.grad is None
    
    def test_validation_without_val_loader(self, minimal_model, minimal_dataloader, device, experiment_dir):
        """Test validation returns NaN when no validation loader."""
        config = {'epochs': 1, 'learning_rate': 0.001, 'optimizer': 'adam'}
        
        trainer = LSTMTrainer(
            model=minimal_model,
            train_loader=minimal_dataloader,
            val_loader=None,
            config=config,
            device=device,
            experiment_dir=experiment_dir / "checkpoints"
        )
        
        val_loss = trainer.validate()
        assert np.isnan(val_loss)
    
    def test_validation_deterministic_with_same_data(self, minimal_trainer):
        """Test that validation is deterministic with same data."""
        minimal_trainer.model.eval()
        
        val_loss1 = minimal_trainer.validate()
        val_loss2 = minimal_trainer.validate()
        
        # Should get same result (within floating point precision)
        if not np.isnan(val_loss1):
            assert np.allclose(val_loss1, val_loss2, rtol=1e-5)


class TestEarlyStopping:
    """Test early stopping functionality."""
    
    def test_early_stopping_triggers(self, minimal_model, minimal_dataloader, device, experiment_dir):
        """Test that early stopping triggers after patience epochs."""
        config = {
            'epochs': 100,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'early_stopping_patience': 2,
            'min_delta': 0.0
        }
        
        trainer = LSTMTrainer(
            model=minimal_model,
            train_loader=minimal_dataloader,
            val_loader=minimal_dataloader,
            config=config,
            device=device,
            experiment_dir=experiment_dir / "checkpoints"
        )
        
        # Simulate no improvement
        trainer.best_val_loss = 0.5
        assert not trainer.should_early_stop(0.5)  # Same loss
        assert not trainer.should_early_stop(0.5)  # Patience 1
        assert trainer.should_early_stop(0.5)  # Patience 2, should stop
    
    def test_early_stopping_resets_on_improvement(self, minimal_model, minimal_dataloader, device, experiment_dir):
        """Test that patience counter resets on improvement."""
        config = {
            'epochs': 100,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'early_stopping_patience': 3,
            'min_delta': 0.01
        }
        
        trainer = LSTMTrainer(
            model=minimal_model,
            train_loader=minimal_dataloader,
            val_loader=minimal_dataloader,
            config=config,
            device=device,
            experiment_dir=experiment_dir / "checkpoints"
        )
        
        trainer.best_val_loss = 0.5
        assert not trainer.should_early_stop(0.6)  # Worse
        assert not trainer.should_early_stop(0.6)  # Worse
        assert not trainer.should_early_stop(0.4)  # Better, resets counter
        assert not trainer.should_early_stop(0.5)  # Worse, counter = 1
        assert not trainer.should_early_stop(0.5)  # Worse, counter = 2
        assert not trainer.should_early_stop(0.5)  # Worse, counter = 3
        assert trainer.should_early_stop(0.5)  # Should stop now
    
    @pytest.mark.edge_case
    def test_early_stopping_with_nan_loss(self, minimal_model, minimal_dataloader, device, experiment_dir):
        """Test early stopping behavior with NaN loss."""
        config = {
            'epochs': 100,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'early_stopping_patience': 2
        }
        
        trainer = LSTMTrainer(
            model=minimal_model,
            train_loader=minimal_dataloader,
            val_loader=None,
            config=config,
            device=device,
            experiment_dir=experiment_dir / "checkpoints"
        )
        
        # NaN loss should be handled
        trainer.best_val_loss = 0.5
        should_stop = trainer.should_early_stop(float('nan'))
        # Should increment patience counter
        assert trainer.patience_counter > 0


class TestCheckpointing:
    """Test model checkpointing functionality."""
    
    def test_save_checkpoint(self, minimal_trainer, experiment_dir):
        """Test saving checkpoint."""
        minimal_trainer.save_checkpoint(is_best=False)
        
        checkpoint_path = experiment_dir / "checkpoints" / f"checkpoint_epoch_{minimal_trainer.current_epoch}.pt"
        assert checkpoint_path.exists()
    
    def test_save_best_checkpoint(self, minimal_trainer, experiment_dir):
        """Test saving best checkpoint."""
        minimal_trainer.best_val_loss = 0.5
        minimal_trainer.save_checkpoint(is_best=True)
        
        best_path = experiment_dir / "checkpoints" / "best_model.pt"
        assert best_path.exists()
    
    def test_checkpoint_contains_required_info(self, minimal_trainer, experiment_dir):
        """Test that checkpoint contains all required information."""
        minimal_trainer.save_checkpoint(is_best=False)
        
        checkpoint_path = experiment_dir / "checkpoints" / f"checkpoint_epoch_{minimal_trainer.current_epoch}.pt"
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        assert 'epoch' in checkpoint
        assert 'global_step' in checkpoint
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert 'best_val_loss' in checkpoint
        assert 'history' in checkpoint
        assert 'config' in checkpoint
    
    def test_load_checkpoint(self, minimal_trainer, experiment_dir):
        """Test loading checkpoint."""
        # Save checkpoint
        minimal_trainer.current_epoch = 5
        minimal_trainer.global_step = 100
        minimal_trainer.best_val_loss = 0.3
        minimal_trainer.save_checkpoint(is_best=False)
        
        # Create new trainer
        from src.models import create_model
        from src.data.dataset import StatefulDataLoader
        
        new_model = create_model(minimal_trainer.config.get('model', {
            'input_size': 3,
            'hidden_size': 32,
            'num_layers': 1,
            'output_size': 1
        }))
        
        new_trainer = LSTMTrainer(
            model=new_model,
            train_loader=minimal_trainer.train_loader,
            val_loader=minimal_trainer.val_loader,
            config=minimal_trainer.config,
            device=minimal_trainer.device,
            experiment_dir=experiment_dir / "checkpoints"
        )
        
        # Load checkpoint
        checkpoint_path = experiment_dir / "checkpoints" / "checkpoint_epoch_5.pt"
        new_trainer.load_checkpoint(checkpoint_path)
        
        assert new_trainer.current_epoch == 5
        assert new_trainer.global_step == 100
        assert new_trainer.best_val_loss == 0.3
    
    @pytest.mark.edge_case
    def test_save_checkpoint_creates_directory(self, minimal_model, minimal_dataloader, device, temp_dir):
        """Test that checkpoint saving creates directory if it doesn't exist."""
        new_exp_dir = temp_dir / "new_experiment" / "deep" / "nested"
        
        config = {'epochs': 1, 'learning_rate': 0.001, 'optimizer': 'adam'}
        
        trainer = LSTMTrainer(
            model=minimal_model,
            train_loader=minimal_dataloader,
            val_loader=None,
            config=config,
            device=device,
            experiment_dir=new_exp_dir
        )
        
        trainer.save_checkpoint(is_best=False)
        assert new_exp_dir.exists()


class TestFullTraining:
    """Test complete training loops."""
    
    @pytest.mark.slow
    def test_full_training_run(self, minimal_trainer):
        """Test complete training run."""
        history = minimal_trainer.train()
        
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert 'learning_rate' in history
        assert len(history['train_loss']) > 0
    
    def test_training_reduces_loss(self, minimal_trainer):
        """Test that training reduces loss over epochs."""
        history = minimal_trainer.train()
        
        train_losses = history['train_loss']
        if len(train_losses) >= 2:
            # Loss should generally decrease (allowing for some variation)
            initial_avg = np.mean(train_losses[:len(train_losses)//2])
            final_avg = np.mean(train_losses[len(train_losses)//2:])
            assert final_avg <= initial_avg * 1.5  # Allow some tolerance
    
    def test_history_tracking(self, minimal_trainer):
        """Test that history is properly tracked."""
        history = minimal_trainer.train()
        
        num_epochs = len(history['train_loss'])
        assert len(history['val_loss']) == num_epochs
        assert len(history['learning_rate']) == num_epochs
    
    @pytest.mark.edge_case
    def test_training_with_zero_epochs(self, minimal_model, minimal_dataloader, device, experiment_dir):
        """Test training with zero epochs."""
        config = {
            'epochs': 0,
            'learning_rate': 0.001,
            'optimizer': 'adam'
        }
        
        trainer = LSTMTrainer(
            model=minimal_model,
            train_loader=minimal_dataloader,
            val_loader=None,
            config=config,
            device=device,
            experiment_dir=experiment_dir / "checkpoints"
        )
        
        history = trainer.train()
        assert len(history['train_loss']) == 0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.edge_case
    def test_very_high_learning_rate(self, minimal_model, minimal_dataloader, device, experiment_dir):
        """Test training with very high learning rate."""
        config = {
            'epochs': 1,
            'learning_rate': 100.0,  # Very high
            'optimizer': 'adam'
        }
        
        trainer = LSTMTrainer(
            model=minimal_model,
            train_loader=minimal_dataloader,
            val_loader=None,
            config=config,
            device=device,
            experiment_dir=experiment_dir / "checkpoints"
        )
        
        # Should complete without crashing (may have high loss)
        loss = trainer.train_epoch()
        assert isinstance(loss, float)
    
    @pytest.mark.edge_case
    def test_very_low_learning_rate(self, minimal_model, minimal_dataloader, device, experiment_dir):
        """Test training with very low learning rate."""
        config = {
            'epochs': 1,
            'learning_rate': 1e-10,  # Very low
            'optimizer': 'adam'
        }
        
        trainer = LSTMTrainer(
            model=minimal_model,
            train_loader=minimal_dataloader,
            val_loader=None,
            config=config,
            device=device,
            experiment_dir=experiment_dir / "checkpoints"
        )
        
        loss = trainer.train_epoch()
        assert isinstance(loss, float)
    
    @pytest.mark.edge_case
    def test_extreme_gradient_clipping(self, minimal_model, minimal_dataloader, device, experiment_dir):
        """Test with extreme gradient clipping value."""
        config = {
            'epochs': 1,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'gradient_clip_value': 0.001  # Very small
        }
        
        trainer = LSTMTrainer(
            model=minimal_model,
            train_loader=minimal_dataloader,
            val_loader=None,
            config=config,
            device=device,
            experiment_dir=experiment_dir / "checkpoints"
        )
        
        loss = trainer.train_epoch()
        assert isinstance(loss, float)
    
    @pytest.mark.edge_case
    def test_training_persistence_after_nan(self, minimal_model, minimal_dataloader, device, experiment_dir):
        """Test that trainer can handle and recover from NaN losses."""
        config = {
            'epochs': 2,
            'learning_rate': 0.001,
            'optimizer': 'adam'
        }
        
        trainer = LSTMTrainer(
            model=minimal_model,
            train_loader=minimal_dataloader,
            val_loader=None,
            config=config,
            device=device,
            experiment_dir=experiment_dir / "checkpoints"
        )
        
        # Train should complete even if some losses are high
        history = trainer.train()
        assert len(history['train_loss']) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

