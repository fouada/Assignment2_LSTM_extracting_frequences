"""
Unit tests for LSTM model module.
"""

import pytest
import torch
from src.models import StatefulLSTMExtractor, create_model


class TestStatefulLSTMExtractor:
    """Test StatefulLSTMExtractor class."""
    
    @pytest.fixture
    def model(self):
        """Create a test model."""
        return StatefulLSTMExtractor(
            input_size=5,
            hidden_size=32,
            num_layers=2,
            output_size=1,
            dropout=0.2,
            bidirectional=False
        )
    
    def test_model_initialization(self, model):
        """Test model initializes correctly."""
        assert model.input_size == 5
        assert model.hidden_size == 32
        assert model.num_layers == 2
        assert model.output_size == 1
    
    def test_forward_single_sample(self, model):
        """Test forward pass with single sample."""
        batch_size = 1
        input_tensor = torch.randn(batch_size, 5)
        
        output = model(input_tensor)
        
        assert output.shape == (batch_size, 1)
    
    def test_forward_batch(self, model):
        """Test forward pass with batch."""
        batch_size = 32
        input_tensor = torch.randn(batch_size, 5)
        
        output = model(input_tensor)
        
        assert output.shape == (batch_size, 1)
    
    def test_forward_sequence(self, model):
        """Test forward pass with sequence."""
        batch_size = 16
        seq_len = 10
        input_tensor = torch.randn(batch_size, seq_len, 5)
        
        output = model(input_tensor)
        
        assert output.shape == (batch_size, seq_len, 1)
    
    def test_state_initialization(self, model):
        """Test hidden state initialization."""
        batch_size = 8
        device = torch.device('cpu')
        
        h, c = model.init_hidden(batch_size, device)
        
        assert h.shape == (2, batch_size, 32)  # num_layers=2
        assert c.shape == (2, batch_size, 32)
    
    def test_state_persistence(self, model):
        """Test that state persists between forward passes."""
        input1 = torch.randn(1, 5)
        input2 = torch.randn(1, 5)
        
        # First forward pass
        model.reset_state()
        _ = model(input1, reset_state=False)
        state1 = model.hidden_state.clone()
        
        # Second forward pass (state should change)
        _ = model(input2, reset_state=False)
        state2 = model.hidden_state.clone()
        
        # States should be different
        assert not torch.allclose(state1, state2)
    
    def test_state_reset(self, model):
        """Test state reset functionality."""
        input_tensor = torch.randn(1, 5)
        
        # Forward pass to set state
        _ = model(input_tensor, reset_state=False)
        assert model.hidden_state is not None
        
        # Reset state
        model.reset_state()
        assert model.hidden_state is None
        assert model.cell_state is None
    
    def test_model_in_eval_mode(self, model):
        """Test model in evaluation mode."""
        model.eval()
        input_tensor = torch.randn(16, 5)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output.shape == (16, 1)
    
    def test_parameter_count(self, model):
        """Test parameter counting."""
        param_count = model.count_parameters()
        assert param_count > 0
        assert isinstance(param_count, int)
    
    def test_save_and_load(self, model, tmp_path):
        """Test saving and loading model."""
        # Save model
        checkpoint = model.get_state_dict_with_config()
        save_path = tmp_path / "model.pt"
        torch.save(checkpoint, save_path)
        
        # Load model
        loaded_checkpoint = torch.load(save_path)
        loaded_model = StatefulLSTMExtractor.from_state_dict_with_config(loaded_checkpoint)
        
        # Check architecture matches
        assert loaded_model.input_size == model.input_size
        assert loaded_model.hidden_size == model.hidden_size
        assert loaded_model.num_layers == model.num_layers


class TestModelFactory:
    """Test model factory functions."""
    
    def test_create_model_from_config(self):
        """Test creating model from config dict."""
        config = {
            'input_size': 5,
            'hidden_size': 64,
            'num_layers': 3,
            'output_size': 1,
            'dropout': 0.3,
            'bidirectional': False
        }
        
        model = create_model(config)
        
        assert isinstance(model, StatefulLSTMExtractor)
        assert model.hidden_size == 64
        assert model.num_layers == 3
    
    def test_create_model_defaults(self):
        """Test model creation with default values."""
        config = {}
        model = create_model(config)
        
        assert isinstance(model, StatefulLSTMExtractor)
        # Should use defaults
        assert model.input_size == 5
        assert model.output_size == 1


class TestBidirectionalLSTM:
    """Test bidirectional LSTM variant."""
    
    def test_bidirectional_output_shape(self):
        """Test bidirectional LSTM output shape."""
        model = StatefulLSTMExtractor(
            input_size=5,
            hidden_size=32,
            num_layers=2,
            output_size=1,
            bidirectional=True
        )
        
        input_tensor = torch.randn(8, 10, 5)  # batch=8, seq=10
        output = model(input_tensor)
        
        assert output.shape == (8, 10, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

