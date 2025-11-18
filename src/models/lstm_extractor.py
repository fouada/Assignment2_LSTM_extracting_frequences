"""
LSTM Frequency Extractor Model
Professional stateful LSTM implementation for frequency extraction.

Author: Professional ML Engineering Team
Date: 2025
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class StatefulLSTMExtractor(nn.Module):
    """
    Stateful LSTM for frequency extraction from mixed signals.
    
    Key Features:
    - Maintains hidden and cell states between forward passes
    - Supports both L=1 (single sample) and L>1 (sequences)
    - Professional architecture with dropout and normalization
    - Designed for conditional regression task
    """
    
    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        """
        Initialize the LSTM extractor.
        
        Args:
            input_size: Input feature size (S[t] + one-hot = 5)
            hidden_size: Hidden state dimension
            num_layers: Number of LSTM layers
            output_size: Output size (1 for single frequency extraction)
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super(StatefulLSTMExtractor, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Input normalization
        self.input_norm = nn.LayerNorm(input_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output layers with residual connection capability
        self.output_norm = nn.LayerNorm(hidden_size * self.num_directions)
        self.fc1 = nn.Linear(hidden_size * self.num_directions, hidden_size // 2)
        self.activation = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
        # Initialize weights
        self._init_weights()
        
        # State storage
        self.hidden_state: Optional[torch.Tensor] = None
        self.cell_state: Optional[torch.Tensor] = None
        
        logger.info(f"StatefulLSTMExtractor initialized:")
        logger.info(f"  Input size: {input_size}")
        logger.info(f"  Hidden size: {hidden_size}")
        logger.info(f"  Num layers: {num_layers}")
        logger.info(f"  Bidirectional: {bidirectional}")
        logger.info(f"  Total parameters: {self.count_parameters():,}")
    
    def _init_weights(self):
        """Initialize weights using Xavier/He initialization."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
            elif 'weight' in name and 'norm' not in name:
                nn.init.kaiming_normal_(param.data)
    
    def init_hidden(
        self, 
        batch_size: int, 
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden and cell states.
        
        Args:
            batch_size: Batch size
            device: Device to create tensors on
            
        Returns:
            Tuple of (hidden_state, cell_state)
        """
        hidden = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size,
            device=device
        )
        
        cell = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size,
            device=device
        )
        
        return hidden, cell
    
    def reset_state(self):
        """Reset the internal state (call at start of new sequence)."""
        self.hidden_state = None
        self.cell_state = None
        logger.debug("LSTM state reset")
    
    def detach_state(self):
        """Detach state from computation graph (for TBPTT)."""
        if self.hidden_state is not None:
            self.hidden_state = self.hidden_state.detach()
        if self.cell_state is not None:
            self.cell_state = self.cell_state.detach()
    
    def forward(
        self,
        x: torch.Tensor,
        reset_state: bool = False,
        return_state: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
               or (batch_size, input_size) for single sample
            reset_state: Whether to reset hidden state before forward pass
            return_state: Whether to return the final hidden state
            
        Returns:
            Output tensor of shape (batch_size, seq_len, output_size)
            If return_state=True, returns (output, hidden_state, cell_state)
        """
        # Handle single sample case
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
            single_sample = True
        else:
            single_sample = False
        
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Normalize input
        x = self.input_norm(x)
        
        # Initialize or reuse state
        if reset_state or self.hidden_state is None:
            self.hidden_state, self.cell_state = self.init_hidden(batch_size, device)
        else:
            # Ensure state batch size matches current batch
            if self.hidden_state.size(1) != batch_size:
                self.hidden_state, self.cell_state = self.init_hidden(batch_size, device)
        
        # LSTM forward pass
        lstm_out, (self.hidden_state, self.cell_state) = self.lstm(
            x, 
            (self.hidden_state, self.cell_state)
        )
        
        # Normalize LSTM output
        lstm_out = self.output_norm(lstm_out)
        
        # Pass through output layers
        out = self.fc1(lstm_out)
        out = self.activation(out)
        out = self.dropout_layer(out)
        out = self.fc2(out)
        
        # Remove sequence dimension if input was single sample
        if single_sample:
            out = out.squeeze(1)
        
        if return_state:
            return out, self.hidden_state, self.cell_state
        
        return out
    
    def predict_sequence(
        self,
        x: torch.Tensor,
        reset_state: bool = True
    ) -> torch.Tensor:
        """
        Predict on a sequence without updating internal state.
        
        Useful for evaluation.
        
        Args:
            x: Input sequence
            reset_state: Whether to reset state before prediction
            
        Returns:
            Output predictions
        """
        self.eval()
        with torch.no_grad():
            # Save current state
            old_h = self.hidden_state
            old_c = self.cell_state
            
            # Forward pass
            if reset_state:
                self.reset_state()
            
            output = self.forward(x, reset_state=False)
            
            # Restore state
            self.hidden_state = old_h
            self.cell_state = old_c
        
        return output
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_state_dict_with_config(self) -> Dict:
        """
        Get model state dict along with architecture config.
        
        Useful for saving/loading models.
        """
        return {
            'state_dict': self.state_dict(),
            'config': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'output_size': self.output_size,
                'dropout': self.dropout,
                'bidirectional': self.bidirectional
            }
        }
    
    @classmethod
    def from_state_dict_with_config(cls, checkpoint: Dict) -> 'StatefulLSTMExtractor':
        """
        Create model from checkpoint with config.
        
        Args:
            checkpoint: Dictionary with 'state_dict' and 'config'
            
        Returns:
            Initialized model with loaded weights
        """
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        return model


def create_model(config: Dict) -> StatefulLSTMExtractor:
    """
    Factory function to create model from config dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized StatefulLSTMExtractor
    """
    model = StatefulLSTMExtractor(
        input_size=config.get('input_size', 5),
        hidden_size=config.get('hidden_size', 128),
        num_layers=config.get('num_layers', 2),
        output_size=config.get('output_size', 1),
        dropout=config.get('dropout', 0.2),
        bidirectional=config.get('bidirectional', False)
    )
    
    logger.info("Model created from config")
    return model

