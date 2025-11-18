"""
Attention-Based LSTM for Explainable Frequency Extraction
INNOVATION: Multi-head attention mechanism for interpretability

Author: Professional ML Engineering Team
Date: November 2025
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Dict
import logging
from .lstm_extractor import StatefulLSTMExtractor

logger = logging.getLogger(__name__)


class AttentionLSTMExtractor(StatefulLSTMExtractor):
    """
    LSTM with Multi-Head Attention for Explainable Frequency Extraction.
    
    INNOVATION HIGHLIGHTS:
    - Shows which time steps are most important for prediction
    - Provides interpretability through attention weights
    - Enables visualization of "what the model is thinking"
    - First attention-based approach for this specific problem
    
    Key Features:
    - Multi-head self-attention mechanism
    - Attention weight tracking for visualization
    - Compatible with existing stateful LSTM interface
    - Minimal overhead compared to base model
    """
    
    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
        attention_heads: int = 4,
        attention_window: int = 50,
        track_attention: bool = True
    ):
        """
        Initialize Attention-Based LSTM Extractor.
        
        Args:
            attention_heads: Number of attention heads
            attention_window: Look-back window for attention (memory efficient)
            track_attention: Whether to store attention weights for visualization
        """
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        self.attention_heads = attention_heads
        self.attention_window = attention_window
        self.track_attention = track_attention
        
        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * self.num_directions,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Attention normalization
        self.attention_norm = nn.LayerNorm(hidden_size * self.num_directions)
        
        # Positional encoding for attention
        self.positional_encoding = PositionalEncoding(
            d_model=hidden_size * self.num_directions,
            max_len=attention_window
        )
        
        # Updated output layers (input size changed due to attention)
        self.fc1 = nn.Linear(
            hidden_size * self.num_directions * 2,  # LSTM + attention
            hidden_size
        )
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        
        # Attention weight storage
        self.attention_weights: List[torch.Tensor] = []
        self.attention_history: List[Dict] = []
        
        # LSTM output buffer for attention computation
        self.lstm_output_buffer: List[torch.Tensor] = []
        self.max_buffer_size = attention_window
        
        logger.info(f"AttentionLSTMExtractor initialized:")
        logger.info(f"  Attention heads: {attention_heads}")
        logger.info(f"  Attention window: {attention_window}")
        logger.info(f"  Track attention: {track_attention}")
        logger.info(f"  Total parameters: {self.count_parameters():,}")
    
    def reset_state(self):
        """Reset both LSTM state and attention buffer."""
        super().reset_state()
        self.lstm_output_buffer = []
        if self.track_attention:
            self.attention_weights = []
            self.attention_history = []
        logger.debug("LSTM state and attention buffer reset")
    
    def forward(
        self,
        x: torch.Tensor,
        reset_state: bool = False,
        return_state: bool = False,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with attention mechanism.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
            reset_state: Whether to reset hidden state
            return_state: Whether to return hidden states
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor and optionally attention weights
        """
        # Handle single sample case
        if x.dim() == 2:
            x = x.unsqueeze(1)
            single_sample = True
        else:
            single_sample = False
        
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Reset if needed
        if reset_state:
            self.reset_state()
        
        # Normalize input
        x = self.input_norm(x)
        
        # Initialize or reuse state
        if self.hidden_state is None:
            self.hidden_state, self.cell_state = self.init_hidden(batch_size, device)
        else:
            if self.hidden_state.size(1) != batch_size:
                self.hidden_state, self.cell_state = self.init_hidden(batch_size, device)
        
        # LSTM forward pass
        lstm_out, (self.hidden_state, self.cell_state) = self.lstm(
            x,
            (self.hidden_state, self.cell_state)
        )
        
        # Normalize LSTM output
        lstm_out = self.output_norm(lstm_out)
        
        # Add to buffer for attention
        self.lstm_output_buffer.append(lstm_out.detach())
        if len(self.lstm_output_buffer) > self.max_buffer_size:
            self.lstm_output_buffer.pop(0)
        
        # Prepare sequence for attention (use buffer)
        if len(self.lstm_output_buffer) > 1:
            # Concatenate buffered outputs
            attention_input = torch.cat(self.lstm_output_buffer, dim=1)
            
            # Add positional encoding
            attention_input = self.positional_encoding(attention_input)
            
            # Apply multi-head attention
            attended_out, attention_weights = self.attention(
                lstm_out,  # query: current output
                attention_input,  # key: historical outputs
                attention_input,  # value: historical outputs
                need_weights=True,
                average_attn_weights=False  # Get per-head weights
            )
            
            # Store attention weights if tracking
            if self.track_attention:
                self.attention_weights.append(attention_weights.detach().cpu())
                self.attention_history.append({
                    'timestamp': len(self.attention_weights),
                    'buffer_size': len(self.lstm_output_buffer),
                    'weights_shape': attention_weights.shape
                })
            
            # Normalize attended output
            attended_out = self.attention_norm(attended_out)
            
            # Combine LSTM output and attended output
            combined = torch.cat([lstm_out, attended_out], dim=-1)
        else:
            # Not enough history for attention yet
            combined = torch.cat([lstm_out, lstm_out], dim=-1)
            attention_weights = None
        
        # Pass through output layers
        out = self.fc1(combined)
        out = self.activation(out)
        out = self.dropout_layer(out)
        out = self.fc2(out)
        out = self.activation(out)
        out = self.dropout_layer(out)
        out = self.fc3(out)
        
        # Remove sequence dimension if input was single sample
        if single_sample:
            out = out.squeeze(1)
        
        # Return based on flags
        if return_attention and return_state:
            return out, self.hidden_state, self.cell_state, attention_weights
        elif return_attention:
            return out, attention_weights
        elif return_state:
            return out, self.hidden_state, self.cell_state
        
        return out
    
    def get_attention_weights(
        self,
        as_numpy: bool = True
    ) -> List:
        """
        Get all stored attention weights.
        
        Args:
            as_numpy: Convert to numpy arrays
            
        Returns:
            List of attention weight tensors/arrays
        """
        if as_numpy:
            return [w.numpy() for w in self.attention_weights]
        return self.attention_weights
    
    def get_attention_statistics(self) -> Dict:
        """
        Get statistics about attention patterns.
        
        Returns:
            Dictionary with attention statistics
        """
        if not self.attention_weights:
            return {
                'total_steps': 0,
                'message': 'No attention weights recorded'
            }
        
        # Concatenate all weights
        all_weights = torch.cat(self.attention_weights, dim=0)
        
        return {
            'total_steps': len(self.attention_weights),
            'attention_heads': self.attention_heads,
            'mean_attention': all_weights.mean().item(),
            'std_attention': all_weights.std().item(),
            'max_attention': all_weights.max().item(),
            'min_attention': all_weights.min().item(),
            'attention_entropy': self._compute_attention_entropy(all_weights)
        }
    
    def _compute_attention_entropy(self, weights: torch.Tensor) -> float:
        """Compute entropy of attention distribution (lower = more focused)."""
        # Average across heads and batches
        avg_weights = weights.mean(dim=(0, 1))
        # Normalize to probability distribution
        probs = avg_weights / avg_weights.sum()
        # Compute entropy
        entropy = -(probs * torch.log(probs + 1e-10)).sum()
        return entropy.item()
    
    def visualize_attention_heatmap(
        self,
        save_path: Optional[str] = None,
        frequency_idx: Optional[int] = None
    ):
        """
        Create attention heatmap visualization.
        
        Args:
            save_path: Path to save figure
            frequency_idx: Which frequency to visualize
        """
        if not self.attention_weights:
            logger.warning("No attention weights to visualize")
            return
        
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Get weights as numpy
        weights_np = [w.numpy() for w in self.attention_weights]
        
        # Average across heads for simplicity
        avg_weights = [w.mean(axis=1) for w in weights_np]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Stack weights into 2D array
        heatmap_data = np.concatenate(avg_weights, axis=0)
        
        im = ax.imshow(
            heatmap_data[:min(500, len(heatmap_data))].T,
            aspect='auto',
            cmap='viridis',
            interpolation='nearest'
        )
        
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Attended Position', fontsize=12)
        ax.set_title(
            f'Attention Heatmap {"(Frequency " + str(frequency_idx) + ")" if frequency_idx is not None else ""}',
            fontsize=14
        )
        
        plt.colorbar(im, ax=ax, label='Attention Weight')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Attention heatmap saved to {save_path}")
        
        return fig


class PositionalEncoding(nn.Module):
    """
    Positional encoding for attention mechanism.
    Helps attention distinguish between different time positions.
    """
    
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:, :x.size(1)]


def create_attention_model(config: Dict) -> AttentionLSTMExtractor:
    """
    Factory function to create attention model from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized AttentionLSTMExtractor
    """
    model = AttentionLSTMExtractor(
        input_size=config.get('input_size', 5),
        hidden_size=config.get('hidden_size', 128),
        num_layers=config.get('num_layers', 2),
        output_size=config.get('output_size', 1),
        dropout=config.get('dropout', 0.2),
        bidirectional=config.get('bidirectional', False),
        attention_heads=config.get('attention_heads', 4),
        attention_window=config.get('attention_window', 50),
        track_attention=config.get('track_attention', True)
    )
    
    logger.info("Attention model created from config")
    return model

