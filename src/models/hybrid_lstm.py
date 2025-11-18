"""
Hybrid Time-Frequency LSTM for Multi-Modal Signal Processing
INNOVATION: Combines time-domain LSTM with frequency-domain analysis

Author: Professional ML Engineering Team
Date: November 2025
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, List
import logging
import numpy as np
from .lstm_extractor import StatefulLSTMExtractor

logger = logging.getLogger(__name__)


class HybridLSTMExtractor(nn.Module):
    """
    Hybrid LSTM combining time-domain and frequency-domain processing.
    
    INNOVATION HIGHLIGHTS:
    - Multi-modal learning: time + frequency domains
    - Combines deep learning with classical DSP
    - Frequency-domain features complement temporal patterns
    - Novel architecture for signal processing
    
    Key Features:
    - Parallel time-domain (LSTM) and frequency-domain (FFT) paths
    - Adaptive fusion of both representations
    - Frequency-aware feature extraction
    - Minimal computational overhead
    """
    
    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
        fft_size: int = 256,
        freq_hidden_size: int = 64,
        fusion_strategy: str = 'concat'  # 'concat', 'add', 'attention'
    ):
        """
        Initialize Hybrid LSTM Extractor.
        
        Args:
            fft_size: Size of FFT window
            freq_hidden_size: Hidden size for frequency path
            fusion_strategy: How to combine time and freq features
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fft_size = fft_size
        self.freq_hidden_size = freq_hidden_size
        self.fusion_strategy = fusion_strategy
        
        # Time-domain path (LSTM)
        self.time_model = StatefulLSTMExtractor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=hidden_size,  # Output features, not final output
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        # Frequency-domain path
        self.freq_encoder = FrequencyEncoder(
            fft_size=fft_size,
            hidden_size=freq_hidden_size,
            dropout=dropout
        )
        
        # Fusion layer
        if fusion_strategy == 'concat':
            fusion_input_size = hidden_size + freq_hidden_size
        elif fusion_strategy == 'add':
            # Need same dimensions
            fusion_input_size = hidden_size
            self.freq_projection = nn.Linear(freq_hidden_size, hidden_size)
        elif fusion_strategy == 'attention':
            fusion_input_size = hidden_size
            self.fusion_attention = FusionAttention(
                time_dim=hidden_size,
                freq_dim=freq_hidden_size
            )
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # Buffer for frequency analysis
        self.signal_buffer: List[torch.Tensor] = []
        self.buffer_size = fft_size
        
        logger.info(f"HybridLSTMExtractor initialized:")
        logger.info(f"  Time path: LSTM (hidden={hidden_size})")
        logger.info(f"  Freq path: FFT (size={fft_size})")
        logger.info(f"  Fusion strategy: {fusion_strategy}")
        logger.info(f"  Total parameters: {self.count_parameters():,}")
    
    def reset_state(self):
        """Reset both time-domain state and signal buffer."""
        self.time_model.reset_state()
        self.signal_buffer = []
        logger.debug("Hybrid model state reset")
    
    def forward(
        self,
        x: torch.Tensor,
        reset_state: bool = False,
        return_intermediate: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through hybrid architecture.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
            reset_state: Whether to reset time-domain state
            return_intermediate: Whether to return intermediate features
            
        Returns:
            Output tensor and optionally intermediate features
        """
        if reset_state:
            self.reset_state()
        
        # Handle single sample case
        if x.dim() == 2:
            x = x.unsqueeze(1)
            single_sample = True
        else:
            single_sample = False
        
        batch_size, seq_len, _ = x.shape
        
        # Extract raw signal (first feature)
        raw_signal = x[:, :, 0]  # Shape: (batch, seq_len)
        
        # Add to buffer
        self.signal_buffer.append(raw_signal.detach())
        if len(self.signal_buffer) > self.buffer_size:
            self.signal_buffer.pop(0)
        
        # === Time-domain path ===
        time_features = self.time_model(x, reset_state=reset_state)
        
        # === Frequency-domain path ===
        if len(self.signal_buffer) >= 32:  # Minimum samples for FFT
            # Concatenate buffer
            signal_window = torch.cat(self.signal_buffer, dim=1)
            
            # Take most recent samples
            if signal_window.size(1) > self.fft_size:
                signal_window = signal_window[:, -self.fft_size:]
            
            # Frequency features
            freq_features = self.freq_encoder(signal_window)
            
            # Expand to match sequence length
            freq_features = freq_features.unsqueeze(1).expand(-1, seq_len, -1)
        else:
            # Not enough samples yet, use zeros
            freq_features = torch.zeros(
                batch_size,
                seq_len,
                self.freq_hidden_size,
                device=x.device
            )
        
        # === Fusion ===
        if self.fusion_strategy == 'concat':
            combined = torch.cat([time_features, freq_features], dim=-1)
        
        elif self.fusion_strategy == 'add':
            # Project frequency features to same dimension
            freq_features = self.freq_projection(freq_features)
            combined = time_features + freq_features
        
        elif self.fusion_strategy == 'attention':
            combined = self.fusion_attention(time_features, freq_features)
        
        # Final output
        output = self.fusion(combined)
        
        # Remove sequence dimension if input was single sample
        if single_sample:
            output = output.squeeze(1)
        
        if return_intermediate:
            return output, {
                'time_features': time_features,
                'freq_features': freq_features,
                'combined': combined
            }
        
        return output
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def visualize_feature_importance(
        self,
        x: torch.Tensor,
        save_path: Optional[str] = None
    ):
        """
        Visualize relative importance of time vs frequency features.
        
        Args:
            x: Input sequence
            save_path: Path to save figure
        """
        import matplotlib.pyplot as plt
        
        self.eval()
        with torch.no_grad():
            # Get intermediate features
            _, intermediate = self.forward(x, return_intermediate=True)
            
            time_features = intermediate['time_features']
            freq_features = intermediate['freq_features']
            
            # Compute feature magnitudes
            time_mag = time_features.abs().mean().item()
            freq_mag = freq_features.abs().mean().item()
            
            # Normalize
            total = time_mag + freq_mag
            time_importance = time_mag / total
            freq_importance = freq_mag / total
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Pie chart
        ax1.pie(
            [time_importance, freq_importance],
            labels=['Time Domain', 'Frequency Domain'],
            autopct='%1.1f%%',
            startangle=90,
            colors=['#3498db', '#e74c3c']
        )
        ax1.set_title('Feature Importance', fontsize=14)
        
        # Bar chart
        features = ['Time\nDomain', 'Frequency\nDomain']
        importances = [time_importance, freq_importance]
        colors = ['#3498db', '#e74c3c']
        
        ax2.bar(features, importances, color=colors, alpha=0.7)
        ax2.set_ylabel('Relative Importance', fontsize=12)
        ax2.set_title('Feature Contribution', fontsize=14)
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance saved to {save_path}")
        
        return fig


class FrequencyEncoder(nn.Module):
    """
    Encoder for frequency-domain features using FFT.
    """
    
    def __init__(
        self,
        fft_size: int = 256,
        hidden_size: int = 64,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.fft_size = fft_size
        
        # FFT outputs complex numbers, we use magnitude spectrum
        # Real FFT has fft_size // 2 + 1 frequency bins
        freq_bins = fft_size // 2 + 1
        
        # Frequency feature extractor
        self.encoder = nn.Sequential(
            nn.Linear(freq_bins, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Learnable frequency weights (attention over frequencies)
        self.freq_attention = nn.Parameter(torch.ones(freq_bins))
    
    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Extract frequency-domain features.
        
        Args:
            signal: Time-domain signal (batch, time)
            
        Returns:
            Frequency features (batch, hidden_size)
        """
        batch_size = signal.size(0)
        
        # Pad if needed
        if signal.size(1) < self.fft_size:
            pad_size = self.fft_size - signal.size(1)
            signal = torch.nn.functional.pad(signal, (0, pad_size))
        elif signal.size(1) > self.fft_size:
            signal = signal[:, :self.fft_size]
        
        # Compute FFT (real-valued FFT)
        freq_spectrum = torch.fft.rfft(signal, n=self.fft_size)
        
        # Magnitude spectrum
        magnitude = torch.abs(freq_spectrum)
        
        # Apply frequency attention
        weighted_magnitude = magnitude * self.freq_attention.unsqueeze(0)
        
        # Extract features
        features = self.encoder(weighted_magnitude)
        
        return features


class FusionAttention(nn.Module):
    """
    Attention mechanism for fusing time and frequency features.
    """
    
    def __init__(self, time_dim: int, freq_dim: int):
        super().__init__()
        
        self.time_dim = time_dim
        self.freq_dim = freq_dim
        
        # Attention weights
        self.time_attention = nn.Sequential(
            nn.Linear(time_dim, time_dim // 2),
            nn.Tanh(),
            nn.Linear(time_dim // 2, 1)
        )
        
        self.freq_attention = nn.Sequential(
            nn.Linear(freq_dim, freq_dim // 2),
            nn.Tanh(),
            nn.Linear(freq_dim // 2, 1)
        )
        
        # Projection to common dimension
        self.freq_projection = nn.Linear(freq_dim, time_dim)
    
    def forward(
        self,
        time_features: torch.Tensor,
        freq_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse time and frequency features with attention.
        
        Args:
            time_features: Time-domain features
            freq_features: Frequency-domain features
            
        Returns:
            Fused features
        """
        # Compute attention weights
        time_weight = torch.sigmoid(self.time_attention(time_features))
        freq_weight = torch.sigmoid(self.freq_attention(freq_features))
        
        # Normalize
        total_weight = time_weight + freq_weight + 1e-8
        time_weight = time_weight / total_weight
        freq_weight = freq_weight / total_weight
        
        # Project frequency features
        freq_features_proj = self.freq_projection(freq_features)
        
        # Weighted combination
        fused = time_weight * time_features + freq_weight * freq_features_proj
        
        return fused


def create_hybrid_model(config: Dict) -> HybridLSTMExtractor:
    """
    Factory function to create hybrid model from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized HybridLSTMExtractor
    """
    model = HybridLSTMExtractor(
        input_size=config.get('input_size', 5),
        hidden_size=config.get('hidden_size', 128),
        num_layers=config.get('num_layers', 2),
        output_size=config.get('output_size', 1),
        dropout=config.get('dropout', 0.2),
        bidirectional=config.get('bidirectional', False),
        fft_size=config.get('fft_size', 256),
        freq_hidden_size=config.get('freq_hidden_size', 64),
        fusion_strategy=config.get('fusion_strategy', 'concat')
    )
    
    logger.info("Hybrid model created from config")
    return model

