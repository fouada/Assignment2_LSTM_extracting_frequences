"""
PyTorch Dataset Module
Professional dataset implementation for LSTM training with stateful processing.

Author: Professional ML Engineering Team
Date: 2025
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Optional
import logging

from .signal_generator import SignalGenerator

logger = logging.getLogger(__name__)


class FrequencyExtractionDataset(Dataset):
    """
    PyTorch Dataset for LSTM frequency extraction task.
    
    Dataset Structure (for L=1):
    - Input: [S[t], C] where S[t] is mixed signal, C is one-hot frequency selector
    - Output: Target_i[t] (pure sine at selected frequency)
    
    Total samples: 40,000 (10,000 time samples × 4 frequencies)
    """
    
    def __init__(
        self,
        signal_generator: SignalGenerator,
        normalize: bool = True,
        device: str = 'cpu',
        normalization_params: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the dataset.

        Args:
            signal_generator: SignalGenerator instance
            normalize: Whether to normalize the mixed signal
            device: Device to put tensors on
            normalization_params: Optional dict with 'mean' and 'std' for test set normalization.
                                 If provided, uses these params instead of computing from data.
        """
        self.generator = signal_generator
        self.normalize = normalize
        self.device = device

        # Generate data
        logger.info("Generating dataset...")
        self.mixed_signal, self.targets = self.generator.generate_complete_dataset()

        self.num_time_samples = len(self.mixed_signal)
        self.num_frequencies = len(self.generator.frequencies)
        self.total_samples = self.num_time_samples * self.num_frequencies

        # Store normalization parameters
        self.signal_mean = 0.0
        self.signal_std = 1.0

        # Optional normalization - normalize ONLY the input (mixed signal)
        # Targets (pure sine waves) remain unchanged at ±1 amplitude
        if self.normalize:
            # Use provided normalization params (for test set) OR compute from data (for train set)
            if normalization_params is not None:
                self.signal_mean = normalization_params['mean']
                self.signal_std = normalization_params['std']
                logger.info(f"Using provided normalization params: mean={self.signal_mean:.4f}, std={self.signal_std:.4f}")
            else:
                # Calculate global statistics from mixed signal
                self.signal_mean = np.mean(self.mixed_signal)
                self.signal_std = np.std(self.mixed_signal)
                logger.info(f"Computed normalization params from data: mean={self.signal_mean:.4f}, std={self.signal_std:.4f}")

            # Normalize ONLY the mixed signal (input)
            self.mixed_signal = (self.mixed_signal - self.signal_mean) / (self.signal_std + 1e-8)

            # CRITICAL FIX: Do NOT normalize targets!
            # Targets are pure sine waves (±1 amplitude) - they should remain unchanged
            # The model learns to map from normalized inputs to pure sine outputs

            logger.info(f"Input (mixed signal) normalized: mean={self.signal_mean:.4f}, std={self.signal_std:.4f}")
            logger.info(f"✅ CORRECT: Targets (pure sine) kept at original scale (±1 amplitude)")

        logger.info(f"Dataset created: {self.total_samples} total samples "
                   f"({self.num_time_samples} time steps × {self.num_frequencies} frequencies)")

    def get_normalization_params(self) -> Dict[str, float]:
        """
        Get normalization parameters for use with test set.

        Returns:
            Dictionary with 'mean' and 'std' keys
        """
        return {
            'mean': self.signal_mean,
            'std': self.signal_std
        }
    
    def __len__(self) -> int:
        """Return total number of samples."""
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Index (0 to total_samples-1)
            
        Returns:
            Tuple of (input_tensor, target_tensor)
            - input_tensor: [S[t], C] shape (5,)
            - target_tensor: Target_i[t] shape (1,)
        """
        # Determine which frequency and which time sample
        freq_idx = idx // self.num_time_samples
        time_idx = idx % self.num_time_samples
        
        # Get mixed signal value at time t
        signal_value = self.mixed_signal[time_idx]
        
        # Create one-hot encoding for frequency selection
        one_hot = np.zeros(self.num_frequencies, dtype=np.float32)
        one_hot[freq_idx] = 1.0
        
        # Concatenate: [S[t], C1, C2, C3, C4]
        input_features = np.concatenate([[signal_value], one_hot])
        
        # Get target (pure sine at selected frequency)
        target_value = self.targets[freq_idx][time_idx]
        
        # Convert to tensors
        input_tensor = torch.tensor(input_features, dtype=torch.float32)
        target_tensor = torch.tensor([target_value], dtype=torch.float32)
        
        return input_tensor, target_tensor
    
    def get_sequence(
        self, 
        start_idx: int, 
        sequence_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence of consecutive samples (for L > 1).
        
        Args:
            start_idx: Starting index
            sequence_length: Number of consecutive samples
            
        Returns:
            Tuple of (input_sequence, target_sequence)
            - input_sequence: shape (sequence_length, num_frequencies + 1)
            - target_sequence: shape (sequence_length, 1)
        """
        inputs = []
        targets = []
        
        for i in range(sequence_length):
            inp, tgt = self.__getitem__(start_idx + i)
            inputs.append(inp)
            targets.append(tgt)
        
        input_sequence = torch.stack(inputs)
        target_sequence = torch.stack(targets)
        
        return input_sequence, target_sequence
    
    def get_sequence_batch(
        self, 
        start_idx: int, 
        sequence_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence of consecutive samples (for L > 1).
        Alias for get_sequence() for backward compatibility.
        
        Args:
            start_idx: Starting index
            sequence_length: Number of consecutive samples
            
        Returns:
            Tuple of (input_sequence, target_sequence)
            - input_sequence: shape (sequence_length, num_frequencies + 1)
            - target_sequence: shape (sequence_length, 1)
        """
        return self.get_sequence(start_idx, sequence_length)
    
    def get_time_series_for_frequency(
        self, 
        freq_idx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get complete time series for a specific frequency.
        
        Useful for visualization and evaluation.
        
        Args:
            freq_idx: Frequency index (0-3)
            
        Returns:
            Tuple of (time_vector, mixed_signal, target_signal)
        """
        time_vector = self.generator.time_vector
        mixed = self.mixed_signal
        target = self.targets[freq_idx]
        
        return time_vector, mixed, target


class StatefulDataLoader:
    """
    Custom data loader that maintains sample order for stateful LSTM training.
    
    For L=1 with state preservation, we need to feed samples in exact temporal order
    within each frequency group.
    """
    
    def __init__(
        self,
        dataset: FrequencyExtractionDataset,
        batch_size: int = 32,
        shuffle_frequencies: bool = False
    ):
        """
        Initialize stateful data loader.
        
        Args:
            dataset: FrequencyExtractionDataset instance
            batch_size: Batch size (number of time steps)
            shuffle_frequencies: Whether to shuffle frequency order between epochs
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle_frequencies = shuffle_frequencies
        
        self.num_time_samples = dataset.num_time_samples
        self.num_frequencies = dataset.num_frequencies
        
        logger.info(f"StatefulDataLoader created: batch_size={batch_size}")
    
    def __iter__(self):
        """
        Iterate through dataset maintaining temporal order.
        
        Yields batches in the format required for stateful LSTM training.
        """
        # Determine frequency processing order
        freq_indices = list(range(self.num_frequencies))
        if self.shuffle_frequencies:
            np.random.shuffle(freq_indices)
        
        for freq_idx in freq_indices:
            # Process all time samples for this frequency
            start_sample = freq_idx * self.num_time_samples
            
            for batch_start in range(0, self.num_time_samples, self.batch_size):
                batch_end = min(batch_start + self.batch_size, self.num_time_samples)
                actual_batch_size = batch_end - batch_start
                
                # Collect batch
                inputs = []
                targets = []
                
                for t in range(batch_start, batch_end):
                    idx = start_sample + t
                    inp, tgt = self.dataset[idx]
                    inputs.append(inp)
                    targets.append(tgt)
                
                # Stack into tensors
                input_batch = torch.stack(inputs)  # (batch_size, 5)
                target_batch = torch.stack(targets)  # (batch_size, 1)
                
                # Return batch with metadata
                yield {
                    'input': input_batch,
                    'target': target_batch,
                    'freq_idx': freq_idx,
                    'time_range': (batch_start, batch_end),
                    'is_first_batch': (batch_start == 0),
                    'is_last_batch': (batch_end == self.num_time_samples)
                }
    
    def __len__(self) -> int:
        """Return number of batches per epoch."""
        batches_per_frequency = (self.num_time_samples + self.batch_size - 1) // self.batch_size
        return batches_per_frequency * self.num_frequencies


def create_dataloaders(
    train_generator: SignalGenerator,
    test_generator: SignalGenerator,
    batch_size: int = 32,
    normalize: bool = True,
    device: str = 'cpu'
) -> Tuple[StatefulDataLoader, StatefulDataLoader, FrequencyExtractionDataset, FrequencyExtractionDataset]:
    """
    Factory function to create train and test data loaders.

    Args:
        train_generator: Training signal generator
        test_generator: Test signal generator
        batch_size: Batch size
        normalize: Whether to normalize signals
        device: Device for tensors

    Returns:
        Tuple of (train_loader, test_loader, train_dataset, test_dataset)
    """
    logger.info("Creating datasets and dataloaders...")

    # Create TRAIN dataset first (computes normalization params)
    train_dataset = FrequencyExtractionDataset(
        train_generator,
        normalize=normalize,
        device=device
    )

    # Create TEST dataset using TRAIN normalization params (CRITICAL FIX!)
    test_normalization_params = train_dataset.get_normalization_params() if normalize else None
    test_dataset = FrequencyExtractionDataset(
        test_generator,
        normalize=normalize,
        device=device,
        normalization_params=test_normalization_params  # Use training stats!
    )

    logger.info("✅ CRITICAL FIX: Test dataset uses SAME normalization as training!")

    # Create stateful loaders
    train_loader = StatefulDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle_frequencies=True  # Shuffle frequency order each epoch
    )

    test_loader = StatefulDataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle_frequencies=False  # Keep consistent for evaluation
    )

    logger.info("Dataloaders created successfully!")
    logger.info(f"Train batches per epoch: {len(train_loader)}")
    logger.info(f"Test batches per epoch: {len(test_loader)}")

    return train_loader, test_loader, train_dataset, test_dataset

