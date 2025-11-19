"""
Sequence Dataset Module for L > 1 experiments
Handles sequential data loading for different sequence lengths.

Author: Professional ML Engineering Team
Date: 2025
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Dict, Optional
import logging

from .signal_generator import SignalGenerator

logger = logging.getLogger(__name__)


class SequenceDataset(Dataset):
    """
    PyTorch Dataset for sequence-based LSTM training (L > 1).
    
    This dataset creates overlapping or non-overlapping sequences
    of length L for training LSTMs with temporal context.
    """
    
    def __init__(
        self,
        signal_generator: SignalGenerator,
        sequence_length: int = 1,
        stride: int = None,
        normalize: bool = True,
        device: str = 'cpu'
    ):
        """
        Initialize the sequence dataset.
        
        Args:
            signal_generator: SignalGenerator instance
            sequence_length: L - number of consecutive time steps per sequence
            stride: Step size between sequences (default: sequence_length for non-overlapping)
            normalize: Whether to normalize the signals
            device: Device to put tensors on
        """
        self.generator = signal_generator
        self.sequence_length = sequence_length
        self.stride = stride if stride is not None else sequence_length
        self.normalize = normalize
        self.device = device
        
        # Generate data
        logger.info(f"Generating sequence dataset with L={sequence_length}, stride={self.stride}...")
        self.mixed_signal, self.targets = self.generator.generate_complete_dataset()
        
        self.num_time_samples = len(self.mixed_signal)
        self.num_frequencies = len(self.generator.frequencies)
        
        # Calculate number of sequences per frequency
        self.sequences_per_freq = (self.num_time_samples - sequence_length) // self.stride + 1
        self.total_sequences = self.sequences_per_freq * self.num_frequencies
        
        # Store normalization parameters
        self.signal_mean = 0.0
        self.signal_std = 1.0
        
        # Normalize if requested
        if self.normalize:
            self.signal_mean = np.mean(self.mixed_signal)
            self.signal_std = np.std(self.mixed_signal)
            
            self.mixed_signal = (self.mixed_signal - self.signal_mean) / (self.signal_std + 1e-8)
            
            for freq_idx in self.targets:
                self.targets[freq_idx] = (self.targets[freq_idx] - self.signal_mean) / (self.signal_std + 1e-8)
            
            logger.info(f"Signals normalized: mean={self.signal_mean:.4f}, std={self.signal_std:.4f}")
        
        logger.info(f"Sequence dataset created:")
        logger.info(f"  Sequence length (L): {sequence_length}")
        logger.info(f"  Sequences per frequency: {self.sequences_per_freq}")
        logger.info(f"  Total sequences: {self.total_sequences}")
        logger.info(f"  Coverage: {self.sequences_per_freq * self.stride}/{self.num_time_samples} time steps")
    
    def __len__(self) -> int:
        """Return total number of sequences."""
        return self.total_sequences
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Get a single sequence.
        
        Args:
            idx: Sequence index (0 to total_sequences-1)
            
        Returns:
            Tuple of (input_sequence, target_sequence, metadata)
            - input_sequence: shape (sequence_length, 5) - [S[t], C1, C2, C3, C4]
            - target_sequence: shape (sequence_length, 1) - Target_i[t]
            - metadata: dict with freq_idx, start_time, end_time
        """
        # Determine which frequency and which sequence
        freq_idx = idx // self.sequences_per_freq
        seq_idx = idx % self.sequences_per_freq
        
        # Calculate start and end time indices
        start_time = seq_idx * self.stride
        end_time = start_time + self.sequence_length
        
        # Build input sequence: [S[t], C] for each time step
        input_sequence = []
        target_sequence = []
        
        # Create one-hot encoding for this frequency
        one_hot = np.zeros(self.num_frequencies, dtype=np.float32)
        one_hot[freq_idx] = 1.0
        
        for t in range(start_time, end_time):
            # Input: [S[t], C1, C2, C3, C4]
            signal_value = self.mixed_signal[t]
            input_features = np.concatenate([[signal_value], one_hot])
            input_sequence.append(input_features)
            
            # Target: pure sine at this frequency and time
            target_value = self.targets[freq_idx][t]
            target_sequence.append([target_value])
        
        # Convert to tensors
        input_tensor = torch.tensor(input_sequence, dtype=torch.float32)
        target_tensor = torch.tensor(target_sequence, dtype=torch.float32)
        
        # Metadata for tracking
        metadata = {
            'freq_idx': freq_idx,
            'seq_idx': seq_idx,
            'start_time': start_time,
            'end_time': end_time
        }
        
        return input_tensor, target_tensor, metadata
    
    def get_full_timeseries(
        self, 
        freq_idx: int,
        batch_size: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get complete time series for a specific frequency as sequences.
        
        Useful for evaluation on complete signal.
        
        Args:
            freq_idx: Frequency index (0-3)
            batch_size: How many sequences to return at once
            
        Returns:
            Tuple of (input_sequences, target_sequences)
        """
        sequences = []
        targets = []
        
        start_idx = freq_idx * self.sequences_per_freq
        end_idx = start_idx + self.sequences_per_freq
        
        for idx in range(start_idx, end_idx):
            inp_seq, tgt_seq, _ = self.__getitem__(idx)
            sequences.append(inp_seq)
            targets.append(tgt_seq)
        
        # Stack sequences
        input_batch = torch.stack(sequences)
        target_batch = torch.stack(targets)
        
        return input_batch, target_batch


class SequenceDataLoader:
    """
    Custom data loader for sequence-based training.
    
    Handles batching of sequences while maintaining proper ordering.
    """
    
    def __init__(
        self,
        dataset: SequenceDataset,
        batch_size: int = 32,
        shuffle: bool = False,
        drop_last: bool = False
    ):
        """
        Initialize sequence data loader.
        
        Args:
            dataset: SequenceDataset instance
            batch_size: Number of sequences per batch
            shuffle: Whether to shuffle sequences
            drop_last: Whether to drop incomplete final batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        logger.info(f"SequenceDataLoader created: batch_size={batch_size}, shuffle={shuffle}")
    
    def __iter__(self):
        """
        Iterate through dataset.
        
        Yields batches of sequences.
        """
        indices = list(range(len(self.dataset)))
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue
            
            # Collect batch
            inputs = []
            targets = []
            metadata_list = []
            
            for idx in batch_indices:
                inp, tgt, meta = self.dataset[idx]
                inputs.append(inp)
                targets.append(tgt)
                metadata_list.append(meta)
            
            # Stack into tensors
            input_batch = torch.stack(inputs)  # (batch_size, seq_len, 5)
            target_batch = torch.stack(targets)  # (batch_size, seq_len, 1)
            
            yield {
                'input': input_batch,
                'target': target_batch,
                'metadata': metadata_list
            }
    
    def __len__(self) -> int:
        """Return number of batches."""
        num_batches = len(self.dataset) // self.batch_size
        if not self.drop_last and len(self.dataset) % self.batch_size != 0:
            num_batches += 1
        return num_batches


def create_sequence_dataloaders(
    train_generator: SignalGenerator,
    test_generator: SignalGenerator,
    sequence_length: int = 10,
    batch_size: int = 32,
    stride: int = None,
    normalize: bool = True,
    shuffle_train: bool = True,
    device: str = 'cpu'
) -> Tuple[SequenceDataLoader, SequenceDataLoader]:
    """
    Factory function to create train and test sequence data loaders.
    
    Args:
        train_generator: Training signal generator
        test_generator: Test signal generator
        sequence_length: L - length of sequences
        batch_size: Number of sequences per batch
        stride: Step between sequences (default: sequence_length)
        normalize: Whether to normalize signals
        shuffle_train: Whether to shuffle training data
        device: Device for tensors
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    logger.info(f"Creating sequence dataloaders with L={sequence_length}...")
    
    # Create datasets
    train_dataset = SequenceDataset(
        train_generator,
        sequence_length=sequence_length,
        stride=stride,
        normalize=normalize,
        device=device
    )
    
    test_dataset = SequenceDataset(
        test_generator,
        sequence_length=sequence_length,
        stride=stride,
        normalize=normalize,
        device=device
    )
    
    # Create loaders
    train_loader = SequenceDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        drop_last=False
    )
    
    test_loader = SequenceDataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    
    logger.info("Sequence dataloaders created successfully!")
    logger.info(f"Train batches per epoch: {len(train_loader)}")
    logger.info(f"Test batches per epoch: {len(test_loader)}")
    
    return train_loader, test_loader

