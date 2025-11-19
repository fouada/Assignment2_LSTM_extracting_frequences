"""
Unit tests for sequence_dataset module.
"""

import pytest
import numpy as np
import torch
from src.data import SignalGenerator, SignalConfig
from src.data.sequence_dataset import (
    SequenceDataset,
    SequenceDataLoader,
    create_sequence_dataloaders
)


class TestSequenceDataset:
    """Test SequenceDataset class."""
    
    @pytest.fixture
    def generator(self):
        """Create a test generator."""
        config = SignalConfig(
            frequencies=[1.0, 3.0, 5.0, 7.0],
            sampling_rate=100,
            duration=1.0,
            amplitude_range=(0.8, 1.2),
            phase_range=(0, 2*np.pi),
            seed=42
        )
        return SignalGenerator(config)
    
    def test_dataset_initialization(self, generator):
        """Test dataset initializes correctly."""
        dataset = SequenceDataset(
            generator,
            sequence_length=10,
            stride=5,
            normalize=True
        )
        
        assert dataset.sequence_length == 10
        assert dataset.stride == 5
        assert dataset.normalize is True
        assert dataset.num_frequencies == 4
        assert dataset.num_time_samples == 100
        
    def test_dataset_initialization_default_stride(self, generator):
        """Test dataset uses sequence_length as default stride."""
        dataset = SequenceDataset(
            generator,
            sequence_length=10,
            normalize=False
        )
        
        assert dataset.stride == 10
    
    def test_dataset_length(self, generator):
        """Test dataset returns correct length."""
        dataset = SequenceDataset(
            generator,
            sequence_length=10,
            stride=5,
            normalize=False
        )
        
        # With 100 samples, L=10, stride=5: (100-10)//5 + 1 = 19 sequences per freq
        # Total: 19 * 4 = 76
        expected_seqs_per_freq = (100 - 10) // 5 + 1
        expected_total = expected_seqs_per_freq * 4
        
        assert dataset.sequences_per_freq == expected_seqs_per_freq
        assert len(dataset) == expected_total
    
    def test_dataset_getitem_shape(self, generator):
        """Test getitem returns correct shapes."""
        dataset = SequenceDataset(
            generator,
            sequence_length=10,
            stride=5,
            normalize=False
        )
        
        input_seq, target_seq, metadata = dataset[0]
        
        # Input: (sequence_length, 5) = (10, 5) for [S[t], C1, C2, C3, C4]
        assert input_seq.shape == (10, 5)
        # Target: (sequence_length, 1) = (10, 1)
        assert target_seq.shape == (10, 1)
        
        assert isinstance(input_seq, torch.Tensor)
        assert isinstance(target_seq, torch.Tensor)
        
    def test_dataset_getitem_metadata(self, generator):
        """Test getitem returns correct metadata."""
        dataset = SequenceDataset(
            generator,
            sequence_length=10,
            stride=5,
            normalize=False
        )
        
        _, _, metadata = dataset[0]
        
        assert 'freq_idx' in metadata
        assert 'seq_idx' in metadata
        assert 'start_time' in metadata
        assert 'end_time' in metadata
        
        assert metadata['freq_idx'] == 0
        assert metadata['seq_idx'] == 0
        assert metadata['start_time'] == 0
        assert metadata['end_time'] == 10
    
    def test_dataset_one_hot_encoding(self, generator):
        """Test one-hot encoding is correct."""
        dataset = SequenceDataset(
            generator,
            sequence_length=10,
            stride=10,
            normalize=False
        )
        
        # First sequence should have C = [1, 0, 0, 0]
        input_seq, _, metadata = dataset[0]
        assert metadata['freq_idx'] == 0
        assert input_seq[0, 1] == 1.0  # C1
        assert input_seq[0, 2] == 0.0  # C2
        assert input_seq[0, 3] == 0.0  # C3
        assert input_seq[0, 4] == 0.0  # C4
        
        # Last frequency should have C = [0, 0, 0, 1]
        # Get the first sequence of the last frequency
        seqs_per_freq = dataset.sequences_per_freq
        last_freq_idx = 3 * seqs_per_freq
        input_seq, _, metadata = dataset[last_freq_idx]
        assert metadata['freq_idx'] == 3
        assert input_seq[0, 1] == 0.0  # C1
        assert input_seq[0, 2] == 0.0  # C2
        assert input_seq[0, 3] == 0.0  # C3
        assert input_seq[0, 4] == 1.0  # C4
    
    def test_dataset_normalization(self, generator):
        """Test signal normalization works."""
        dataset_norm = SequenceDataset(
            generator,
            sequence_length=10,
            normalize=True
        )
        
        dataset_no_norm = SequenceDataset(
            generator,
            sequence_length=10,
            normalize=False
        )
        
        # Normalized dataset should have stored mean and std
        assert dataset_norm.signal_mean != 0.0
        assert dataset_norm.signal_std != 1.0
        
        # Non-normalized should have defaults
        assert dataset_no_norm.signal_mean == 0.0
        assert dataset_no_norm.signal_std == 1.0
        
        # Normalized signals should be different
        input_norm, _, _ = dataset_norm[0]
        input_no_norm, _, _ = dataset_no_norm[0]
        
        assert not torch.allclose(input_norm[:, 0], input_no_norm[:, 0])
    
    def test_dataset_sequence_indexing(self, generator):
        """Test that different indices give different sequences."""
        dataset = SequenceDataset(
            generator,
            sequence_length=10,
            stride=5,
            normalize=False
        )
        
        _, _, meta0 = dataset[0]
        _, _, meta1 = dataset[1]
        
        # Second sequence should start 5 samples later (stride=5)
        assert meta1['start_time'] == meta0['start_time'] + 5
        assert meta1['end_time'] == meta0['end_time'] + 5
    
    def test_get_full_timeseries(self, generator):
        """Test getting complete time series for a frequency."""
        dataset = SequenceDataset(
            generator,
            sequence_length=10,
            stride=10,
            normalize=False
        )
        
        input_batch, target_batch = dataset.get_full_timeseries(freq_idx=0)
        
        # Should return all sequences for frequency 0
        assert input_batch.shape[0] == dataset.sequences_per_freq
        assert input_batch.shape[1] == 10  # sequence_length
        assert input_batch.shape[2] == 5   # features
        
        assert target_batch.shape[0] == dataset.sequences_per_freq
        assert target_batch.shape[1] == 10
        assert target_batch.shape[2] == 1
    
    def test_get_full_timeseries_different_frequencies(self, generator):
        """Test that different frequencies give different time series."""
        dataset = SequenceDataset(
            generator,
            sequence_length=10,
            stride=10,
            normalize=False
        )
        
        input_0, target_0 = dataset.get_full_timeseries(freq_idx=0)
        input_1, target_1 = dataset.get_full_timeseries(freq_idx=1)
        
        # Targets should be different for different frequencies
        assert not torch.allclose(target_0, target_1)
        
        # One-hot encodings should be different
        assert not torch.allclose(input_0[:, :, 1:], input_1[:, :, 1:])


class TestSequenceDataLoader:
    """Test SequenceDataLoader class."""
    
    @pytest.fixture
    def dataset(self):
        """Create a test dataset."""
        config = SignalConfig(
            frequencies=[1.0, 3.0],
            sampling_rate=100,
            duration=1.0,
            amplitude_range=(0.8, 1.2),
            phase_range=(0, 2*np.pi),
            seed=42
        )
        generator = SignalGenerator(config)
        return SequenceDataset(
            generator,
            sequence_length=10,
            stride=10,
            normalize=False
        )
    
    def test_dataloader_initialization(self, dataset):
        """Test dataloader initializes correctly."""
        loader = SequenceDataLoader(
            dataset,
            batch_size=8,
            shuffle=True,
            drop_last=True
        )
        
        assert loader.batch_size == 8
        assert loader.shuffle is True
        assert loader.drop_last is True
    
    def test_dataloader_length(self, dataset):
        """Test dataloader returns correct length."""
        # Dataset has 2 frequencies, 10 sequences each = 20 total
        loader = SequenceDataLoader(dataset, batch_size=8, drop_last=False)
        
        # 20 sequences / 8 batch_size = 3 batches (8, 8, 4)
        assert len(loader) == 3
        
        # With drop_last=True
        loader_drop = SequenceDataLoader(dataset, batch_size=8, drop_last=True)
        # Should drop the incomplete batch
        assert len(loader_drop) == 2
    
    def test_dataloader_iteration(self, dataset):
        """Test dataloader iteration works."""
        loader = SequenceDataLoader(dataset, batch_size=8, shuffle=False)
        
        batches = list(loader)
        assert len(batches) > 0
        
        # First batch should have correct structure
        batch = batches[0]
        assert 'input' in batch
        assert 'target' in batch
        assert 'metadata' in batch
        
        # Check shapes
        assert batch['input'].shape[0] == 8  # batch_size
        assert batch['input'].shape[1] == 10  # sequence_length
        assert batch['input'].shape[2] == 3   # features (S[t] + 2 one-hot for 2 frequencies)
        
        assert batch['target'].shape[0] == 8
        assert batch['target'].shape[1] == 10
        assert batch['target'].shape[2] == 1
        
        assert len(batch['metadata']) == 8
    
    def test_dataloader_shuffle(self, dataset):
        """Test that shuffle produces different orderings."""
        np.random.seed(1)
        loader1 = SequenceDataLoader(dataset, batch_size=4, shuffle=True)
        batches1 = list(loader1)
        meta1 = [b['metadata'][0]['freq_idx'] for b in batches1]
        
        np.random.seed(2)
        loader2 = SequenceDataLoader(dataset, batch_size=4, shuffle=True)
        batches2 = list(loader2)
        meta2 = [b['metadata'][0]['freq_idx'] for b in batches2]
        
        # Different seeds should produce different orderings (most likely)
        # Note: this could theoretically fail with extremely low probability
        assert meta1 != meta2 or len(meta1) < 2
    
    def test_dataloader_no_shuffle(self, dataset):
        """Test that without shuffle, data is in order."""
        loader = SequenceDataLoader(dataset, batch_size=4, shuffle=False)
        
        batches = list(loader)
        
        # Extract frequency indices from metadata
        all_freq_indices = []
        for batch in batches:
            for meta in batch['metadata']:
                all_freq_indices.append(meta['freq_idx'])
        
        # First half should be frequency 0, second half frequency 1
        # (assuming sequences_per_freq >= batch_size)
        seqs_per_freq = dataset.sequences_per_freq
        assert all(all_freq_indices[i] == 0 for i in range(seqs_per_freq))
        assert all(all_freq_indices[i] == 1 for i in range(seqs_per_freq, len(all_freq_indices)))
    
    def test_dataloader_drop_last(self, dataset):
        """Test drop_last functionality."""
        # With a batch size that doesn't divide evenly
        loader_keep = SequenceDataLoader(dataset, batch_size=7, drop_last=False)
        loader_drop = SequenceDataLoader(dataset, batch_size=7, drop_last=True)
        
        batches_keep = list(loader_keep)
        batches_drop = list(loader_drop)
        
        # Should have different numbers of batches
        assert len(batches_keep) > len(batches_drop)
        
        # Last batch of loader_keep should be incomplete
        last_batch = batches_keep[-1]
        assert last_batch['input'].shape[0] < 7
        
        # All batches in loader_drop should be complete
        for batch in batches_drop:
            assert batch['input'].shape[0] == 7


class TestCreateSequenceDataloaders:
    """Test create_sequence_dataloaders factory function."""
    
    @pytest.fixture
    def generators(self):
        """Create train and test generators."""
        config_train = SignalConfig(
            frequencies=[1.0, 3.0, 5.0],
            sampling_rate=100,
            duration=1.0,
            amplitude_range=(0.8, 1.2),
            phase_range=(0, 2*np.pi),
            seed=1
        )
        
        config_test = SignalConfig(
            frequencies=[1.0, 3.0, 5.0],
            sampling_rate=100,
            duration=1.0,
            amplitude_range=(0.8, 1.2),
            phase_range=(0, 2*np.pi),
            seed=2
        )
        
        train_gen = SignalGenerator(config_train)
        test_gen = SignalGenerator(config_test)
        
        return train_gen, test_gen
    
    def test_create_dataloaders_basic(self, generators):
        """Test basic dataloader creation."""
        train_gen, test_gen = generators
        
        train_loader, test_loader = create_sequence_dataloaders(
            train_gen,
            test_gen,
            sequence_length=10,
            batch_size=8,
            normalize=True,
            shuffle_train=True
        )
        
        assert isinstance(train_loader, SequenceDataLoader)
        assert isinstance(test_loader, SequenceDataLoader)
        
        assert train_loader.batch_size == 8
        assert test_loader.batch_size == 8
        
        assert train_loader.shuffle is True
        assert test_loader.shuffle is False
    
    def test_create_dataloaders_custom_stride(self, generators):
        """Test dataloader creation with custom stride."""
        train_gen, test_gen = generators
        
        train_loader, test_loader = create_sequence_dataloaders(
            train_gen,
            test_gen,
            sequence_length=10,
            batch_size=8,
            stride=5,
            normalize=True
        )
        
        # Check that stride was applied
        assert train_loader.dataset.stride == 5
        assert test_loader.dataset.stride == 5
    
    def test_create_dataloaders_no_normalization(self, generators):
        """Test dataloader creation without normalization."""
        train_gen, test_gen = generators
        
        train_loader, test_loader = create_sequence_dataloaders(
            train_gen,
            test_gen,
            sequence_length=10,
            batch_size=8,
            normalize=False
        )
        
        assert train_loader.dataset.normalize is False
        assert test_loader.dataset.normalize is False
        
        assert train_loader.dataset.signal_mean == 0.0
        assert train_loader.dataset.signal_std == 1.0
    
    def test_create_dataloaders_different_data(self, generators):
        """Test that train and test loaders have different data."""
        train_gen, test_gen = generators
        
        train_loader, test_loader = create_sequence_dataloaders(
            train_gen,
            test_gen,
            sequence_length=10,
            batch_size=8,
            normalize=False
        )
        
        # Get first batch from each
        train_batch = next(iter(train_loader))
        test_batch = next(iter(test_loader))
        
        # Data should be different due to different seeds
        assert not torch.allclose(train_batch['input'], test_batch['input'])


class TestSequenceDatasetEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_sequence_length_one(self):
        """Test with sequence length of 1 (edge case)."""
        config = SignalConfig(
            frequencies=[1.0, 3.0],
            sampling_rate=100,
            duration=1.0,
            amplitude_range=(0.8, 1.2),
            phase_range=(0, 2*np.pi),
            seed=42
        )
        generator = SignalGenerator(config)
        
        dataset = SequenceDataset(
            generator,
            sequence_length=1,
            normalize=False
        )
        
        input_seq, target_seq, _ = dataset[0]
        
        assert input_seq.shape == (1, 3)  # (1, [S[t], C1, C2])
        assert target_seq.shape == (1, 1)
    
    def test_large_stride(self):
        """Test with stride larger than sequence length."""
        config = SignalConfig(
            frequencies=[1.0],
            sampling_rate=100,
            duration=1.0,
            amplitude_range=(0.8, 1.2),
            phase_range=(0, 2*np.pi),
            seed=42
        )
        generator = SignalGenerator(config)
        
        dataset = SequenceDataset(
            generator,
            sequence_length=10,
            stride=20,
            normalize=False
        )
        
        # Should create non-overlapping sequences with gaps
        assert dataset.stride == 20
        assert len(dataset) > 0
    
    def test_device_parameter(self):
        """Test device parameter is stored."""
        config = SignalConfig(
            frequencies=[1.0, 3.0],
            sampling_rate=100,
            duration=1.0,
            amplitude_range=(0.8, 1.2),
            phase_range=(0, 2*np.pi),
            seed=42
        )
        generator = SignalGenerator(config)
        
        dataset = SequenceDataset(
            generator,
            sequence_length=10,
            device='cpu'
        )
        
        assert dataset.device == 'cpu'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

