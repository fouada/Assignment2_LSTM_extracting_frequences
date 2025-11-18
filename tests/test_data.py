"""
Unit tests for data generation module.
"""

import pytest
import numpy as np
import torch
from src.data import (
    SignalGenerator,
    SignalConfig,
    FrequencyExtractionDataset,
    create_train_test_generators
)


class TestSignalGenerator:
    """Test SignalGenerator class."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return SignalConfig(
            frequencies=[1.0, 3.0, 5.0, 7.0],
            sampling_rate=1000,
            duration=10.0,
            amplitude_range=(0.8, 1.2),
            phase_range=(0, 2*np.pi),
            seed=42
        )
    
    def test_generator_initialization(self, config):
        """Test generator initializes correctly."""
        generator = SignalGenerator(config)
        
        assert generator.num_frequencies == 4
        assert generator.num_samples == 10000
        assert len(generator.time_vector) == 10000
    
    def test_noisy_sine_generation(self, config):
        """Test noisy sine wave generation."""
        generator = SignalGenerator(config)
        time = np.linspace(0, 1, 100)
        
        noisy_sine = generator.generate_noisy_sine(1.0, time)
        
        assert len(noisy_sine) == 100
        assert isinstance(noisy_sine, np.ndarray)
    
    def test_pure_sine_generation(self, config):
        """Test pure sine wave generation."""
        generator = SignalGenerator(config)
        time = np.linspace(0, 1, 100)
        
        pure_sine = generator.generate_pure_sine(1.0, time)
        
        assert len(pure_sine) == 100
        # Verify it's a proper sine wave
        expected = np.sin(2 * np.pi * 1.0 * time)
        np.testing.assert_array_almost_equal(pure_sine, expected)
    
    def test_mixed_signal_shape(self, config):
        """Test mixed signal has correct shape."""
        generator = SignalGenerator(config)
        mixed = generator.generate_mixed_signal()
        
        assert mixed.shape == (10000,)
    
    def test_targets_generation(self, config):
        """Test targets are generated correctly."""
        generator = SignalGenerator(config)
        targets = generator.generate_all_targets()
        
        assert len(targets) == 4
        for i in range(4):
            assert len(targets[i]) == 10000
    
    def test_different_seeds_produce_different_noise(self):
        """Test that different seeds produce different noise."""
        config1 = SignalConfig(
            frequencies=[1.0],
            sampling_rate=100,
            duration=1.0,
            amplitude_range=(0.8, 1.2),
            phase_range=(0, 2*np.pi),
            seed=1
        )
        
        config2 = SignalConfig(
            frequencies=[1.0],
            sampling_rate=100,
            duration=1.0,
            amplitude_range=(0.8, 1.2),
            phase_range=(0, 2*np.pi),
            seed=2
        )
        
        gen1 = SignalGenerator(config1)
        gen2 = SignalGenerator(config2)
        
        mixed1 = gen1.generate_mixed_signal()
        mixed2 = gen2.generate_mixed_signal()
        
        # Signals should be different due to different noise
        assert not np.allclose(mixed1, mixed2)


class TestFrequencyExtractionDataset:
    """Test FrequencyExtractionDataset class."""
    
    @pytest.fixture
    def generator(self):
        """Create a test generator."""
        config = SignalConfig(
            frequencies=[1.0, 3.0],
            sampling_rate=100,
            duration=1.0,
            amplitude_range=(0.8, 1.2),
            phase_range=(0, 2*np.pi),
            seed=42
        )
        return SignalGenerator(config)
    
    def test_dataset_length(self, generator):
        """Test dataset returns correct length."""
        dataset = FrequencyExtractionDataset(generator, normalize=False)
        
        # Should be num_time_samples * num_frequencies
        assert len(dataset) == 100 * 2  # 100 samples, 2 frequencies
    
    def test_dataset_item_shape(self, generator):
        """Test dataset items have correct shapes."""
        dataset = FrequencyExtractionDataset(generator, normalize=False)
        
        input_tensor, target_tensor = dataset[0]
        
        assert input_tensor.shape == (3,)  # [S[t], C1, C2]
        assert target_tensor.shape == (1,)
    
    def test_one_hot_encoding(self, generator):
        """Test one-hot encoding is correct."""
        dataset = FrequencyExtractionDataset(generator, normalize=False)
        
        # First 100 samples should have C = [1, 0]
        input_tensor, _ = dataset[0]
        assert input_tensor[1] == 1.0
        assert input_tensor[2] == 0.0
        
        # Next 100 samples should have C = [0, 1]
        input_tensor, _ = dataset[100]
        assert input_tensor[1] == 0.0
        assert input_tensor[2] == 1.0
    
    def test_normalization(self, generator):
        """Test signal normalization."""
        dataset = FrequencyExtractionDataset(generator, normalize=True)
        
        # Check that signal values are normalized
        # (approximately zero mean, unit std)
        all_signals = [dataset[i][0][0].item() for i in range(len(dataset))]
        mean = np.mean(all_signals)
        std = np.std(all_signals)
        
        assert abs(mean) < 0.1  # Close to zero
        assert abs(std - 1.0) < 0.2  # Close to 1.0


class TestFactoryFunctions:
    """Test factory functions."""
    
    def test_create_train_test_generators(self):
        """Test train/test generator creation."""
        train_gen, test_gen = create_train_test_generators(
            frequencies=[1.0, 3.0],
            sampling_rate=100,
            duration=1.0,
            train_seed=1,
            test_seed=2
        )
        
        assert isinstance(train_gen, SignalGenerator)
        assert isinstance(test_gen, SignalGenerator)
        
        # Generate signals to verify they're different
        train_signal, _ = train_gen.generate_complete_dataset()
        test_signal, _ = test_gen.generate_complete_dataset()
        
        assert not np.allclose(train_signal, test_signal)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

