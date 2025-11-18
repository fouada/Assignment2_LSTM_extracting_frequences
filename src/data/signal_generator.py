"""
Signal Generator Module
Professional implementation of noisy mixed signal generation for LSTM training.

Author: Professional ML Engineering Team
Date: 2025
"""

import numpy as np
from typing import Tuple, List, Dict
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SignalConfig:
    """Configuration for signal generation."""
    frequencies: List[float]
    sampling_rate: int
    duration: float
    amplitude_range: Tuple[float, float]
    phase_range: Tuple[float, float]
    seed: int


class SignalGenerator:
    """
    Professional signal generator for creating noisy mixed signals.
    
    This class implements the signal generation logic according to the
    assignment specifications with proper encapsulation and error handling.
    """
    
    def __init__(self, config: SignalConfig):
        """
        Initialize the signal generator.
        
        Args:
            config: SignalConfig object with generation parameters
        """
        self.config = config
        self.frequencies = np.array(config.frequencies)
        self.num_frequencies = len(self.frequencies)
        self.num_samples = int(config.sampling_rate * config.duration)
        self.time_vector = np.linspace(0, config.duration, self.num_samples)
        
        # Set random seed for reproducibility
        np.random.seed(config.seed)
        
        logger.info(f"SignalGenerator initialized with seed={config.seed}")
        logger.info(f"Frequencies: {self.frequencies} Hz")
        logger.info(f"Total samples: {self.num_samples}")
    
    def generate_noisy_sine(
        self, 
        frequency: float, 
        time: np.ndarray
    ) -> np.ndarray:
        """
        Generate a single noisy sine wave with random amplitude and phase.
        
        According to specs:
        - Amplitude: Uniform(0.8, 1.2) at each sample
        - Phase: Uniform(0, 2π) at each sample
        
        Args:
            frequency: Frequency in Hz
            time: Time vector
            
        Returns:
            Noisy sine wave array
        """
        num_samples = len(time)
        
        # Generate random amplitude and phase for EACH sample
        amplitudes = np.random.uniform(
            self.config.amplitude_range[0],
            self.config.amplitude_range[1],
            size=num_samples
        )
        
        phases = np.random.uniform(
            self.config.phase_range[0],
            self.config.phase_range[1],
            size=num_samples
        )
        
        # Create noisy sine wave: A(t) * sin(2π*f*t + φ(t))
        noisy_sine = amplitudes * np.sin(2 * np.pi * frequency * time + phases)
        
        return noisy_sine
    
    def generate_pure_sine(
        self, 
        frequency: float, 
        time: np.ndarray
    ) -> np.ndarray:
        """
        Generate a pure sine wave (target/ground truth).
        
        According to specs: sin(2π*f*t)
        No amplitude or phase variation!
        
        Args:
            frequency: Frequency in Hz
            time: Time vector
            
        Returns:
            Pure sine wave array
        """
        return np.sin(2 * np.pi * frequency * time)
    
    def generate_mixed_signal(self) -> np.ndarray:
        """
        Generate the mixed noisy signal S(t).
        
        According to specs:
        S(t) = (1/4) * Σ(i=1 to 4) Sinus_noisy_i(t)
        
        Returns:
            Mixed signal array of shape (num_samples,)
        """
        # Generate all noisy sine waves
        noisy_sines = np.zeros((self.num_frequencies, self.num_samples))
        
        for i, freq in enumerate(self.frequencies):
            noisy_sines[i] = self.generate_noisy_sine(freq, self.time_vector)
            logger.debug(f"Generated noisy sine for f{i+1}={freq}Hz")
        
        # Normalize sum
        mixed_signal = np.mean(noisy_sines, axis=0)  # Same as sum / 4
        
        logger.info(f"Mixed signal generated: shape={mixed_signal.shape}, "
                   f"range=[{mixed_signal.min():.3f}, {mixed_signal.max():.3f}]")
        
        return mixed_signal
    
    def generate_all_targets(self) -> Dict[int, np.ndarray]:
        """
        Generate all pure target signals.
        
        Returns:
            Dictionary mapping frequency index to pure sine wave
        """
        targets = {}
        
        for i, freq in enumerate(self.frequencies):
            targets[i] = self.generate_pure_sine(freq, self.time_vector)
            logger.debug(f"Generated pure target for f{i+1}={freq}Hz")
        
        return targets
    
    def generate_complete_dataset(self) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """
        Generate complete dataset: mixed signal + all targets.
        
        Returns:
            Tuple of (mixed_signal, targets_dict)
        """
        logger.info("Generating complete dataset...")
        
        mixed_signal = self.generate_mixed_signal()
        targets = self.generate_all_targets()
        
        logger.info("Dataset generation complete!")
        
        return mixed_signal, targets
    
    def get_statistics(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Calculate signal statistics for analysis.
        
        Args:
            signal: Input signal array
            
        Returns:
            Dictionary of statistics
        """
        return {
            'mean': float(np.mean(signal)),
            'std': float(np.std(signal)),
            'min': float(np.min(signal)),
            'max': float(np.max(signal)),
            'rms': float(np.sqrt(np.mean(signal**2)))
        }


def create_train_test_generators(
    frequencies: List[float],
    sampling_rate: int,
    duration: float,
    amplitude_range: Tuple[float, float] = (0.8, 1.2),
    phase_range: Tuple[float, float] = (0, 2*np.pi),
    train_seed: int = 1,
    test_seed: int = 2
) -> Tuple[SignalGenerator, SignalGenerator]:
    """
    Factory function to create train and test signal generators.
    
    Args:
        frequencies: List of frequencies in Hz
        sampling_rate: Sampling rate in Hz
        duration: Duration in seconds
        amplitude_range: Range for random amplitudes
        phase_range: Range for random phases
        train_seed: Random seed for training data
        test_seed: Random seed for test data
        
    Returns:
        Tuple of (train_generator, test_generator)
    """
    train_config = SignalConfig(
        frequencies=frequencies,
        sampling_rate=sampling_rate,
        duration=duration,
        amplitude_range=amplitude_range,
        phase_range=phase_range,
        seed=train_seed
    )
    
    test_config = SignalConfig(
        frequencies=frequencies,
        sampling_rate=sampling_rate,
        duration=duration,
        amplitude_range=amplitude_range,
        phase_range=phase_range,
        seed=test_seed
    )
    
    train_generator = SignalGenerator(train_config)
    test_generator = SignalGenerator(test_config)
    
    logger.info("Train and test generators created successfully!")
    
    return train_generator, test_generator

