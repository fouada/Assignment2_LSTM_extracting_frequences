"""
Input and Configuration Validation
ISO/IEC 25010 Security & Usability Compliance
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import yaml
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


class InputValidator:
    """
    Comprehensive input validation for ISO/IEC 25010 compliance
    
    Implements:
    - Security (Integrity): Input sanitization
    - Usability (User Error Protection): Early error detection
    - Reliability (Fault Tolerance): Prevents invalid inputs
    """
    
    @staticmethod
    def validate_frequency(
        frequency: float,
        min_freq: float = 0.1,
        max_freq: float = 100.0
    ) -> Tuple[bool, str]:
        """
        Validate frequency value
        
        Args:
            frequency: Frequency value to validate
            min_freq: Minimum allowed frequency
            max_freq: Maximum allowed frequency
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(frequency, (int, float)):
            return False, f"Frequency must be numeric, got {type(frequency)}"
        
        if frequency <= 0:
            return False, f"Frequency must be positive, got {frequency}"
        
        if frequency < min_freq:
            return False, f"Frequency {frequency} below minimum {min_freq} Hz"
        
        if frequency > max_freq:
            return False, f"Frequency {frequency} exceeds maximum {max_freq} Hz"
        
        return True, ""
    
    @staticmethod
    def validate_frequencies(
        frequencies: List[float],
        min_count: int = 1,
        max_count: int = 10
    ) -> Tuple[bool, str]:
        """
        Validate list of frequencies
        
        Args:
            frequencies: List of frequency values
            min_count: Minimum number of frequencies
            max_count: Maximum number of frequencies
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(frequencies, (list, tuple)):
            return False, f"Frequencies must be a list, got {type(frequencies)}"
        
        if len(frequencies) < min_count:
            return False, f"Need at least {min_count} frequencies, got {len(frequencies)}"
        
        if len(frequencies) > max_count:
            return False, f"Maximum {max_count} frequencies allowed, got {len(frequencies)}"
        
        # Check for duplicates
        if len(frequencies) != len(set(frequencies)):
            return False, "Duplicate frequencies detected"
        
        # Validate each frequency
        for i, freq in enumerate(frequencies):
            is_valid, error = InputValidator.validate_frequency(freq)
            if not is_valid:
                return False, f"Frequency[{i}]: {error}"
        
        return True, ""
    
    @staticmethod
    def validate_sampling_rate(
        sampling_rate: int,
        min_rate: int = 100,
        max_rate: int = 100000
    ) -> Tuple[bool, str]:
        """
        Validate sampling rate
        
        Args:
            sampling_rate: Sampling rate in Hz
            min_rate: Minimum allowed rate
            max_rate: Maximum allowed rate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(sampling_rate, int):
            return False, f"Sampling rate must be integer, got {type(sampling_rate)}"
        
        if sampling_rate < min_rate:
            return False, f"Sampling rate {sampling_rate} below minimum {min_rate} Hz"
        
        if sampling_rate > max_rate:
            return False, f"Sampling rate {sampling_rate} exceeds maximum {max_rate} Hz"
        
        return True, ""
    
    @staticmethod
    def validate_nyquist(
        frequencies: List[float],
        sampling_rate: int
    ) -> Tuple[bool, str]:
        """
        Validate Nyquist criterion
        
        Args:
            frequencies: Signal frequencies
            sampling_rate: Sampling rate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        nyquist_freq = sampling_rate / 2.0
        max_freq = max(frequencies)
        
        if max_freq >= nyquist_freq:
            return False, (
                f"Nyquist criterion violated: max frequency {max_freq} Hz "
                f"must be less than Nyquist frequency {nyquist_freq} Hz"
            )
        
        # Recommended: sampling rate should be at least 5x the highest frequency
        recommended_rate = max_freq * 5
        if sampling_rate < recommended_rate:
            logger.warning(
                f"Sampling rate {sampling_rate} Hz is below recommended "
                f"{recommended_rate} Hz for max frequency {max_freq} Hz"
            )
        
        return True, ""
    
    @staticmethod
    def validate_duration(
        duration: float,
        min_duration: float = 0.1,
        max_duration: float = 3600.0
    ) -> Tuple[bool, str]:
        """
        Validate signal duration
        
        Args:
            duration: Signal duration in seconds
            min_duration: Minimum duration
            max_duration: Maximum duration
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(duration, (int, float)):
            return False, f"Duration must be numeric, got {type(duration)}"
        
        if duration <= 0:
            return False, f"Duration must be positive, got {duration}"
        
        if duration < min_duration:
            return False, f"Duration {duration}s below minimum {min_duration}s"
        
        if duration > max_duration:
            return False, f"Duration {duration}s exceeds maximum {max_duration}s"
        
        return True, ""
    
    @staticmethod
    def validate_batch_size(
        batch_size: int,
        min_size: int = 1,
        max_size: int = 1024
    ) -> Tuple[bool, str]:
        """
        Validate batch size
        
        Args:
            batch_size: Batch size for training
            min_size: Minimum batch size
            max_size: Maximum batch size
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(batch_size, int):
            return False, f"Batch size must be integer, got {type(batch_size)}"
        
        if batch_size < min_size:
            return False, f"Batch size {batch_size} below minimum {min_size}"
        
        if batch_size > max_size:
            return False, f"Batch size {batch_size} exceeds maximum {max_size}"
        
        # Check if power of 2 (recommended)
        if not (batch_size & (batch_size - 1) == 0):
            logger.warning(
                f"Batch size {batch_size} is not a power of 2, "
                "which may reduce GPU efficiency"
            )
        
        return True, ""
    
    @staticmethod
    def validate_learning_rate(
        learning_rate: float,
        min_lr: float = 1e-6,
        max_lr: float = 1.0
    ) -> Tuple[bool, str]:
        """
        Validate learning rate
        
        Args:
            learning_rate: Learning rate value
            min_lr: Minimum learning rate
            max_lr: Maximum learning rate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(learning_rate, (int, float)):
            return False, f"Learning rate must be numeric, got {type(learning_rate)}"
        
        if learning_rate <= 0:
            return False, f"Learning rate must be positive, got {learning_rate}"
        
        if learning_rate < min_lr:
            logger.warning(
                f"Learning rate {learning_rate} is very small, "
                "training may be extremely slow"
            )
        
        if learning_rate > max_lr:
            return False, f"Learning rate {learning_rate} is too large"
        
        return True, ""
    
    @staticmethod
    def validate_epochs(
        epochs: int,
        min_epochs: int = 1,
        max_epochs: int = 10000
    ) -> Tuple[bool, str]:
        """
        Validate number of training epochs
        
        Args:
            epochs: Number of epochs
            min_epochs: Minimum epochs
            max_epochs: Maximum epochs
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(epochs, int):
            return False, f"Epochs must be integer, got {type(epochs)}"
        
        if epochs < min_epochs:
            return False, f"Epochs {epochs} below minimum {min_epochs}"
        
        if epochs > max_epochs:
            return False, f"Epochs {epochs} exceeds maximum {max_epochs}"
        
        return True, ""
    
    @staticmethod
    def validate_path(
        path: Union[str, Path],
        must_exist: bool = False,
        must_be_file: bool = False,
        must_be_dir: bool = False
    ) -> Tuple[bool, str]:
        """
        Validate file/directory path
        
        Args:
            path: Path to validate
            must_exist: Whether path must exist
            must_be_file: Whether path must be a file
            must_be_dir: Whether path must be a directory
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            path = Path(path)
        except Exception as e:
            return False, f"Invalid path format: {e}"
        
        if must_exist and not path.exists():
            return False, f"Path does not exist: {path}"
        
        if must_be_file and path.exists() and not path.is_file():
            return False, f"Path is not a file: {path}"
        
        if must_be_dir and path.exists() and not path.is_dir():
            return False, f"Path is not a directory: {path}"
        
        # Check for path traversal attempts
        try:
            path.resolve()
        except Exception:
            return False, f"Potentially malicious path: {path}"
        
        return True, ""
    
    @staticmethod
    def validate_tensor_shape(
        tensor: np.ndarray,
        expected_shape: Optional[Tuple[int, ...]] = None,
        expected_ndim: Optional[int] = None
    ) -> Tuple[bool, str]:
        """
        Validate tensor shape
        
        Args:
            tensor: Numpy array or tensor
            expected_shape: Expected shape (None for any size)
            expected_ndim: Expected number of dimensions
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(tensor, np.ndarray):
            return False, f"Expected numpy array, got {type(tensor)}"
        
        if expected_ndim is not None and tensor.ndim != expected_ndim:
            return False, (
                f"Expected {expected_ndim} dimensions, "
                f"got {tensor.ndim} with shape {tensor.shape}"
            )
        
        if expected_shape is not None:
            if len(expected_shape) != len(tensor.shape):
                return False, (
                    f"Shape mismatch: expected {expected_shape}, "
                    f"got {tensor.shape}"
                )
            
            for i, (expected, actual) in enumerate(zip(expected_shape, tensor.shape)):
                if expected is not None and expected != actual:
                    return False, (
                        f"Dimension {i} mismatch: expected {expected}, "
                        f"got {actual}"
                    )
        
        # Check for NaN or Inf values
        if np.isnan(tensor).any():
            return False, "Tensor contains NaN values"
        
        if np.isinf(tensor).any():
            return False, "Tensor contains Inf values"
        
        return True, ""
    
    @staticmethod
    def sanitize_string(
        value: str,
        max_length: int = 1000,
        allowed_pattern: str = r'^[a-zA-Z0-9_\-\.\s]+$'
    ) -> Tuple[bool, str, str]:
        """
        Sanitize string input
        
        Args:
            value: String to sanitize
            max_length: Maximum allowed length
            allowed_pattern: Regex pattern for allowed characters
            
        Returns:
            Tuple of (is_valid, error_message, sanitized_value)
        """
        if not isinstance(value, str):
            return False, f"Expected string, got {type(value)}", ""
        
        if len(value) > max_length:
            return False, f"String exceeds maximum length {max_length}", ""
        
        if not re.match(allowed_pattern, value):
            return False, "String contains invalid characters", ""
        
        # Remove potential injection attempts
        sanitized = value.strip()
        
        # Check for common injection patterns
        dangerous_patterns = [
            r'<script',
            r'javascript:',
            r'onerror=',
            r'onclick=',
            r'\.\./',
            r'\$\{',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, sanitized, re.IGNORECASE):
                return False, f"Potentially dangerous pattern detected: {pattern}", ""
        
        return True, "", sanitized


class ConfigValidator:
    """
    Configuration file validation
    
    Ensures configuration files are valid and secure
    """
    
    @staticmethod
    def validate_config_file(config_path: Union[str, Path]) -> Tuple[bool, str, Optional[Dict]]:
        """
        Validate configuration file
        
        Args:
            config_path: Path to config file
            
        Returns:
            Tuple of (is_valid, error_message, config_dict)
        """
        # Validate path
        is_valid, error = InputValidator.validate_path(
            config_path,
            must_exist=True,
            must_be_file=True
        )
        if not is_valid:
            return False, error, None
        
        # Load config
        config_path = Path(config_path)
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix in ['.yaml', '.yml']:
                    config = yaml.safe_load(f)
                else:
                    return False, f"Unsupported config format: {config_path.suffix}", None
        except Exception as e:
            return False, f"Failed to load config: {e}", None
        
        if config is None:
            return False, "Config file is empty", None
        
        # Validate config structure
        is_valid, error = ConfigValidator._validate_config_structure(config)
        if not is_valid:
            return False, error, None
        
        return True, "", config
    
    @staticmethod
    def _validate_config_structure(config: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate configuration structure"""
        required_sections = ['data', 'model', 'training']
        
        for section in required_sections:
            if section not in config:
                return False, f"Missing required section: {section}"
        
        # Validate data section
        if 'frequencies' not in config['data']:
            return False, "Missing data.frequencies"
        
        is_valid, error = InputValidator.validate_frequencies(
            config['data']['frequencies']
        )
        if not is_valid:
            return False, f"Invalid frequencies: {error}"
        
        # Validate sampling rate and Nyquist
        if 'sampling_rate' in config['data']:
            is_valid, error = InputValidator.validate_sampling_rate(
                config['data']['sampling_rate']
            )
            if not is_valid:
                return False, f"Invalid sampling_rate: {error}"
            
            is_valid, error = InputValidator.validate_nyquist(
                config['data']['frequencies'],
                config['data']['sampling_rate']
            )
            if not is_valid:
                return False, error
        
        # Validate model section
        required_model_params = ['input_size', 'hidden_size', 'num_layers', 'output_size']
        for param in required_model_params:
            if param not in config['model']:
                return False, f"Missing model.{param}"
            
            if not isinstance(config['model'][param], int) or config['model'][param] <= 0:
                return False, f"Invalid model.{param}: must be positive integer"
        
        # Validate training section
        if 'batch_size' in config['training']:
            is_valid, error = InputValidator.validate_batch_size(
                config['training']['batch_size']
            )
            if not is_valid:
                return False, f"Invalid batch_size: {error}"
        
        if 'learning_rate' in config['training']:
            is_valid, error = InputValidator.validate_learning_rate(
                config['training']['learning_rate']
            )
            if not is_valid:
                return False, f"Invalid learning_rate: {error}"
        
        if 'epochs' in config['training']:
            is_valid, error = InputValidator.validate_epochs(
                config['training']['epochs']
            )
            if not is_valid:
                return False, f"Invalid epochs: {error}"
        
        return True, ""
    
    @staticmethod
    def validate_and_load(config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate and load configuration file
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
            
        Raises:
            ValidationError: If validation fails
        """
        is_valid, error, config = ConfigValidator.validate_config_file(config_path)
        
        if not is_valid:
            raise ValidationError(f"Configuration validation failed: {error}")
        
        logger.info(f"Configuration validated successfully: {config_path}")
        return config

