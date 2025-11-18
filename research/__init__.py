"""
Research Module for LSTM Frequency Extraction
In-depth analysis tools for systematic research.

This module provides:
1. Sensitivity analysis - systematic hyperparameter search
2. Comparative analysis - architecture and configuration comparisons
3. Mathematical framework - theoretical proofs and bounds
4. Automated research pipeline - orchestrated experiments

Author: Research Team
Date: November 2025
"""

__version__ = "1.0.0"
__author__ = "Research Team"

from .sensitivity_analysis import (
    SensitivityAnalyzer,
    SensitivityConfig,
    ExperimentResult,
    create_default_sensitivity_config
)

from .comparative_analysis import (
    ComparativeAnalyzer,
    ComparisonResult,
    GRUExtractor,
    SimpleRNN
)

__all__ = [
    'SensitivityAnalyzer',
    'SensitivityConfig',
    'ExperimentResult',
    'create_default_sensitivity_config',
    'ComparativeAnalyzer',
    'ComparisonResult',
    'GRUExtractor',
    'SimpleRNN'
]

