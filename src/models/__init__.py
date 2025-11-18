"""LSTM model architecture module with innovative variants."""

# Base model
from .lstm_extractor import StatefulLSTMExtractor, create_model

# Innovative models (NEW!)
from .attention_lstm import AttentionLSTMExtractor, create_attention_model
from .bayesian_lstm import BayesianLSTMExtractor, create_bayesian_model
from .hybrid_lstm import HybridLSTMExtractor, create_hybrid_model

__all__ = [
    # Base models
    "StatefulLSTMExtractor",
    "create_model",
    # Innovative models
    "AttentionLSTMExtractor",
    "create_attention_model",
    "BayesianLSTMExtractor",
    "create_bayesian_model",
    "HybridLSTMExtractor",
    "create_hybrid_model",
]

