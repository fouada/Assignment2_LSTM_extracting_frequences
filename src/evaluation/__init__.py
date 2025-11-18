"""Evaluation and metrics module."""

from .metrics import (
    FrequencyExtractionMetrics,
    evaluate_model,
    compare_train_test_performance
)

__all__ = [
    "FrequencyExtractionMetrics",
    "evaluate_model",
    "compare_train_test_performance",
]

