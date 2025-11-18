"""
Visualization Module
Professional plotting utilities for frequency extraction analysis.

Author: Professional ML Engineering Team
Date: 2025
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set professional style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12


class FrequencyExtractionVisualizer:
    """
    Professional visualization utilities for frequency extraction results.
    """
    
    def __init__(self, save_dir: Optional[Path] = None):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Plots will be saved to: {save_dir}")
    
    def plot_single_frequency_comparison(
        self,
        time: np.ndarray,
        mixed_signal: np.ndarray,
        target: np.ndarray,
        prediction: np.ndarray,
        frequency: float,
        freq_idx: int,
        save_name: Optional[str] = None
    ):
        """
        Plot Graph 1: Comparison for a single frequency.
        
        Shows:
        - Target (pure sine, line)
        - LSTM output (dots)
        - Mixed signal (gray background)
        
        Args:
            time: Time vector
            mixed_signal: Mixed noisy input signal
            target: Ground truth pure sine
            prediction: LSTM predictions
            frequency: Frequency value in Hz
            freq_idx: Frequency index
            save_name: Optional filename to save plot
        """
        fig, ax = plt.subplots(figsize=(16, 6))
        
        # Plot mixed signal as background
        ax.plot(time, mixed_signal, color='lightgray', alpha=0.5, 
                linewidth=1, label='Mixed Signal (Noisy Input)', zorder=1)
        
        # Plot target (pure)
        ax.plot(time, target, 'b-', linewidth=2.5, 
                label='Target (Pure Sine)', zorder=3, alpha=0.8)
        
        # Plot LSTM output
        ax.scatter(time, prediction, c='red', s=10, alpha=0.6,
                  label='LSTM Output', zorder=2)
        
        ax.set_xlabel('Time (seconds)', fontweight='bold')
        ax.set_ylabel('Amplitude', fontweight='bold')
        ax.set_title(f'Frequency Extraction: f{freq_idx+1} = {frequency} Hz', 
                    fontweight='bold', fontsize=18)
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Add MSE annotation
        mse = np.mean((target - prediction) ** 2)
        ax.text(0.02, 0.98, f'MSE: {mse:.6f}', 
                transform=ax.transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            save_path = self.save_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved: {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_all_frequencies_grid(
        self,
        time: np.ndarray,
        targets: Dict[int, np.ndarray],
        predictions: Dict[int, np.ndarray],
        frequencies: List[float],
        mixed_signal: Optional[np.ndarray] = None,
        save_name: Optional[str] = None
    ):
        """
        Plot Graph 2: All 4 frequencies in 2×2 grid.
        
        Args:
            time: Time vector
            targets: Dictionary mapping freq_idx to target signals
            predictions: Dictionary mapping freq_idx to predictions
            frequencies: List of frequency values
            mixed_signal: Optional mixed signal to show in background
            save_name: Optional filename to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (freq, ax) in enumerate(zip(frequencies, axes)):
            target = targets[idx]
            prediction = predictions[idx]
            
            # Plot mixed signal if provided
            if mixed_signal is not None:
                ax.plot(time, mixed_signal, color='lightgray', 
                       alpha=0.3, linewidth=0.5, label='Mixed Signal')
            
            # Plot target
            ax.plot(time, target, 'b-', linewidth=2, 
                   label='Target', alpha=0.8)
            
            # Plot prediction
            ax.plot(time, prediction, 'r--', linewidth=1.5, 
                   label='LSTM Output', alpha=0.7)
            
            # Calculate metrics
            mse = np.mean((target - prediction) ** 2)
            r2 = 1 - (np.sum((target - prediction) ** 2) / 
                     np.sum((target - np.mean(target)) ** 2))
            
            ax.set_xlabel('Time (seconds)', fontweight='bold')
            ax.set_ylabel('Amplitude', fontweight='bold')
            ax.set_title(f'f{idx+1} = {freq} Hz\nMSE: {mse:.6f}, R²: {r2:.4f}', 
                        fontweight='bold')
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('All Frequency Extractions (Test Set)', 
                    fontsize=20, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_name and self.save_dir:
            save_path = self.save_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved: {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        save_name: Optional[str] = None
    ):
        """
        Plot training history.
        
        Args:
            history: Dictionary with 'train_loss', 'val_loss', 'learning_rate'
            save_name: Optional filename to save plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Plot losses
        ax1.plot(epochs, history['train_loss'], 'b-o', 
                label='Training Loss', linewidth=2, markersize=4)
        if 'val_loss' in history and history['val_loss']:
            ax1.plot(epochs, history['val_loss'], 'r-s', 
                    label='Validation Loss', linewidth=2, markersize=4)
        ax1.set_xlabel('Epoch', fontweight='bold')
        ax1.set_ylabel('MSE Loss', fontweight='bold')
        ax1.set_title('Training and Validation Loss', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot learning rate
        ax2.plot(epochs, history['learning_rate'], 'g-^', 
                linewidth=2, markersize=4)
        ax2.set_xlabel('Epoch', fontweight='bold')
        ax2.set_ylabel('Learning Rate', fontweight='bold')
        ax2.set_title('Learning Rate Schedule', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            save_path = self.save_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved: {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_error_distribution(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        save_name: Optional[str] = None
    ):
        """
        Plot error distribution analysis.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            save_name: Optional filename to save plot
        """
        errors = predictions - targets
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Error histogram
        axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Prediction Error', fontweight='bold')
        axes[0].set_ylabel('Frequency', fontweight='bold')
        axes[0].set_title('Error Distribution', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Prediction vs Target scatter
        axes[1].scatter(targets, predictions, alpha=0.3, s=10)
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        axes[1].plot([min_val, max_val], [min_val, max_val], 
                     'r--', linewidth=2, label='Perfect Prediction')
        axes[1].set_xlabel('Target', fontweight='bold')
        axes[1].set_ylabel('Prediction', fontweight='bold')
        axes[1].set_title('Prediction vs Target', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Residual plot
        axes[2].scatter(targets, errors, alpha=0.3, s=10)
        axes[2].axhline(0, color='red', linestyle='--', linewidth=2)
        axes[2].set_xlabel('Target', fontweight='bold')
        axes[2].set_ylabel('Residual (Prediction - Target)', fontweight='bold')
        axes[2].set_title('Residual Plot', fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            save_path = self.save_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved: {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_metrics_comparison(
        self,
        train_metrics: Dict,
        test_metrics: Dict,
        save_name: Optional[str] = None
    ):
        """
        Plot train vs test metrics comparison.
        
        Args:
            train_metrics: Training metrics
            test_metrics: Test metrics
            save_name: Optional filename to save plot
        """
        metrics_names = ['mse', 'mae', 'r2_score']
        train_values = [train_metrics['overall'][m] for m in metrics_names]
        test_values = [test_metrics['overall'][m] for m in metrics_names]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width/2, train_values, width, label='Train', 
                      color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, test_values, width, label='Test', 
                      color='coral', alpha=0.8)
        
        ax.set_xlabel('Metrics', fontweight='bold')
        ax.set_ylabel('Value', fontweight='bold')
        ax.set_title('Train vs Test Metrics Comparison', 
                     fontweight='bold', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(['MSE', 'MAE', 'R² Score'])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            save_path = self.save_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved: {save_path}")
        
        plt.show()
        plt.close()


def create_all_visualizations(
    test_dataset,
    predictions_dict: Dict[int, np.ndarray],
    frequencies: List[float],
    history: Dict[str, List[float]],
    train_metrics: Dict,
    test_metrics: Dict,
    save_dir: Path
):
    """
    Create all required visualizations for the assignment.
    
    Args:
        test_dataset: Test dataset
        predictions_dict: Dictionary mapping freq_idx to predictions
        frequencies: List of frequencies
        history: Training history
        train_metrics: Training metrics
        test_metrics: Test metrics
        save_dir: Directory to save plots
    """
    visualizer = FrequencyExtractionVisualizer(save_dir)
    
    logger.info("Creating all visualizations...")
    
    # Get time and signals
    time = test_dataset.generator.time_vector
    mixed_signal = test_dataset.mixed_signal
    
    # Graph 1: Single frequency comparison (f2 = 3Hz as example)
    freq_idx = 1  # f2
    target_f2 = test_dataset.targets[freq_idx]
    pred_f2 = predictions_dict[freq_idx]
    
    visualizer.plot_single_frequency_comparison(
        time, mixed_signal, target_f2, pred_f2,
        frequencies[freq_idx], freq_idx,
        save_name='graph1_single_frequency_f2'
    )
    
    # Graph 2: All frequencies grid
    targets_dict = test_dataset.targets
    visualizer.plot_all_frequencies_grid(
        time, targets_dict, predictions_dict, frequencies,
        mixed_signal=mixed_signal,
        save_name='graph2_all_frequencies'
    )
    
    # Training history
    visualizer.plot_training_history(history, save_name='training_history')
    
    # Error distribution
    all_predictions = np.concatenate([predictions_dict[i] for i in range(len(frequencies))])
    all_targets = np.concatenate([targets_dict[i] for i in range(len(frequencies))])
    visualizer.plot_error_distribution(
        all_predictions, all_targets,
        save_name='error_distribution'
    )
    
    # Metrics comparison
    visualizer.plot_metrics_comparison(
        train_metrics, test_metrics,
        save_name='metrics_comparison'
    )
    
    logger.info("All visualizations created successfully!")

