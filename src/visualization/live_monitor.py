"""
Live Training Monitor for Dashboard Integration
Real-time data streaming for interactive visualization

Author: Professional ML Engineering Team
Date: 2025
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading
import time
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class LiveTrainingMonitor:
    """
    Monitor training progress and export data for dashboard consumption.
    
    Features:
    - Real-time metric tracking
    - JSON export for dashboard
    - Thread-safe data updates
    - Automatic file synchronization
    """
    
    def __init__(self, experiment_dir: Path, update_interval: float = 1.0):
        """
        Initialize live monitor.
        
        Args:
            experiment_dir: Experiment directory path
            update_interval: Update interval in seconds
        """
        self.experiment_dir = experiment_dir
        self.update_interval = update_interval
        
        # Data storage
        self.training_history = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'gradient_norm': [],
            'epoch_time': [],
            'timestamp': [],
        }
        
        self.current_epoch_data = {}
        self.test_results = {}
        
        # Thread safety
        self.lock = threading.Lock()
        self.running = False
        self.update_thread = None
        
        # File paths
        self.history_file = experiment_dir / 'live_training_history.json'
        self.results_file = experiment_dir / 'live_test_results.json'
        self.status_file = experiment_dir / 'training_status.json'
        
        logger.info(f"Live monitor initialized for {experiment_dir}")
    
    def start(self):
        """Start the live monitoring thread."""
        if self.running:
            logger.warning("Monitor already running")
            return
        
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        logger.info("Live monitor started")
    
    def stop(self):
        """Stop the live monitoring thread."""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
        logger.info("Live monitor stopped")
    
    def _update_loop(self):
        """Background thread for periodic file updates."""
        while self.running:
            try:
                self._save_data()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in monitor update loop: {e}")
    
    def _save_data(self):
        """Save current data to files (thread-safe)."""
        with self.lock:
            try:
                # Save training history
                with open(self.history_file, 'w') as f:
                    json.dump(self.training_history, f, indent=2)
                
                # Save test results if available
                if self.test_results:
                    with open(self.results_file, 'w') as f:
                        json.dump(self.test_results, f, indent=2)
                
                # Save status
                status = {
                    'is_training': self.running,
                    'last_update': datetime.now().isoformat(),
                    'current_epoch': len(self.training_history['epochs']),
                    'total_epochs': self.current_epoch_data.get('total_epochs', 'unknown'),
                }
                with open(self.status_file, 'w') as f:
                    json.dump(status, f, indent=2)
                    
            except Exception as e:
                logger.error(f"Error saving monitor data: {e}")
    
    def update_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        gradient_norm: Optional[float] = None,
        epoch_time: Optional[float] = None,
        **kwargs
    ):
        """
        Update training data for current epoch.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss
            val_loss: Validation loss (optional)
            learning_rate: Current learning rate (optional)
            gradient_norm: Gradient norm (optional)
            epoch_time: Epoch duration in seconds (optional)
            **kwargs: Additional metrics
        """
        with self.lock:
            self.training_history['epochs'].append(epoch)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss if val_loss is not None else train_loss)
            self.training_history['learning_rate'].append(learning_rate if learning_rate is not None else 0.0)
            self.training_history['gradient_norm'].append(gradient_norm if gradient_norm is not None else 0.0)
            self.training_history['epoch_time'].append(epoch_time if epoch_time is not None else 0.0)
            self.training_history['timestamp'].append(datetime.now().isoformat())
            
            # Store any additional metrics
            for key, value in kwargs.items():
                if key not in self.training_history:
                    self.training_history[key] = []
                self.training_history[key].append(value)
            
            logger.debug(f"Updated epoch {epoch}: train_loss={train_loss:.6f}")
    
    def update_test_results(
        self,
        overall_metrics: Dict[str, float],
        per_frequency_metrics: Optional[Dict[int, Dict[str, float]]] = None
    ):
        """
        Update test results.
        
        Args:
            overall_metrics: Overall test metrics
            per_frequency_metrics: Per-frequency metrics (optional)
        """
        with self.lock:
            self.test_results = {
                'overall': overall_metrics,
                'per_frequency': per_frequency_metrics if per_frequency_metrics else {},
                'timestamp': datetime.now().isoformat()
            }
            logger.info("Test results updated")
    
    def update_predictions(
        self,
        frequency_idx: int,
        time: np.ndarray,
        target: np.ndarray,
        prediction: np.ndarray,
        mixed_signal: Optional[np.ndarray] = None
    ):
        """
        Update prediction data for a specific frequency.
        
        Args:
            frequency_idx: Frequency index (0-3)
            time: Time vector
            target: Target signal
            prediction: Predicted signal
            mixed_signal: Mixed input signal (optional)
        """
        with self.lock:
            # Save prediction data
            pred_file = self.experiment_dir / f'predictions_f{frequency_idx}.npz'
            np.savez(
                pred_file,
                time=time,
                target=target,
                prediction=prediction,
                mixed_signal=mixed_signal if mixed_signal is not None else np.array([])
            )
            logger.debug(f"Saved predictions for frequency {frequency_idx}")
    
    def set_total_epochs(self, total_epochs: int):
        """Set total number of epochs for progress tracking."""
        with self.lock:
            self.current_epoch_data['total_epochs'] = total_epochs
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get current training summary.
        
        Returns:
            Dictionary with training summary
        """
        with self.lock:
            if not self.training_history['epochs']:
                return {'status': 'no_data'}
            
            latest_epoch = self.training_history['epochs'][-1]
            latest_train_loss = self.training_history['train_loss'][-1]
            latest_val_loss = self.training_history['val_loss'][-1]
            
            summary = {
                'status': 'training' if self.running else 'completed',
                'current_epoch': latest_epoch,
                'total_epochs': self.current_epoch_data.get('total_epochs', '?'),
                'latest_train_loss': latest_train_loss,
                'latest_val_loss': latest_val_loss,
                'best_train_loss': min(self.training_history['train_loss']),
                'best_val_loss': min(self.training_history['val_loss']),
                'total_time': sum(self.training_history['epoch_time']),
            }
            
            return summary
    
    def load_predictions(self, frequency_idx: int) -> Optional[Dict[str, np.ndarray]]:
        """
        Load prediction data for a specific frequency.
        
        Args:
            frequency_idx: Frequency index (0-3)
            
        Returns:
            Dictionary with prediction data or None
        """
        pred_file = self.experiment_dir / f'predictions_f{frequency_idx}.npz'
        
        if not pred_file.exists():
            return None
        
        try:
            data = np.load(pred_file)
            return {
                'time': data['time'],
                'target': data['target'],
                'prediction': data['prediction'],
                'mixed_signal': data['mixed_signal'] if 'mixed_signal' in data else None
            }
        except Exception as e:
            logger.error(f"Error loading predictions: {e}")
            return None


def create_live_monitor(experiment_dir: Path, auto_start: bool = True) -> LiveTrainingMonitor:
    """
    Create and optionally start a live training monitor.
    
    Args:
        experiment_dir: Experiment directory path
        auto_start: Whether to start monitoring immediately
        
    Returns:
        LiveTrainingMonitor instance
    """
    monitor = LiveTrainingMonitor(experiment_dir)
    
    if auto_start:
        monitor.start()
    
    return monitor


class DashboardDataExporter:
    """
    Export training data in dashboard-compatible format.
    
    This class provides utilities to convert training data into formats
    that can be easily consumed by the interactive dashboard.
    """
    
    @staticmethod
    def export_training_history(
        history: Dict[str, List],
        output_file: Path
    ):
        """
        Export training history to JSON.
        
        Args:
            history: Training history dictionary
            output_file: Output file path
        """
        try:
            with open(output_file, 'w') as f:
                json.dump(history, f, indent=2)
            logger.info(f"Exported training history to {output_file}")
        except Exception as e:
            logger.error(f"Error exporting training history: {e}")
    
    @staticmethod
    def export_test_results(
        test_results: Dict,
        output_file: Path
    ):
        """
        Export test results to JSON.
        
        Args:
            test_results: Test results dictionary
            output_file: Output file path
        """
        try:
            # Convert any numpy types to Python types
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                return obj
            
            serializable_results = convert_numpy(test_results)
            
            with open(output_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            logger.info(f"Exported test results to {output_file}")
        except Exception as e:
            logger.error(f"Error exporting test results: {e}")
    
    @staticmethod
    def export_predictions(
        predictions_dict: Dict[int, np.ndarray],
        targets_dict: Dict[int, np.ndarray],
        time: np.ndarray,
        mixed_signal: np.ndarray,
        output_dir: Path
    ):
        """
        Export prediction data for all frequencies.
        
        Args:
            predictions_dict: Dictionary of predictions per frequency
            targets_dict: Dictionary of targets per frequency
            time: Time vector
            mixed_signal: Mixed signal
            output_dir: Output directory
        """
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for freq_idx in predictions_dict.keys():
                output_file = output_dir / f'predictions_f{freq_idx}.npz'
                np.savez(
                    output_file,
                    time=time,
                    target=targets_dict[freq_idx],
                    prediction=predictions_dict[freq_idx],
                    mixed_signal=mixed_signal
                )
            
            logger.info(f"Exported predictions to {output_dir}")
        except Exception as e:
            logger.error(f"Error exporting predictions: {e}")

