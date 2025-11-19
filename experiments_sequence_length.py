"""
Comprehensive Experiment: Impact of Sequence Length (L) on LSTM Performance

This script systematically tests different L values and compares:
- Training convergence speed
- Final MSE performance
- Generalization capability
- Temporal learning efficiency
- Memory and computational requirements

Author: Professional ML Engineering Team
Date: 2025
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import time
import json
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data.signal_generator import create_train_test_generators
from src.data.dataset import create_dataloaders as create_stateful_dataloaders
from src.data.sequence_dataset import create_sequence_dataloaders
from src.models.lstm_extractor import StatefulLSTMExtractor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Store results from a single experiment."""
    sequence_length: int
    train_mse: float
    test_mse: float
    best_epoch: int
    total_epochs: int
    training_time: float
    convergence_speed: float  # Epochs to reach 90% of final performance
    parameters: int
    memory_mb: float
    samples_per_second: float
    final_train_loss: float
    final_test_loss: float
    train_history: List[float]
    test_history: List[float]


class SequenceLengthExperiment:
    """
    Orchestrate experiments with different sequence lengths.
    """
    
    def __init__(
        self,
        config_path: str = "config/config.yaml",
        output_dir: str = "experiments/sequence_length_comparison"
    ):
        """
        Initialize experiment.
        
        Args:
            config_path: Path to configuration file
            output_dir: Directory to save results
        """
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load base configuration
        with open(config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        # Setup device
        self.device = self._setup_device()
        
        # Create signal generators (same for all experiments)
        logger.info("Creating signal generators...")
        self.train_gen, self.test_gen = create_train_test_generators(
            frequencies=self.base_config['data']['frequencies'],
            sampling_rate=self.base_config['data']['sampling_rate'],
            duration=self.base_config['data']['duration'],
            amplitude_range=self.base_config['data']['amplitude_range'],
            phase_range=self.base_config['data']['phase_range'],
            train_seed=self.base_config['data']['train_seed'],
            test_seed=self.base_config['data']['test_seed']
        )
        
        self.results: Dict[int, ExperimentResult] = {}
        
        logger.info(f"Experiment initialized. Results will be saved to: {self.output_dir}")
    
    def _setup_device(self) -> torch.device:
        """Setup computation device."""
        device_config = self.base_config['compute']['device']
        
        if device_config == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        else:
            device = torch.device(device_config)
        
        logger.info(f"Using device: {device}")
        return device
    
    def run_single_experiment(
        self,
        sequence_length: int,
        epochs: int = None,
        batch_size: int = None
    ) -> ExperimentResult:
        """
        Run experiment with a specific sequence length.
        
        Args:
            sequence_length: L value to test
            epochs: Number of training epochs (default from config)
            batch_size: Batch size (default from config)
            
        Returns:
            ExperimentResult object
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Running Experiment: L = {sequence_length}")
        logger.info(f"{'='*80}\n")
        
        # Use config defaults if not specified
        epochs = epochs or self.base_config['training']['epochs']
        batch_size = batch_size or self.base_config['training']['batch_size']
        
        start_time = time.time()
        
        # Create appropriate data loaders
        if sequence_length == 1:
            # Use stateful loader for L=1
            logger.info("Using stateful data loader (L=1 mode)")
            train_loader, test_loader = create_stateful_dataloaders(
                self.train_gen,
                self.test_gen,
                batch_size=batch_size,
                normalize=True,
                device=str(self.device)
            )
            stateful_mode = True
        else:
            # Use sequence loader for L>1
            logger.info(f"Using sequence data loader (L={sequence_length} mode)")
            train_loader, test_loader = create_sequence_dataloaders(
                self.train_gen,
                self.test_gen,
                sequence_length=sequence_length,
                batch_size=batch_size,
                stride=None,  # Non-overlapping sequences
                normalize=True,
                shuffle_train=True,
                device=str(self.device)
            )
            stateful_mode = False
        
        # Create model
        model = StatefulLSTMExtractor(
            input_size=self.base_config['model']['input_size'],
            hidden_size=self.base_config['model']['hidden_size'],
            num_layers=self.base_config['model']['num_layers'],
            output_size=self.base_config['model']['output_size'],
            dropout=self.base_config['model']['dropout'],
            bidirectional=self.base_config['model']['bidirectional']
        ).to(self.device)
        
        # Count parameters
        num_params = model.count_parameters()
        
        # Setup optimizer and criterion
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(self.base_config['training']['learning_rate']),
            weight_decay=float(self.base_config['training']['weight_decay'])
        )
        
        criterion = nn.MSELoss()
        
        # Training loop
        train_losses = []
        test_losses = []
        best_test_loss = float('inf')
        best_epoch = 0
        patience = self.base_config['training']['early_stopping_patience']
        patience_counter = 0
        
        logger.info(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            num_batches = 0
            
            if stateful_mode:
                # L=1: Stateful training with state preservation
                for batch in train_loader:
                    # Reset state at start of new frequency
                    if batch['is_first_batch']:
                        model.reset_state()
                    
                    inputs = batch['input'].to(self.device)
                    targets = batch['target'].to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs, reset_state=False)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.base_config['training']['gradient_clip_value']
                    )
                    
                    optimizer.step()
                    
                    # Detach state to prevent memory issues
                    model.detach_state()
                    
                    train_loss += loss.item()
                    num_batches += 1
            else:
                # L>1: Sequence training
                for batch in train_loader:
                    inputs = batch['input'].to(self.device)  # (batch, seq_len, 5)
                    targets = batch['target'].to(self.device)  # (batch, seq_len, 1)
                    
                    optimizer.zero_grad()
                    model.reset_state()  # Reset for each sequence
                    outputs = model(inputs, reset_state=False)  # (batch, seq_len, 1)
                    
                    loss = criterion(outputs, targets)
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.base_config['training']['gradient_clip_value']
                    )
                    
                    optimizer.step()
                    
                    train_loss += loss.item()
                    num_batches += 1
            
            avg_train_loss = train_loss / num_batches
            train_losses.append(avg_train_loss)
            
            # Evaluation phase
            model.eval()
            test_loss = 0.0
            num_test_batches = 0
            
            with torch.no_grad():
                if stateful_mode:
                    for batch in test_loader:
                        if batch['is_first_batch']:
                            model.reset_state()
                        
                        inputs = batch['input'].to(self.device)
                        targets = batch['target'].to(self.device)
                        
                        outputs = model(inputs, reset_state=False)
                        loss = criterion(outputs, targets)
                        
                        test_loss += loss.item()
                        num_test_batches += 1
                else:
                    for batch in test_loader:
                        inputs = batch['input'].to(self.device)
                        targets = batch['target'].to(self.device)
                        
                        model.reset_state()
                        outputs = model(inputs, reset_state=False)
                        loss = criterion(outputs, targets)
                        
                        test_loss += loss.item()
                        num_test_batches += 1
            
            avg_test_loss = test_loss / num_test_batches
            test_losses.append(avg_test_loss)
            
            # Check for improvement
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                best_epoch = epoch
                patience_counter = 0
                
                # Save best model
                save_path = self.output_dir / f"best_model_L{sequence_length}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_loss': avg_test_loss,
                    'sequence_length': sequence_length
                }, save_path)
            else:
                patience_counter += 1
            
            # Logging
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}] "
                          f"Train Loss: {avg_train_loss:.6f} "
                          f"Test Loss: {avg_test_loss:.6f} "
                          f"Best: {best_test_loss:.6f} @ epoch {best_epoch+1}")
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        training_time = time.time() - start_time
        
        # Calculate convergence speed (epochs to reach 90% of final performance)
        target_loss = train_losses[-1] * 1.1  # 90% of final (lower is better)
        convergence_epoch = next((i for i, loss in enumerate(train_losses) if loss <= target_loss), len(train_losses))
        
        # Calculate memory usage (approximate)
        if self.device.type == 'cuda':
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            memory_mb = 0.0  # Not easily measurable for CPU/MPS
        
        # Calculate throughput
        total_samples = len(train_loader) * batch_size * (epoch + 1)
        samples_per_second = total_samples / training_time
        
        # Create result object
        result = ExperimentResult(
            sequence_length=sequence_length,
            train_mse=train_losses[-1],
            test_mse=test_losses[-1],
            best_epoch=best_epoch,
            total_epochs=epoch + 1,
            training_time=training_time,
            convergence_speed=convergence_epoch,
            parameters=num_params,
            memory_mb=memory_mb,
            samples_per_second=samples_per_second,
            final_train_loss=train_losses[-1],
            final_test_loss=test_losses[-1],
            train_history=train_losses,
            test_history=test_losses
        )
        
        logger.info(f"\nExperiment L={sequence_length} completed:")
        logger.info(f"  Final Train MSE: {result.train_mse:.6f}")
        logger.info(f"  Final Test MSE: {result.test_mse:.6f}")
        logger.info(f"  Best Test MSE: {best_test_loss:.6f} @ epoch {best_epoch+1}")
        logger.info(f"  Training time: {training_time:.2f}s")
        logger.info(f"  Convergence speed: {convergence_epoch} epochs")
        logger.info(f"  Throughput: {samples_per_second:.2f} samples/sec")
        
        return result
    
    def run_all_experiments(
        self,
        sequence_lengths: List[int] = [1, 10, 50, 100, 500],
        epochs: int = None
    ):
        """
        Run experiments for all specified sequence lengths.
        
        Args:
            sequence_lengths: List of L values to test
            epochs: Number of epochs (default from config)
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"STARTING COMPREHENSIVE SEQUENCE LENGTH EXPERIMENTS")
        logger.info(f"Testing L values: {sequence_lengths}")
        logger.info(f"{'='*80}\n")
        
        for L in sequence_lengths:
            try:
                result = self.run_single_experiment(L, epochs=epochs)
                self.results[L] = result
                
                # Save intermediate results
                self._save_results()
                
            except Exception as e:
                logger.error(f"Error running experiment with L={L}: {e}", exc_info=True)
                continue
        
        logger.info(f"\n{'='*80}")
        logger.info("ALL EXPERIMENTS COMPLETED")
        logger.info(f"{'='*80}\n")
        
        # Generate comparative analysis
        self.generate_comparative_analysis()
    
    def _save_results(self):
        """Save results to JSON file."""
        results_dict = {
            L: asdict(result) for L, result in self.results.items()
        }
        
        results_path = self.output_dir / "results_summary.json"
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
    
    def generate_comparative_analysis(self):
        """Generate comprehensive comparative analysis and visualizations."""
        if not self.results:
            logger.warning("No results to analyze")
            return
        
        logger.info("Generating comparative analysis...")
        
        # Create comparison plots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Training curves comparison
        ax1 = plt.subplot(2, 3, 1)
        for L, result in sorted(self.results.items()):
            ax1.plot(result.train_history, label=f'L={L}', linewidth=2, alpha=0.8)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Training Loss (MSE)', fontsize=12)
        ax1.set_title('Training Loss Convergence', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 2. Test curves comparison
        ax2 = plt.subplot(2, 3, 2)
        for L, result in sorted(self.results.items()):
            ax2.plot(result.test_history, label=f'L={L}', linewidth=2, alpha=0.8)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Test Loss (MSE)', fontsize=12)
        ax2.set_title('Test Loss Convergence', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # 3. Final MSE comparison
        ax3 = plt.subplot(2, 3, 3)
        L_values = sorted(self.results.keys())
        train_mses = [self.results[L].train_mse for L in L_values]
        test_mses = [self.results[L].test_mse for L in L_values]
        
        x = np.arange(len(L_values))
        width = 0.35
        ax3.bar(x - width/2, train_mses, width, label='Train MSE', alpha=0.8)
        ax3.bar(x + width/2, test_mses, width, label='Test MSE', alpha=0.8)
        ax3.set_xlabel('Sequence Length (L)', fontsize=12)
        ax3.set_ylabel('Final MSE', fontsize=12)
        ax3.set_title('Final Performance Comparison', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'L={L}' for L in L_values])
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Training time comparison
        ax4 = plt.subplot(2, 3, 4)
        training_times = [self.results[L].training_time for L in L_values]
        ax4.bar(range(len(L_values)), training_times, alpha=0.8, color='coral')
        ax4.set_xlabel('Sequence Length (L)', fontsize=12)
        ax4.set_ylabel('Training Time (seconds)', fontsize=12)
        ax4.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
        ax4.set_xticks(range(len(L_values)))
        ax4.set_xticklabels([f'L={L}' for L in L_values])
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Convergence speed comparison
        ax5 = plt.subplot(2, 3, 5)
        convergence_speeds = [self.results[L].convergence_speed for L in L_values]
        ax5.bar(range(len(L_values)), convergence_speeds, alpha=0.8, color='green')
        ax5.set_xlabel('Sequence Length (L)', fontsize=12)
        ax5.set_ylabel('Epochs to Convergence', fontsize=12)
        ax5.set_title('Convergence Speed Comparison', fontsize=14, fontweight='bold')
        ax5.set_xticks(range(len(L_values)))
        ax5.set_xticklabels([f'L={L}' for L in L_values])
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Generalization gap
        ax6 = plt.subplot(2, 3, 6)
        gaps = [self.results[L].test_mse - self.results[L].train_mse for L in L_values]
        colors = ['red' if g > 0 else 'blue' for g in gaps]
        ax6.bar(range(len(L_values)), gaps, alpha=0.8, color=colors)
        ax6.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax6.set_xlabel('Sequence Length (L)', fontsize=12)
        ax6.set_ylabel('Generalization Gap (Test - Train MSE)', fontsize=12)
        ax6.set_title('Generalization Analysis', fontsize=14, fontweight='bold')
        ax6.set_xticks(range(len(L_values)))
        ax6.set_xticklabels([f'L={L}' for L in L_values])
        ax6.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_path = self.output_dir / "comparative_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparative plots saved to {plot_path}")
        plt.close()
        
        # Generate text report
        self._generate_text_report()
    
    def _generate_text_report(self):
        """Generate comprehensive text report."""
        report_path = self.output_dir / "analysis_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE SEQUENCE LENGTH EXPERIMENT ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Experiments: {len(self.results)}\n")
            f.write(f"Device: {self.device}\n\n")
            
            # Summary table
            f.write("RESULTS SUMMARY\n")
            f.write("-"*80 + "\n")
            f.write(f"{'L':<8} {'Train MSE':<12} {'Test MSE':<12} {'Gap':<12} {'Time(s)':<10} {'Conv.Epochs':<12}\n")
            f.write("-"*80 + "\n")
            
            for L in sorted(self.results.keys()):
                result = self.results[L]
                gap = result.test_mse - result.train_mse
                f.write(f"{L:<8} {result.train_mse:<12.6f} {result.test_mse:<12.6f} "
                       f"{gap:<12.6f} {result.training_time:<10.2f} {result.convergence_speed:<12}\n")
            
            f.write("\n\nDETAILED ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            # Find best configurations
            best_test = min(self.results.items(), key=lambda x: x[1].test_mse)
            best_train = min(self.results.items(), key=lambda x: x[1].train_mse)
            fastest_convergence = min(self.results.items(), key=lambda x: x[1].convergence_speed)
            fastest_training = min(self.results.items(), key=lambda x: x[1].training_time)
            
            f.write(f"Best Test Performance: L={best_test[0]} (MSE={best_test[1].test_mse:.6f})\n")
            f.write(f"Best Train Performance: L={best_train[0]} (MSE={best_train[1].train_mse:.6f})\n")
            f.write(f"Fastest Convergence: L={fastest_convergence[0]} ({fastest_convergence[1].convergence_speed} epochs)\n")
            f.write(f"Fastest Training: L={fastest_training[0]} ({fastest_training[1].training_time:.2f}s)\n\n")
            
            # Key insights
            f.write("KEY INSIGHTS\n")
            f.write("-"*80 + "\n")
            
            # Analyze performance trend
            L_sorted = sorted(self.results.keys())
            test_mses = [self.results[L].test_mse for L in L_sorted]
            
            if test_mses[0] > test_mses[-1]:
                trend = "Performance improves with increasing L"
            elif test_mses[0] < test_mses[-1]:
                trend = "Performance degrades with increasing L"
            else:
                trend = "Performance is relatively stable across L values"
            
            f.write(f"1. {trend}\n")
            
            # Generalization analysis
            gaps = [self.results[L].test_mse - self.results[L].train_mse for L in L_sorted]
            avg_gap = np.mean(gaps)
            if avg_gap > 0:
                f.write(f"2. Average generalization gap: {avg_gap:.6f} (mild overfitting)\n")
            else:
                f.write(f"2. Average generalization gap: {avg_gap:.6f} (good generalization)\n")
            
            # Efficiency analysis
            time_ratio = self.results[L_sorted[-1]].training_time / self.results[L_sorted[0]].training_time
            f.write(f"3. Training time increases {time_ratio:.2f}x from L={L_sorted[0]} to L={L_sorted[-1]}\n")
            
            f.write("\n\nRECOMMENDATIONS\n")
            f.write("-"*80 + "\n")
            f.write(f"• For best accuracy: Use L={best_test[0]}\n")
            f.write(f"• For fastest training: Use L={fastest_training[0]}\n")
            f.write(f"• For balanced performance: Consider L={L_sorted[len(L_sorted)//2]}\n")
            
        logger.info(f"Text report saved to {report_path}")


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Sequence Length Experiment')
    parser.add_argument(
        '--sequence-lengths',
        type=int,
        nargs='+',
        default=[1, 10, 50, 100, 500],
        help='List of sequence lengths to test'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (default from config)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='experiments/sequence_length_comparison',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Create experiment
    experiment = SequenceLengthExperiment(
        config_path=args.config,
        output_dir=args.output_dir
    )
    
    # Run all experiments
    experiment.run_all_experiments(
        sequence_lengths=args.sequence_lengths,
        epochs=args.epochs
    )
    
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT COMPLETED SUCCESSFULLY!")
    logger.info(f"Results saved to: {experiment.output_dir}")
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    main()

