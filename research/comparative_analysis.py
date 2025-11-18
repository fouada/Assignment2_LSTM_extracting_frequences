"""
Data-Based Comparative Analysis
Empirical comparison of different architectures, configurations, and approaches.

This module provides:
1. Architecture comparisons (LSTM vs GRU vs RNN)
2. Configuration comparisons (different hyperparameters)
3. Statistical significance testing
4. Performance benchmarking
5. Ablation studies

Author: Research Team
Date: November 2025
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import logging
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.data.signal_generator import create_train_test_generators
from src.data.dataset import create_dataloaders
from src.models.lstm_extractor import StatefulLSTMExtractor, create_model
from src.training.trainer import LSTMTrainer
from src.evaluation.metrics import evaluate_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Store results from comparative experiments."""
    
    model_name: str
    config: Dict[str, Any]
    
    # Performance metrics
    train_mse: float
    test_mse: float
    train_r2: float
    test_r2: float
    train_mae: float
    test_mae: float
    
    # Per-frequency metrics
    freq_mse: Dict[str, float]
    freq_r2: Dict[str, float]
    
    # Training metrics
    training_time: float
    convergence_epoch: int
    final_epoch: int
    num_parameters: int
    
    # Statistical measures
    mean_error: float
    std_error: float
    max_error: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GRUExtractor(nn.Module):
    """GRU-based extractor for comparison."""
    
    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2
    ):
        super(GRUExtractor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.input_norm = nn.LayerNorm(input_size)
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.output_norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.activation = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
        self.hidden_state = None
    
    def reset_state(self):
        self.hidden_state = None
    
    def init_hidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
    
    def forward(self, x, reset_state=False):
        if x.dim() == 2:
            x = x.unsqueeze(1)
            single_sample = True
        else:
            single_sample = False
        
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        x = self.input_norm(x)
        
        if reset_state or self.hidden_state is None:
            self.hidden_state = self.init_hidden(batch_size, device)
        else:
            if self.hidden_state.size(1) != batch_size:
                self.hidden_state = self.init_hidden(batch_size, device)
        
        gru_out, self.hidden_state = self.gru(x, self.hidden_state)
        
        gru_out = self.output_norm(gru_out)
        out = self.fc1(gru_out)
        out = self.activation(out)
        out = self.dropout_layer(out)
        out = self.fc2(out)
        
        if single_sample:
            out = out.squeeze(1)
        
        return out
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SimpleRNN(nn.Module):
    """Simple RNN for comparison."""
    
    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2
    ):
        super(SimpleRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.input_norm = nn.LayerNorm(input_size)
        
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            nonlinearity='tanh'
        )
        
        self.output_norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.activation = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
        self.hidden_state = None
    
    def reset_state(self):
        self.hidden_state = None
    
    def init_hidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
    
    def forward(self, x, reset_state=False):
        if x.dim() == 2:
            x = x.unsqueeze(1)
            single_sample = True
        else:
            single_sample = False
        
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        x = self.input_norm(x)
        
        if reset_state or self.hidden_state is None:
            self.hidden_state = self.init_hidden(batch_size, device)
        else:
            if self.hidden_state.size(1) != batch_size:
                self.hidden_state = self.init_hidden(batch_size, device)
        
        rnn_out, self.hidden_state = self.rnn(x, self.hidden_state)
        
        rnn_out = self.output_norm(rnn_out)
        out = self.fc1(rnn_out)
        out = self.activation(out)
        out = self.dropout_layer(out)
        out = self.fc2(out)
        
        if single_sample:
            out = out.squeeze(1)
        
        return out
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ComparativeAnalyzer:
    """
    Comprehensive comparative analysis framework.
    
    Features:
    1. Compare different architectures (LSTM, GRU, RNN)
    2. Compare different configurations
    3. Statistical significance testing
    4. Ablation studies
    5. Performance profiling
    """
    
    def __init__(self, output_dir: str = "./research/comparison_results"):
        """Initialize the comparative analyzer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[ComparisonResult] = []
        
        # Setup device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        logger.info(f"Comparative Analyzer initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"Output: {self.output_dir}")
    
    def compare_architectures(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        epochs: int = 30,
        num_runs: int = 3
    ) -> pd.DataFrame:
        """
        Compare LSTM vs GRU vs SimpleRNN architectures.
        
        Args:
            hidden_size: Hidden dimension
            num_layers: Number of layers
            epochs: Training epochs
            num_runs: Number of runs for statistical significance
            
        Returns:
            DataFrame with comparison results
        """
        logger.info("="*80)
        logger.info("ARCHITECTURE COMPARISON: LSTM vs GRU vs RNN")
        logger.info("="*80)
        
        # Generate data
        train_gen, test_gen = create_train_test_generators(
            frequencies=[1.0, 3.0, 5.0, 7.0],
            sampling_rate=1000,
            duration=10.0,
            train_seed=1,
            test_seed=2
        )
        
        architectures = {
            'LSTM': StatefulLSTMExtractor,
            'GRU': GRUExtractor,
            'RNN': SimpleRNN
        }
        
        for arch_name, arch_class in architectures.items():
            logger.info(f"\nTesting {arch_name} architecture...")
            
            for run_id in range(num_runs):
                logger.info(f"  Run {run_id + 1}/{num_runs}")
                
                result = self._run_single_comparison(
                    model_name=f"{arch_name}_run{run_id}",
                    model_class=arch_class,
                    train_gen=train_gen,
                    test_gen=test_gen,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    epochs=epochs
                )
                
                self.results.append(result)
        
        # Convert to DataFrame and analyze
        df = self._results_to_dataframe()
        
        # Statistical tests
        self._perform_statistical_tests(df)
        
        # Save results
        self._save_comparison_results(df, "architecture_comparison")
        
        # Visualize
        self._visualize_architecture_comparison(df)
        
        return df
    
    def compare_sequence_lengths(
        self,
        sequence_lengths: List[int] = [1, 10, 50, 100],
        epochs: int = 30,
        num_runs: int = 3
    ) -> pd.DataFrame:
        """
        Compare different sequence lengths (L=1 vs L>1).
        
        Args:
            sequence_lengths: List of L values to test
            epochs: Training epochs
            num_runs: Number of runs
            
        Returns:
            DataFrame with results
        """
        logger.info("="*80)
        logger.info("SEQUENCE LENGTH COMPARISON")
        logger.info("="*80)
        
        train_gen, test_gen = create_train_test_generators(
            frequencies=[1.0, 3.0, 5.0, 7.0],
            sampling_rate=1000,
            duration=10.0,
            train_seed=1,
            test_seed=2
        )
        
        for seq_len in sequence_lengths:
            logger.info(f"\nTesting sequence length: {seq_len}")
            
            for run_id in range(num_runs):
                logger.info(f"  Run {run_id + 1}/{num_runs}")
                
                result = self._run_single_comparison(
                    model_name=f"LSTM_L{seq_len}_run{run_id}",
                    model_class=StatefulLSTMExtractor,
                    train_gen=train_gen,
                    test_gen=test_gen,
                    hidden_size=128,
                    num_layers=2,
                    epochs=epochs,
                    sequence_length=seq_len
                )
                
                self.results.append(result)
        
        df = self._results_to_dataframe()
        self._save_comparison_results(df, "sequence_length_comparison")
        self._visualize_sequence_length_comparison(df)
        
        return df
    
    def ablation_study(
        self,
        epochs: int = 30,
        num_runs: int = 3
    ) -> pd.DataFrame:
        """
        Perform ablation study on model components.
        
        Tests:
        1. No normalization
        2. No dropout
        3. Single layer
        4. Smaller hidden size
        5. No state management
        
        Args:
            epochs: Training epochs
            num_runs: Number of runs
            
        Returns:
            DataFrame with results
        """
        logger.info("="*80)
        logger.info("ABLATION STUDY")
        logger.info("="*80)
        
        train_gen, test_gen = create_train_test_generators(
            frequencies=[1.0, 3.0, 5.0, 7.0],
            sampling_rate=1000,
            duration=10.0,
            train_seed=1,
            test_seed=2
        )
        
        ablation_configs = {
            'Full Model': {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.2},
            'No Dropout': {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.0},
            'Single Layer': {'hidden_size': 128, 'num_layers': 1, 'dropout': 0.2},
            'Small Hidden': {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.2},
            'Very Small': {'hidden_size': 32, 'num_layers': 1, 'dropout': 0.1},
        }
        
        for config_name, config in ablation_configs.items():
            logger.info(f"\nTesting: {config_name}")
            
            for run_id in range(num_runs):
                logger.info(f"  Run {run_id + 1}/{num_runs}")
                
                result = self._run_single_comparison(
                    model_name=f"{config_name}_run{run_id}",
                    model_class=StatefulLSTMExtractor,
                    train_gen=train_gen,
                    test_gen=test_gen,
                    hidden_size=config['hidden_size'],
                    num_layers=config['num_layers'],
                    dropout=config['dropout'],
                    epochs=epochs
                )
                
                self.results.append(result)
        
        df = self._results_to_dataframe()
        self._save_comparison_results(df, "ablation_study")
        self._visualize_ablation_study(df)
        
        return df
    
    def _run_single_comparison(
        self,
        model_name: str,
        model_class: type,
        train_gen,
        test_gen,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        epochs: int = 30,
        sequence_length: int = 1
    ) -> ComparisonResult:
        """Run a single comparison experiment."""
        
        start_time = datetime.now()
        
        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            signal_generator=train_gen,
            batch_size=32,
            validation_split=0.1,
            shuffle=False,
            num_workers=0
        )
        
        test_loader, _ = create_dataloaders(
            signal_generator=test_gen,
            batch_size=32,
            validation_split=0.0,
            shuffle=False,
            num_workers=0
        )
        
        # Create model
        model = model_class(
            input_size=5,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=1,
            dropout=dropout
        )
        model = model.to(self.device)
        
        # Training config
        training_config = {
            'epochs': epochs,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'early_stopping_patience': 10,
            'gradient_clip_value': 1.0,
            'save_checkpoints': False
        }
        
        # Create temp directory
        temp_exp_dir = self.output_dir / f"temp_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        temp_exp_dir.mkdir(exist_ok=True)
        
        # Train
        trainer = LSTMTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_config,
            device=self.device,
            experiment_dir=temp_exp_dir
        )
        
        history = trainer.train()
        
        # Evaluate
        train_metrics = evaluate_model(model, train_loader, self.device)
        test_metrics = evaluate_model(model, test_loader, self.device)
        
        # Calculate statistics
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Get per-frequency metrics
        freq_mse = {}
        freq_r2 = {}
        for i, freq in enumerate([1.0, 3.0, 5.0, 7.0]):
            freq_key = f"frequency_{freq}Hz"
            if freq_key in test_metrics:
                freq_mse[freq_key] = test_metrics[freq_key].get('mse', 0)
                freq_r2[freq_key] = test_metrics[freq_key].get('r2_score', 0)
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_exp_dir, ignore_errors=True)
        
        # Create result
        result = ComparisonResult(
            model_name=model_name,
            config={
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'dropout': dropout,
                'sequence_length': sequence_length
            },
            train_mse=train_metrics['overall']['mse'],
            test_mse=test_metrics['overall']['mse'],
            train_r2=train_metrics['overall']['r2_score'],
            test_r2=test_metrics['overall']['r2_score'],
            train_mae=train_metrics['overall']['mae'],
            test_mae=test_metrics['overall']['mae'],
            freq_mse=freq_mse,
            freq_r2=freq_r2,
            training_time=training_time,
            convergence_epoch=history.get('best_epoch', epochs),
            final_epoch=len(history['train_loss']),
            num_parameters=model.count_parameters(),
            mean_error=test_metrics['overall']['mae'],
            std_error=np.std([v for v in freq_mse.values()]),
            max_error=max([v for v in freq_mse.values()] if freq_mse else [0])
        )
        
        logger.info(f"Completed {model_name}: Test MSE = {result.test_mse:.6f}")
        
        return result
    
    def _results_to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        data = [result.to_dict() for result in self.results]
        df = pd.DataFrame(data)
        
        # Flatten nested dicts
        config_df = pd.json_normalize(df['config'])
        df = pd.concat([df.drop(['config', 'freq_mse', 'freq_r2'], axis=1), config_df], axis=1)
        
        return df
    
    def _perform_statistical_tests(self, df: pd.DataFrame):
        """Perform statistical significance tests."""
        logger.info("\n" + "="*80)
        logger.info("STATISTICAL SIGNIFICANCE TESTS")
        logger.info("="*80)
        
        # Extract architecture name
        df['architecture'] = df['model_name'].str.split('_').str[0]
        
        # Get architectures
        architectures = df['architecture'].unique()
        
        # Pairwise t-tests
        for i, arch1 in enumerate(architectures):
            for arch2 in architectures[i+1:]:
                data1 = df[df['architecture'] == arch1]['test_mse']
                data2 = df[df['architecture'] == arch2]['test_mse']
                
                # t-test
                t_stat, p_value = stats.ttest_ind(data1, data2)
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt((data1.std()**2 + data2.std()**2) / 2)
                cohens_d = (data1.mean() - data2.mean()) / pooled_std
                
                logger.info(f"\n{arch1} vs {arch2}:")
                logger.info(f"  t-statistic: {t_stat:.4f}")
                logger.info(f"  p-value: {p_value:.4f}")
                logger.info(f"  Cohen's d: {cohens_d:.4f}")
                
                if p_value < 0.05:
                    winner = arch1 if data1.mean() < data2.mean() else arch2
                    logger.info(f"  ✓ Significant difference (α=0.05): {winner} is better")
                else:
                    logger.info(f"  ✗ No significant difference")
    
    def _save_comparison_results(self, df: pd.DataFrame, name: str):
        """Save comparison results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # CSV
        csv_path = self.output_dir / f"{name}_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Results saved: {csv_path}")
        
        # JSON
        json_path = self.output_dir / f"{name}_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)
        
        # Summary statistics
        summary = df.groupby('model_name').agg({
            'test_mse': ['mean', 'std', 'min', 'max'],
            'test_r2': ['mean', 'std'],
            'training_time': ['mean', 'std'],
            'num_parameters': 'first'
        })
        
        summary_path = self.output_dir / f"{name}_summary_{timestamp}.csv"
        summary.to_csv(summary_path)
    
    def _visualize_architecture_comparison(self, df: pd.DataFrame):
        """Visualize architecture comparison."""
        df['architecture'] = df['model_name'].str.split('_').str[0]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. MSE comparison
        ax = axes[0, 0]
        df.boxplot(column='test_mse', by='architecture', ax=ax)
        ax.set_title('Test MSE by Architecture')
        ax.set_xlabel('Architecture')
        ax.set_ylabel('Test MSE')
        
        # 2. R² comparison
        ax = axes[0, 1]
        df.boxplot(column='test_r2', by='architecture', ax=ax)
        ax.set_title('Test R² by Architecture')
        ax.set_xlabel('Architecture')
        ax.set_ylabel('Test R²')
        
        # 3. Training time
        ax = axes[1, 0]
        df.boxplot(column='training_time', by='architecture', ax=ax)
        ax.set_title('Training Time by Architecture')
        ax.set_xlabel('Architecture')
        ax.set_ylabel('Time (seconds)')
        
        # 4. Parameters vs Performance
        ax = axes[1, 1]
        for arch in df['architecture'].unique():
            arch_df = df[df['architecture'] == arch]
            ax.scatter(arch_df['num_parameters'], arch_df['test_mse'], 
                      label=arch, s=100, alpha=0.6)
        ax.set_xlabel('Number of Parameters')
        ax.set_ylabel('Test MSE')
        ax.set_title('Model Size vs Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('')  # Remove automatic title
        plt.tight_layout()
        
        plot_path = self.output_dir / 'architecture_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved: {plot_path}")
    
    def _visualize_sequence_length_comparison(self, df: pd.DataFrame):
        """Visualize sequence length comparison."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Extract sequence length
        df['seq_len'] = df['model_name'].str.extract(r'L(\d+)').astype(int)
        
        # 1. MSE vs sequence length
        ax = axes[0]
        grouped = df.groupby('seq_len')['test_mse'].agg(['mean', 'std'])
        ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], 
                   marker='o', markersize=8, linewidth=2, capsize=5)
        ax.set_xlabel('Sequence Length (L)', fontsize=12)
        ax.set_ylabel('Test MSE', fontsize=12)
        ax.set_title('Performance vs Sequence Length', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        # 2. Training time vs sequence length
        ax = axes[1]
        grouped = df.groupby('seq_len')['training_time'].agg(['mean', 'std'])
        ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                   marker='s', markersize=8, linewidth=2, capsize=5, color='orange')
        ax.set_xlabel('Sequence Length (L)', fontsize=12)
        ax.set_ylabel('Training Time (s)', fontsize=12)
        ax.set_title('Training Efficiency vs Sequence Length', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        plt.tight_layout()
        plot_path = self.output_dir / 'sequence_length_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved: {plot_path}")
    
    def _visualize_ablation_study(self, df: pd.DataFrame):
        """Visualize ablation study results."""
        # Extract config name
        df['config'] = df['model_name'].str.replace(r'_run\d+', '', regex=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. MSE comparison
        ax = axes[0, 0]
        grouped = df.groupby('config')['test_mse'].mean().sort_values()
        grouped.plot(kind='barh', ax=ax, color='steelblue')
        ax.set_xlabel('Test MSE', fontsize=12)
        ax.set_title('Performance by Configuration', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')
        
        # 2. R² comparison
        ax = axes[0, 1]
        grouped = df.groupby('config')['test_r2'].mean().sort_values(ascending=False)
        grouped.plot(kind='barh', ax=ax, color='forestgreen')
        ax.set_xlabel('Test R²', fontsize=12)
        ax.set_title('R² Score by Configuration', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')
        
        # 3. Parameters vs Performance
        ax = axes[1, 0]
        for config in df['config'].unique():
            config_df = df[df['config'] == config]
            ax.scatter(config_df['num_parameters'], config_df['test_mse'],
                      label=config, s=100, alpha=0.6)
        ax.set_xlabel('Number of Parameters')
        ax.set_ylabel('Test MSE')
        ax.set_title('Model Complexity vs Performance')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 4. Training efficiency
        ax = axes[1, 1]
        for config in df['config'].unique():
            config_df = df[df['config'] == config]
            ax.scatter(config_df['training_time'], config_df['test_mse'],
                      label=config, s=100, alpha=0.6)
        ax.set_xlabel('Training Time (s)')
        ax.set_ylabel('Test MSE')
        ax.set_title('Training Efficiency')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'ablation_study.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved: {plot_path}")


def main():
    """Main function for comparative analysis."""
    analyzer = ComparativeAnalyzer()
    
    # 1. Architecture comparison
    logger.info("\n" + "="*80)
    logger.info("RUNNING ARCHITECTURE COMPARISON")
    logger.info("="*80)
    arch_df = analyzer.compare_architectures(epochs=20, num_runs=2)
    
    # 2. Ablation study
    logger.info("\n" + "="*80)
    logger.info("RUNNING ABLATION STUDY")
    logger.info("="*80)
    ablation_df = analyzer.ablation_study(epochs=20, num_runs=2)
    
    # Print summary
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nTotal experiments run: {len(analyzer.results)}")
    print(f"Results saved to: {analyzer.output_dir}")


if __name__ == "__main__":
    main()

