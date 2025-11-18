"""
Systematic Sensitivity Analysis for LSTM Frequency Extraction
Comprehensive hyperparameter search and analysis framework.

This module provides automated tools for:
1. Grid search over hyperparameter space
2. Statistical analysis of results
3. Visualization of parameter effects
4. Identification of optimal configurations

Author: Research Team
Date: November 2025
"""

import torch
import numpy as np
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.data.signal_generator import create_train_test_generators
from src.data.dataset import create_dataloaders
from src.models.lstm_extractor import create_model
from src.training.trainer import LSTMTrainer
from src.evaluation.metrics import evaluate_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SensitivityConfig:
    """Configuration for sensitivity analysis experiments."""
    
    # Parameter ranges to test
    hidden_sizes: List[int]
    num_layers: List[int]
    dropout_rates: List[float]
    learning_rates: List[float]
    batch_sizes: List[int]
    
    # Training configuration
    epochs: int = 50
    patience: int = 10
    
    # Data configuration
    frequencies: List[float] = None
    sampling_rate: int = 1000
    duration: float = 10.0
    
    # Experiment settings
    num_runs: int = 3  # Multiple runs for statistical significance
    device: str = "auto"
    output_dir: str = "./research/sensitivity_results"
    
    def __post_init__(self):
        if self.frequencies is None:
            self.frequencies = [1.0, 3.0, 5.0, 7.0]


@dataclass
class ExperimentResult:
    """Store results from a single experiment."""
    
    # Configuration
    config: Dict[str, Any]
    run_id: int
    
    # Training metrics
    train_mse: float
    val_mse: float
    test_mse: float
    
    # Additional metrics
    train_r2: float
    test_r2: float
    train_mae: float
    test_mae: float
    
    # Training info
    final_epoch: int
    training_time: float
    num_parameters: int
    
    # Convergence info
    converged: bool
    best_epoch: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class SensitivityAnalyzer:
    """
    Systematic sensitivity analysis for LSTM hyperparameters.
    
    This class orchestrates:
    1. Hyperparameter grid search
    2. Multiple runs for statistical significance
    3. Result collection and analysis
    4. Visualization of parameter effects
    """
    
    def __init__(self, config: SensitivityConfig):
        """
        Initialize the sensitivity analyzer.
        
        Args:
            config: Sensitivity analysis configuration
        """
        self.config = config
        self.results: List[ExperimentResult] = []
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device setup
        self.device = self._setup_device()
        
        # Generate data once (reuse for all experiments)
        logger.info("Generating datasets for sensitivity analysis...")
        self.train_gen, self.test_gen = create_train_test_generators(
            frequencies=config.frequencies,
            sampling_rate=config.sampling_rate,
            duration=config.duration,
            train_seed=1,
            test_seed=2
        )
        
        logger.info(f"Sensitivity analyzer initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup computation device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self.config.device)
    
    def generate_parameter_grid(self) -> List[Dict[str, Any]]:
        """
        Generate all parameter combinations to test.
        
        Returns:
            List of parameter dictionaries
        """
        param_names = ['hidden_size', 'num_layers', 'dropout', 'learning_rate', 'batch_size']
        param_values = [
            self.config.hidden_sizes,
            self.config.num_layers,
            self.config.dropout_rates,
            self.config.learning_rates,
            self.config.batch_sizes
        ]
        
        # Generate all combinations
        combinations = list(product(*param_values))
        
        # Convert to list of dictionaries
        param_grid = []
        for combo in combinations:
            param_dict = dict(zip(param_names, combo))
            param_grid.append(param_dict)
        
        logger.info(f"Generated {len(param_grid)} parameter combinations")
        return param_grid
    
    def run_single_experiment(
        self,
        params: Dict[str, Any],
        run_id: int
    ) -> ExperimentResult:
        """
        Run a single experiment with given parameters.
        
        Args:
            params: Hyperparameter configuration
            run_id: Run identifier (for multiple runs)
            
        Returns:
            ExperimentResult with metrics
        """
        logger.info(f"Running experiment: {params}, Run: {run_id}")
        
        start_time = datetime.now()
        
        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            signal_generator=self.train_gen,
            batch_size=params['batch_size'],
            validation_split=0.1,
            shuffle=False,  # Important: maintain order for stateful LSTM
            num_workers=0
        )
        
        test_loader, _ = create_dataloaders(
            signal_generator=self.test_gen,
            batch_size=params['batch_size'],
            validation_split=0.0,
            shuffle=False,
            num_workers=0
        )
        
        # Create model
        model_config = {
            'input_size': 5,
            'hidden_size': params['hidden_size'],
            'num_layers': params['num_layers'],
            'output_size': 1,
            'dropout': params['dropout'],
            'bidirectional': False
        }
        model = create_model(model_config)
        model = model.to(self.device)
        
        # Training configuration
        training_config = {
            'epochs': self.config.epochs,
            'learning_rate': params['learning_rate'],
            'optimizer': 'adam',
            'early_stopping_patience': self.config.patience,
            'gradient_clip_value': 1.0,
            'save_checkpoints': False  # Don't save for sensitivity analysis
        }
        
        # Create temporary experiment directory
        temp_exp_dir = self.output_dir / f"temp_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        temp_exp_dir.mkdir(exist_ok=True)
        
        # Train model
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
        
        # Calculate training time
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Create result
        result = ExperimentResult(
            config=params,
            run_id=run_id,
            train_mse=train_metrics['overall']['mse'],
            val_mse=history['val_loss'][-1],
            test_mse=test_metrics['overall']['mse'],
            train_r2=train_metrics['overall']['r2_score'],
            test_r2=test_metrics['overall']['r2_score'],
            train_mae=train_metrics['overall']['mae'],
            test_mae=test_metrics['overall']['mae'],
            final_epoch=len(history['train_loss']),
            training_time=training_time,
            num_parameters=model.count_parameters(),
            converged=history['best_epoch'] < self.config.epochs - self.config.patience,
            best_epoch=history['best_epoch']
        )
        
        # Cleanup temp directory
        import shutil
        shutil.rmtree(temp_exp_dir, ignore_errors=True)
        
        logger.info(f"Experiment completed: Test MSE = {result.test_mse:.6f}")
        
        return result
    
    def run_full_analysis(self) -> pd.DataFrame:
        """
        Run complete sensitivity analysis.
        
        Returns:
            DataFrame with all results
        """
        logger.info("="*80)
        logger.info("STARTING SYSTEMATIC SENSITIVITY ANALYSIS")
        logger.info("="*80)
        
        # Generate parameter grid
        param_grid = self.generate_parameter_grid()
        total_experiments = len(param_grid) * self.config.num_runs
        
        logger.info(f"Total experiments to run: {total_experiments}")
        logger.info(f"Estimated time: {total_experiments * 2} minutes (approx)")
        
        # Run all experiments
        experiment_count = 0
        for params in param_grid:
            for run_id in range(self.config.num_runs):
                experiment_count += 1
                logger.info(f"\nExperiment {experiment_count}/{total_experiments}")
                
                try:
                    result = self.run_single_experiment(params, run_id)
                    self.results.append(result)
                except Exception as e:
                    logger.error(f"Experiment failed: {e}")
                    continue
        
        # Convert results to DataFrame
        df = self.results_to_dataframe()
        
        # Save results
        self._save_results(df)
        
        logger.info("="*80)
        logger.info("SENSITIVITY ANALYSIS COMPLETED")
        logger.info("="*80)
        
        return df
    
    def results_to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        data = [result.to_dict() for result in self.results]
        df = pd.DataFrame(data)
        
        # Flatten config dictionary
        config_df = pd.json_normalize(df['config'])
        df = pd.concat([df.drop('config', axis=1), config_df], axis=1)
        
        return df
    
    def _save_results(self, df: pd.DataFrame):
        """Save results to disk."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save CSV
        csv_path = self.output_dir / f"sensitivity_results_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to: {csv_path}")
        
        # Save JSON
        json_path = self.output_dir / f"sensitivity_results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)
        
        # Save config
        config_path = self.output_dir / f"sensitivity_config_{timestamp}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(asdict(self.config), f)
    
    def analyze_results(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform statistical analysis on results.
        
        Args:
            df: Results DataFrame
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("Performing statistical analysis...")
        
        analysis = {}
        
        # 1. Overall statistics
        analysis['overall_stats'] = {
            'mean_test_mse': df['test_mse'].mean(),
            'std_test_mse': df['test_mse'].std(),
            'min_test_mse': df['test_mse'].min(),
            'max_test_mse': df['test_mse'].max(),
            'mean_training_time': df['training_time'].mean(),
            'convergence_rate': df['converged'].mean()
        }
        
        # 2. Best configuration
        best_idx = df['test_mse'].idxmin()
        analysis['best_config'] = df.loc[best_idx].to_dict()
        
        # 3. Parameter importance (correlation with test MSE)
        param_cols = ['hidden_size', 'num_layers', 'dropout', 'learning_rate', 'batch_size']
        correlations = {}
        for param in param_cols:
            corr, pval = stats.spearmanr(df[param], df['test_mse'])
            correlations[param] = {'correlation': corr, 'p_value': pval}
        analysis['parameter_correlations'] = correlations
        
        # 4. Effect of each parameter (grouped statistics)
        parameter_effects = {}
        for param in param_cols:
            grouped = df.groupby(param)['test_mse'].agg(['mean', 'std', 'min', 'max', 'count'])
            parameter_effects[param] = grouped.to_dict('index')
        analysis['parameter_effects'] = parameter_effects
        
        # 5. Stability analysis (variance across runs)
        config_cols = ['hidden_size', 'num_layers', 'dropout', 'learning_rate', 'batch_size']
        stability = df.groupby(config_cols)['test_mse'].agg(['mean', 'std', 'count'])
        stability['coefficient_of_variation'] = stability['std'] / stability['mean']
        analysis['stability'] = {
            'mean_cv': stability['coefficient_of_variation'].mean(),
            'max_cv': stability['coefficient_of_variation'].max(),
            'most_stable_config': stability['coefficient_of_variation'].idxmin()
        }
        
        # Save analysis
        analysis_path = self.output_dir / f"sensitivity_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(analysis_path, 'w') as f:
            # Convert numpy types for JSON serialization
            def convert_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.Series):
                    return obj.to_dict()
                return obj
            
            json.dump(analysis, f, indent=2, default=convert_types)
        
        logger.info(f"Analysis saved to: {analysis_path}")
        
        return analysis
    
    def visualize_results(self, df: pd.DataFrame):
        """
        Create comprehensive visualizations of sensitivity analysis.
        
        Args:
            df: Results DataFrame
        """
        logger.info("Creating visualizations...")
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (15, 10)
        
        # 1. Parameter sweep plots
        self._plot_parameter_sweeps(df)
        
        # 2. Heatmaps for parameter interactions
        self._plot_parameter_heatmaps(df)
        
        # 3. Performance distributions
        self._plot_performance_distributions(df)
        
        # 4. Training efficiency
        self._plot_training_efficiency(df)
        
        # 5. Convergence analysis
        self._plot_convergence_analysis(df)
        
        logger.info("Visualizations created successfully")
    
    def _plot_parameter_sweeps(self, df: pd.DataFrame):
        """Plot effect of each parameter on performance."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        params = ['hidden_size', 'num_layers', 'dropout', 'learning_rate', 'batch_size']
        
        for idx, param in enumerate(params):
            ax = axes[idx]
            
            # Group by parameter and compute statistics
            grouped = df.groupby(param)['test_mse'].agg(['mean', 'std', 'min', 'max'])
            
            # Plot
            x = grouped.index
            ax.plot(x, grouped['mean'], 'o-', linewidth=2, markersize=8, label='Mean')
            ax.fill_between(x, grouped['mean'] - grouped['std'], 
                           grouped['mean'] + grouped['std'], alpha=0.3, label='±1 std')
            
            ax.set_xlabel(param.replace('_', ' ').title(), fontsize=12)
            ax.set_ylabel('Test MSE', fontsize=12)
            ax.set_title(f'Effect of {param.replace("_", " ").title()}', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Log scale for learning rate
            if param == 'learning_rate':
                ax.set_xscale('log')
        
        # Remove extra subplot
        fig.delaxes(axes[-1])
        
        plt.tight_layout()
        plot_path = self.output_dir / 'parameter_sweeps.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Parameter sweep plot saved: {plot_path}")
    
    def _plot_parameter_heatmaps(self, df: pd.DataFrame):
        """Plot heatmaps showing parameter interactions."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        interactions = [
            ('hidden_size', 'num_layers'),
            ('learning_rate', 'batch_size'),
            ('hidden_size', 'dropout'),
            ('num_layers', 'dropout')
        ]
        
        for idx, (param1, param2) in enumerate(interactions):
            ax = axes[idx // 2, idx % 2]
            
            # Pivot table for heatmap
            pivot = df.pivot_table(
                values='test_mse',
                index=param1,
                columns=param2,
                aggfunc='mean'
            )
            
            # Plot heatmap
            sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Test MSE'})
            ax.set_title(f'Interaction: {param1} vs {param2}', fontsize=12)
            ax.set_xlabel(param2.replace('_', ' ').title())
            ax.set_ylabel(param1.replace('_', ' ').title())
        
        plt.tight_layout()
        plot_path = self.output_dir / 'parameter_interactions.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Parameter interaction heatmaps saved: {plot_path}")
    
    def _plot_performance_distributions(self, df: pd.DataFrame):
        """Plot performance metric distributions."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics = [
            ('test_mse', 'Test MSE'),
            ('test_r2', 'Test R²'),
            ('test_mae', 'Test MAE'),
            ('training_time', 'Training Time (s)')
        ]
        
        for idx, (metric, label) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            # Histogram
            ax.hist(df[metric], bins=30, alpha=0.7, edgecolor='black')
            ax.axvline(df[metric].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
            ax.axvline(df[metric].median(), color='green', linestyle='--', linewidth=2, label='Median')
            
            ax.set_xlabel(label, fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(f'Distribution of {label}', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'performance_distributions.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance distributions saved: {plot_path}")
    
    def _plot_training_efficiency(self, df: pd.DataFrame):
        """Plot training efficiency analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. Performance vs training time
        ax = axes[0]
        scatter = ax.scatter(df['training_time'], df['test_mse'], 
                            c=df['num_parameters'], s=100, alpha=0.6, cmap='viridis')
        ax.set_xlabel('Training Time (seconds)', fontsize=12)
        ax.set_ylabel('Test MSE', fontsize=12)
        ax.set_title('Performance vs Training Time', fontsize=14)
        plt.colorbar(scatter, ax=ax, label='Number of Parameters')
        ax.grid(True, alpha=0.3)
        
        # 2. Performance vs model size
        ax = axes[1]
        scatter = ax.scatter(df['num_parameters'], df['test_mse'],
                            c=df['training_time'], s=100, alpha=0.6, cmap='plasma')
        ax.set_xlabel('Number of Parameters', fontsize=12)
        ax.set_ylabel('Test MSE', fontsize=12)
        ax.set_title('Performance vs Model Size', fontsize=14)
        plt.colorbar(scatter, ax=ax, label='Training Time (s)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'training_efficiency.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training efficiency plot saved: {plot_path}")
    
    def _plot_convergence_analysis(self, df: pd.DataFrame):
        """Plot convergence analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. Convergence rate by configuration
        ax = axes[0]
        convergence_by_param = {}
        for param in ['hidden_size', 'num_layers', 'dropout']:
            grouped = df.groupby(param)['converged'].mean()
            convergence_by_param[param] = grouped
        
        x = np.arange(len(convergence_by_param))
        width = 0.25
        for idx, (param, values) in enumerate(convergence_by_param.items()):
            ax.bar(x + idx*width, values.values, width, label=param, alpha=0.8)
        
        ax.set_xlabel('Parameter Value', fontsize=12)
        ax.set_ylabel('Convergence Rate', fontsize=12)
        ax.set_title('Convergence Rate by Parameter', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Epochs to convergence
        ax = axes[1]
        converged_df = df[df['converged']]
        ax.hist(converged_df['best_epoch'], bins=20, alpha=0.7, edgecolor='black')
        ax.axvline(converged_df['best_epoch'].mean(), color='red', 
                  linestyle='--', linewidth=2, label='Mean')
        ax.set_xlabel('Epochs to Best Model', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Convergence Speed', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'convergence_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Convergence analysis plot saved: {plot_path}")


def create_default_sensitivity_config() -> SensitivityConfig:
    """Create a default sensitivity analysis configuration."""
    return SensitivityConfig(
        hidden_sizes=[64, 128, 256],
        num_layers=[1, 2, 3],
        dropout_rates=[0.0, 0.1, 0.2, 0.3],
        learning_rates=[0.0001, 0.001, 0.01],
        batch_sizes=[16, 32, 64],
        epochs=30,
        patience=5,
        num_runs=3,
        output_dir="./research/sensitivity_results"
    )


def main():
    """Main function to run sensitivity analysis."""
    # Create configuration
    config = create_default_sensitivity_config()
    
    # For quick testing, use smaller ranges
    config.hidden_sizes = [64, 128]
    config.num_layers = [1, 2]
    config.dropout_rates = [0.0, 0.2]
    config.learning_rates = [0.001, 0.01]
    config.batch_sizes = [32]
    config.epochs = 20
    config.num_runs = 2
    
    # Create analyzer
    analyzer = SensitivityAnalyzer(config)
    
    # Run analysis
    results_df = analyzer.run_full_analysis()
    
    # Analyze results
    analysis = analyzer.analyze_results(results_df)
    
    # Create visualizations
    analyzer.visualize_results(results_df)
    
    # Print summary
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("="*80)
    print(f"\nBest Test MSE: {analysis['overall_stats']['min_test_mse']:.6f}")
    print(f"Mean Test MSE: {analysis['overall_stats']['mean_test_mse']:.6f}")
    print(f"Std Test MSE: {analysis['overall_stats']['std_test_mse']:.6f}")
    print(f"\nConvergence Rate: {analysis['overall_stats']['convergence_rate']*100:.1f}%")
    print(f"Mean Training Time: {analysis['overall_stats']['mean_training_time']:.1f}s")
    
    print("\n" + "="*80)
    print("BEST CONFIGURATION")
    print("="*80)
    for key, value in analysis['best_config'].items():
        print(f"{key}: {value}")
    
    print("\n" + "="*80)
    print("PARAMETER IMPORTANCE (Correlation with Test MSE)")
    print("="*80)
    for param, corr_data in analysis['parameter_correlations'].items():
        print(f"{param}: {corr_data['correlation']:.3f} (p={corr_data['p_value']:.4f})")


if __name__ == "__main__":
    main()

