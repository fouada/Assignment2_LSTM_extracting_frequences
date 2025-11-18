"""
Production Main Entry Point with Plugin Architecture
Professional ML Framework with Extensibility

Author: Professional ML Engineering Team
Date: 2025
"""

import torch
import yaml
import logging
from pathlib import Path
from datetime import datetime
import numpy as np

# Core framework
from src.core.plugin import PluginManager
from src.core.events import EventManager, get_event_manager
from src.core.hooks import HookManager, get_hook_manager
from src.core.registry import ComponentRegistry, get_component_registry
from src.core.container import Container, ServiceProvider
from src.core.config import ConfigManager

# Original modules
from src.data.signal_generator import create_train_test_generators
from src.data.dataset import create_dataloaders, FrequencyExtractionDataset
from src.models.lstm_extractor import create_model
from src.training.trainer import LSTMTrainer
from src.evaluation.metrics import evaluate_model, compare_train_test_performance
from src.visualization.plotter import create_all_visualizations

# Plugins
from plugins.tensorboard_plugin import TensorBoardPlugin
from plugins.early_stopping_plugin import EarlyStoppingPlugin
from plugins.custom_metrics_plugin import CustomMetricsPlugin
from plugins.data_augmentation_plugin import DataAugmentationPlugin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_production.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProductionMLFramework:
    """
    Production-level ML framework with plugin architecture.
    
    Features:
    - Plugin system for extensibility
    - Event-driven architecture
    - Hook-based customization
    - Component registry
    - Dependency injection
    - Advanced configuration management
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """
        Initialize the framework.
        
        Args:
            config_path: Path to configuration file
        """
        logger.info("="*80)
        logger.info("Initializing Production ML Framework")
        logger.info("="*80)
        
        # Initialize core components
        self.config = ConfigManager(config_path)
        self.event_manager = get_event_manager()
        self.hook_manager = get_hook_manager()
        self.registry = get_component_registry()
        self.container = Container()
        self.plugin_manager = PluginManager()
        
        # Service provider for easy access
        self.services = ServiceProvider(self.container)
        self.services.add_service('config', self.config)
        self.services.add_service('events', self.event_manager)
        self.services.add_service('hooks', self.hook_manager)
        self.services.add_service('registry', self.registry)
        
        # State
        self.experiment_dir = None
        self.device = None
        
        logger.info("Framework core initialized")
    
    def initialize_plugins(self) -> None:
        """Initialize and register all plugins."""
        logger.info("\n" + "="*80)
        logger.info("Initializing Plugins")
        logger.info("="*80)
        
        # Get plugin configurations
        plugin_configs = self.config.get('plugins', {})
        
        # Create plugin instances
        plugins = [
            TensorBoardPlugin(),
            EarlyStoppingPlugin(),
            CustomMetricsPlugin(),
            DataAugmentationPlugin(),
        ]
        
        # Initialize and register plugins
        for plugin in plugins:
            plugin_name = plugin.__class__.__name__.replace('Plugin', '').lower()
            plugin_config = plugin_configs.get(plugin_name, {})
            
            # Add framework services to plugin config
            plugin_config['event_manager'] = self.event_manager
            plugin_config['hook_manager'] = self.hook_manager
            plugin_config['registry'] = self.registry
            
            plugin.initialize(**plugin_config)
            self.plugin_manager.register_plugin(plugin)
            
            logger.info(f"✓ Plugin registered: {plugin.name} v{plugin.version}")
        
        logger.info(f"\nTotal plugins loaded: {len(self.plugin_manager)}")
    
    def setup_device(self) -> torch.device:
        """Setup computation device."""
        device_config = self.config.get('compute.device', 'auto')
        
        if device_config == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        else:
            device = torch.device(device_config)
        
        self.device = device
        logger.info(f"Using device: {device}")
        return device
    
    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info(f"Random seed set to {seed}")
    
    def create_experiment_directory(self) -> Path:
        """Create directory for experiment outputs."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = self.config.get('experiment.name', 'experiment')
        exp_dir = Path(self.config.get('experiment.save_dir', './experiments')) / f"{exp_name}_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (exp_dir / 'plots').mkdir(exist_ok=True)
        (exp_dir / 'checkpoints').mkdir(exist_ok=True)
        
        self.experiment_dir = exp_dir
        logger.info(f"Experiment directory: {exp_dir}")
        
        # Save configuration
        self.config.save(exp_dir / 'config.yaml')
        
        return exp_dir
    
    def run(self) -> dict:
        """
        Main execution pipeline.
        
        Returns:
            Dictionary with experiment results
        """
        # Publish training start event
        self.event_manager.publish(
            EventManager.TRAINING_START,
            data={
                'config': self.config.to_dict(),
                'experiment_dir': self.experiment_dir
            },
            source='framework'
        )
        
        # ========================================================================
        # Step 1: Generate Data
        # ========================================================================
        logger.info("\n" + "="*80)
        logger.info("STEP 1: Data Generation")
        logger.info("="*80)
        
        self.hook_manager.execute(HookManager.BEFORE_DATA_LOAD)
        
        train_generator, test_generator = create_train_test_generators(
            frequencies=self.config.get('data.frequencies'),
            sampling_rate=self.config.get('data.sampling_rate'),
            duration=self.config.get('data.duration'),
            amplitude_range=self.config.get('data.amplitude_range'),
            phase_range=self.config.get('data.phase_range'),
            train_seed=self.config.get('data.train_seed'),
            test_seed=self.config.get('data.test_seed')
        )
        
        self.hook_manager.execute(HookManager.AFTER_DATA_LOAD, train_generator, test_generator)
        
        self.event_manager.publish(
            EventManager.DATA_LOADED,
            data={'num_frequencies': len(self.config.get('data.frequencies'))},
            source='framework'
        )
        
        # ========================================================================
        # Step 2: Create Datasets and Dataloaders
        # ========================================================================
        logger.info("\n" + "="*80)
        logger.info("STEP 2: Dataset Creation")
        logger.info("="*80)
        
        train_loader, test_loader = create_dataloaders(
            train_generator=train_generator,
            test_generator=test_generator,
            batch_size=self.config.get('training.batch_size'),
            normalize=True,
            device='cpu'
        )
        
        # Keep datasets for later visualization
        train_dataset = FrequencyExtractionDataset(train_generator, normalize=True)
        test_dataset = FrequencyExtractionDataset(test_generator, normalize=True)
        
        self.event_manager.publish(
            EventManager.DATA_PREPROCESSED,
            data={
                'train_samples': len(train_dataset),
                'test_samples': len(test_dataset)
            },
            source='framework'
        )
        
        # ========================================================================
        # Step 3: Create Model
        # ========================================================================
        logger.info("\n" + "="*80)
        logger.info("STEP 3: Model Creation")
        logger.info("="*80)
        
        model = create_model(self.config.get('model'))
        model = model.to(self.device)
        
        self.event_manager.publish(
            EventManager.MODEL_CREATED,
            data={
                'model': model,
                'parameters': model.count_parameters()
            },
            source='framework'
        )
        
        # ========================================================================
        # Step 4: Training
        # ========================================================================
        logger.info("\n" + "="*80)
        logger.info("STEP 4: Model Training")
        logger.info("="*80)
        
        trainer = LSTMTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            config=self.config.get('training'),
            device=self.device,
            experiment_dir=self.experiment_dir / 'checkpoints'
        )
        
        # Integrate with event system
        trainer.event_manager = self.event_manager
        trainer.hook_manager = self.hook_manager
        
        history = trainer.train()
        
        self.event_manager.publish(
            EventManager.TRAINING_END,
            data={'history': history},
            source='framework'
        )
        
        # ========================================================================
        # Step 5: Evaluation
        # ========================================================================
        logger.info("\n" + "="*80)
        logger.info("STEP 5: Model Evaluation")
        logger.info("="*80)
        
        self.event_manager.publish(
            EventManager.EVALUATION_START,
            source='framework'
        )
        
        # Load best model
        best_model_path = self.experiment_dir / 'checkpoints' / 'best_model.pt'
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Best model loaded for evaluation")
        
        # Evaluate on train set
        logger.info("\nEvaluating on TRAIN set...")
        train_results = evaluate_model(model, train_loader, self.device, compute_per_frequency=True)
        
        # Evaluate on test set
        logger.info("\nEvaluating on TEST set...")
        test_results = evaluate_model(model, test_loader, self.device, compute_per_frequency=True)
        
        # Compare performance
        logger.info("\n" + "-"*80)
        logger.info("GENERALIZATION ANALYSIS")
        logger.info("-"*80)
        comparison = compare_train_test_performance(train_results, test_results)
        
        # Print detailed results
        self._print_results(train_results, test_results, comparison)
        
        self.event_manager.publish(
            EventManager.EVALUATION_END,
            data={
                'train_results': train_results,
                'test_results': test_results,
                'comparison': comparison
            },
            source='framework'
        )
        
        # ========================================================================
        # Step 6: Visualization
        # ========================================================================
        logger.info("\n" + "="*80)
        logger.info("STEP 6: Creating Visualizations")
        logger.info("="*80)
        
        # Extract predictions per frequency for visualization
        test_predictions_dict = self._extract_predictions_per_frequency(model, test_dataset)
        
        # Create all visualizations
        create_all_visualizations(
            test_dataset=test_dataset,
            predictions_dict=test_predictions_dict,
            frequencies=self.config.get('data.frequencies'),
            history=history,
            train_metrics=train_results,
            test_metrics=test_results,
            save_dir=self.experiment_dir / 'plots'
        )
        
        self.event_manager.publish(
            EventManager.PLOT_CREATED,
            data={'save_dir': self.experiment_dir / 'plots'},
            source='framework'
        )
        
        # ========================================================================
        # Final Summary
        # ========================================================================
        self._print_summary(train_results, test_results, comparison)
        
        return {
            'experiment_dir': self.experiment_dir,
            'train_results': train_results,
            'test_results': test_results,
            'comparison': comparison,
            'history': history
        }
    
    def _extract_predictions_per_frequency(
        self,
        model: torch.nn.Module,
        dataset: FrequencyExtractionDataset
    ) -> dict:
        """Extract predictions for each frequency separately."""
        model.eval()
        predictions = {i: [] for i in range(dataset.num_frequencies)}
        
        with torch.no_grad():
            for freq_idx in range(dataset.num_frequencies):
                model.reset_state()
                
                for t in range(dataset.num_time_samples):
                    idx = freq_idx * dataset.num_time_samples + t
                    input_tensor, _ = dataset[idx]
                    input_tensor = input_tensor.unsqueeze(0).to(self.device)
                    
                    pred = model(input_tensor, reset_state=False)
                    predictions[freq_idx].append(pred.cpu().item())
        
        # Convert to numpy arrays
        predictions = {k: np.array(v) for k, v in predictions.items()}
        
        return predictions
    
    def _print_results(self, train_results, test_results, comparison):
        """Print detailed evaluation results."""
        logger.info("\nTRAIN SET METRICS:")
        for metric, value in train_results['overall'].items():
            logger.info(f"  {metric}: {value:.6f}")
        
        logger.info("\nTEST SET METRICS:")
        for metric, value in test_results['overall'].items():
            logger.info(f"  {metric}: {value:.6f}")
        
        logger.info("\nPER-FREQUENCY METRICS (TEST SET):")
        for freq_idx, metrics in test_results['per_frequency'].items():
            freq_hz = self.config.get('data.frequencies')[freq_idx]
            logger.info(f"\n  Frequency {freq_idx+1} ({freq_hz} Hz):")
            for metric, value in metrics.items():
                if metric != 'num_samples':
                    logger.info(f"    {metric}: {value:.6f}")
    
    def _print_summary(self, train_results, test_results, comparison):
        """Print final summary."""
        logger.info("\n" + "="*80)
        logger.info("EXPERIMENT COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info(f"\nResults saved to: {self.experiment_dir}")
        logger.info(f"- Plots: {self.experiment_dir / 'plots'}")
        logger.info(f"- Checkpoints: {self.experiment_dir / 'checkpoints'}")
        logger.info(f"- Tensorboard logs: {self.experiment_dir / 'checkpoints' / 'tensorboard'}")
        
        logger.info("\n" + "="*80)
        logger.info("FINAL METRICS SUMMARY")
        logger.info("="*80)
        logger.info(f"Train MSE: {train_results['overall']['mse']:.6f}")
        logger.info(f"Test MSE:  {test_results['overall']['mse']:.6f}")
        logger.info(f"Train R²:  {train_results['overall']['r2_score']:.4f}")
        logger.info(f"Test R²:   {test_results['overall']['r2_score']:.4f}")
        logger.info(f"\nGeneralization Status: {comparison['overall_generalization']['status']}")
        
        if comparison['overall_generalization']['good']:
            logger.info("✅ SUCCESS: Model generalizes well to unseen noise!")
        else:
            logger.warning("⚠️  WARNING: Model may be overfitting. Review results.")
        
        logger.info("\n" + "="*80)


def main():
    """Main execution function."""
    # Create framework
    framework = ProductionMLFramework('config/config.yaml')
    
    # Set seed
    framework.set_seed(framework.config.get('reproducibility.seed', 42))
    
    # Setup device
    framework.setup_device()
    
    # Create experiment directory
    framework.create_experiment_directory()
    
    # Initialize plugins
    framework.initialize_plugins()
    
    # Run the pipeline
    results = framework.run()
    
    return results


if __name__ == '__main__':
    main()

