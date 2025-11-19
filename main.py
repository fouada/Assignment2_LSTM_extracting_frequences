"""
Main Entry Point for LSTM Frequency Extraction System
Professional MIT-Level Implementation

Author: Professional ML Engineering Team
Date: 2025
"""

import torch
import yaml
import logging
import time
from pathlib import Path
from datetime import datetime
import numpy as np

from src.data.signal_generator import create_train_test_generators
from src.data.dataset import create_dataloaders, FrequencyExtractionDataset
from src.models.lstm_extractor import create_model
from src.training.trainer import LSTMTrainer
from src.evaluation.metrics import evaluate_model, compare_train_test_performance
from src.evaluation.cost_analysis import create_cost_analyzer
from src.visualization.plotter import create_all_visualizations
from src.visualization.cost_visualizer import create_cost_visualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = 'config/config.yaml') -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def setup_device(config: dict) -> torch.device:
    """Setup computation device."""
    device_config = config['compute']['device']
    
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


def create_experiment_directory(config: dict) -> Path:
    """Create directory for experiment outputs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = config['experiment']['name']
    exp_dir = Path(config['experiment']['save_dir']) / f"{exp_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / 'plots').mkdir(exist_ok=True)
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    
    logger.info(f"Experiment directory: {exp_dir}")
    return exp_dir


def extract_predictions_per_frequency(
    model: torch.nn.Module,
    dataset: FrequencyExtractionDataset,
    device: torch.device
) -> dict:
    """
    Extract predictions for each frequency separately.
    
    Returns:
        Dictionary mapping freq_idx to prediction array
    """
    model.eval()
    predictions = {i: [] for i in range(dataset.num_frequencies)}
    
    with torch.no_grad():
        for freq_idx in range(dataset.num_frequencies):
            model.reset_state()
            
            for t in range(dataset.num_time_samples):
                idx = freq_idx * dataset.num_time_samples + t
                input_tensor, _ = dataset[idx]
                input_tensor = input_tensor.unsqueeze(0).to(device)
                
                pred = model(input_tensor, reset_state=False)
                predictions[freq_idx].append(pred.cpu().item())
    
    # Convert to numpy arrays
    predictions = {k: np.array(v) for k, v in predictions.items()}
    
    return predictions


def main():
    """Main execution function."""
    logger.info("="*80)
    logger.info("LSTM Frequency Extraction - Professional Implementation")
    logger.info("="*80)
    
    # Load configuration
    config = load_config()
    logger.info("Configuration loaded successfully")
    
    # Set random seed
    set_seed(config['reproducibility']['seed'])
    
    # Setup device
    device = setup_device(config)
    
    # Create experiment directory
    exp_dir = create_experiment_directory(config)
    
    # Save configuration
    with open(exp_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # ========================================================================
    # Step 1: Generate Data
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 1: Data Generation")
    logger.info("="*80)
    
    train_generator, test_generator = create_train_test_generators(
        frequencies=config['data']['frequencies'],
        sampling_rate=config['data']['sampling_rate'],
        duration=config['data']['duration'],
        amplitude_range=config['data']['amplitude_range'],
        phase_range=config['data']['phase_range'],
        train_seed=config['data']['train_seed'],
        test_seed=config['data']['test_seed']
    )
    
    # ========================================================================
    # Step 2: Create Datasets and Dataloaders
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 2: Dataset Creation")
    logger.info("="*80)

    # CRITICAL FIX: create_dataloaders now returns datasets too
    train_loader, test_loader, train_dataset, test_dataset = create_dataloaders(
        train_generator=train_generator,
        test_generator=test_generator,
        batch_size=config['training']['batch_size'],
        normalize=True,
        device='cpu'
    )

    # Datasets are now created correctly with proper normalization sharing
    
    # ========================================================================
    # Step 3: Create Model
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 3: Model Creation")
    logger.info("="*80)
    
    model = create_model(config['model'])
    model = model.to(device)
    
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
        config=config['training'],
        device=device,
        experiment_dir=exp_dir / 'checkpoints'
    )
    
    # Measure training time for cost analysis
    training_start_time = time.time()
    history = trainer.train()
    training_time_seconds = time.time() - training_start_time
    
    logger.info(f"Training completed in {training_time_seconds:.2f} seconds ({training_time_seconds/60:.2f} minutes)")
    
    # ========================================================================
    # Step 5: Evaluation
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 5: Model Evaluation")
    logger.info("="*80)
    
    # Load best model
    best_model_path = exp_dir / 'checkpoints' / 'best_model.pt'
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Best model loaded for evaluation")
    
    # Evaluate on train set
    logger.info("\nEvaluating on TRAIN set...")
    train_results = evaluate_model(model, train_loader, device, compute_per_frequency=True)
    
    # Evaluate on test set
    logger.info("\nEvaluating on TEST set...")
    test_results = evaluate_model(model, test_loader, device, compute_per_frequency=True)
    
    # Compare performance
    logger.info("\n" + "-"*80)
    logger.info("GENERALIZATION ANALYSIS")
    logger.info("-"*80)
    comparison = compare_train_test_performance(train_results, test_results)
    
    # Print detailed results
    logger.info("\nTRAIN SET METRICS:")
    for metric, value in train_results['overall'].items():
        logger.info(f"  {metric}: {value:.6f}")
    
    logger.info("\nTEST SET METRICS:")
    for metric, value in test_results['overall'].items():
        logger.info(f"  {metric}: {value:.6f}")
    
    logger.info("\nPER-FREQUENCY METRICS (TEST SET):")
    for freq_idx, metrics in test_results['per_frequency'].items():
        freq_hz = config['data']['frequencies'][freq_idx]
        logger.info(f"\n  Frequency {freq_idx+1} ({freq_hz} Hz):")
        for metric, value in metrics.items():
            if metric != 'num_samples':
                logger.info(f"    {metric}: {value:.6f}")
    
    # ========================================================================
    # Step 6: Visualization
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 6: Creating Visualizations")
    logger.info("="*80)
    
    # Extract predictions per frequency for visualization
    test_predictions_dict = extract_predictions_per_frequency(model, test_dataset, device)
    
    # Create all visualizations
    create_all_visualizations(
        test_dataset=test_dataset,
        predictions_dict=test_predictions_dict,
        frequencies=config['data']['frequencies'],
        history=history,
        train_metrics=train_results,
        test_metrics=test_results,
        save_dir=exp_dir / 'plots'
    )
    
    # ========================================================================
    # Step 7: Cost Analysis (Optional but Recommended)
    # ========================================================================
    if config.get('cost_analysis', {}).get('enabled', True):
        logger.info("\n" + "="*80)
        logger.info("STEP 7: Cost Analysis & Optimization Recommendations")
        logger.info("="*80)
        
        try:
            # Create cost analyzer
            cost_analyzer = create_cost_analyzer(model, device)
            
            # Create sample input for inference benchmarking
            sample_input = torch.randn(1, 1, config['model']['input_size']).to(device)
            
            # Perform cost analysis
            cost_breakdown = cost_analyzer.analyze_costs(
                training_time_seconds=training_time_seconds,
                sample_input=sample_input,
                final_mse=test_results['overall']['mse']
            )
            
            # Generate optimization recommendations
            recommendations = cost_analyzer.generate_recommendations(
                breakdown=cost_breakdown,
                current_config=config
            )
            
            # Print recommendations
            cost_analyzer.print_recommendations(recommendations)
            
            # Create cost analysis directory
            cost_analysis_dir = exp_dir / 'cost_analysis'
            cost_analysis_dir.mkdir(exist_ok=True)
            
            # Export analysis to JSON
            cost_analyzer.export_analysis(
                breakdown=cost_breakdown,
                recommendations=recommendations,
                save_path=cost_analysis_dir / 'cost_analysis.json'
            )
            
            # Create visualizations
            cost_visualizer = create_cost_visualizer()
            
            cost_visualizer.create_comprehensive_cost_dashboard(
                breakdown=cost_breakdown,
                recommendations=recommendations,
                save_path=cost_analysis_dir / 'cost_dashboard.png'
            )
            
            cost_visualizer.create_cost_comparison_chart(
                breakdown=cost_breakdown,
                save_path=cost_analysis_dir / 'cost_comparison.png'
            )
            
            logger.info(f"Cost analysis saved to: {cost_analysis_dir}")
            
        except Exception as e:
            logger.warning(f"Cost analysis failed (non-critical): {e}")
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    logger.info(f"\nResults saved to: {exp_dir}")
    logger.info(f"- Plots: {exp_dir / 'plots'}")
    logger.info(f"- Checkpoints: {exp_dir / 'checkpoints'}")
    logger.info(f"- Cost Analysis: {exp_dir / 'cost_analysis'}")
    logger.info(f"- Tensorboard logs: {exp_dir / 'checkpoints' / 'tensorboard'}")
    
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
    logger.info("💡 TIP: Run 'python cost_analysis_report.py' for detailed cost insights!")
    logger.info("="*80)


if __name__ == '__main__':
    main()

