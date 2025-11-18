"""
Main Entry Point with Interactive Dashboard Integration
LSTM Frequency Extraction System with Real-time Monitoring

Author: Professional ML Engineering Team
Date: 2025

Usage:
    python main_with_dashboard.py                # Train and launch dashboard
    python main_with_dashboard.py --no-dashboard # Train only (no dashboard)
    python main_with_dashboard.py --dashboard-only # Launch dashboard for latest experiment
"""

import torch
import yaml
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import argparse
import sys
import threading
import time

from src.data.signal_generator import create_train_test_generators
from src.data.dataset import create_dataloaders, FrequencyExtractionDataset
from src.models.lstm_extractor import create_model
from src.training.trainer import LSTMTrainer
from src.evaluation.metrics import evaluate_model, compare_train_test_performance
from src.visualization.plotter import create_all_visualizations
from src.visualization.live_monitor import create_live_monitor, DashboardDataExporter

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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='LSTM Frequency Extraction with Interactive Dashboard'
    )
    
    parser.add_argument(
        '--no-dashboard',
        action='store_true',
        help='Train without launching dashboard'
    )
    
    parser.add_argument(
        '--dashboard-only',
        action='store_true',
        help='Only launch dashboard for latest experiment (no training)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8050,
        help='Dashboard port (default: 8050)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file'
    )
    
    return parser.parse_args()


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


def launch_dashboard_async(exp_dir: Path, port: int):
    """Launch dashboard in a separate thread."""
    try:
        from src.visualization.interactive_dashboard import create_dashboard
        
        logger.info(f"\n{'='*80}")
        logger.info("LAUNCHING INTERACTIVE DASHBOARD")
        logger.info(f"{'='*80}")
        logger.info(f"Dashboard will be available at: http://localhost:{port}")
        logger.info(f"{'='*80}\n")
        
        # Give training a moment to initialize
        time.sleep(2)
        
        dashboard = create_dashboard(experiment_dir=exp_dir, port=port)
        dashboard.run(debug=False)
        
    except Exception as e:
        logger.error(f"Error launching dashboard: {e}")


def main():
    """Main execution function."""
    args = parse_args()
    
    logger.info("="*80)
    logger.info("LSTM Frequency Extraction - Professional Implementation")
    logger.info("with Interactive Dashboard Integration")
    logger.info("="*80)
    
    # Dashboard-only mode
    if args.dashboard_only:
        logger.info("\nDashboard-only mode: Launching dashboard for latest experiment...")
        
        # Find latest experiment
        experiments_dir = Path('experiments')
        if not experiments_dir.exists():
            logger.error("No experiments directory found. Run training first.")
            sys.exit(1)
        
        exp_dirs = [d for d in experiments_dir.iterdir() 
                   if d.is_dir() and d.name.startswith('lstm_frequency_extraction_')]
        
        if not exp_dirs:
            logger.error("No experiments found. Run training first.")
            sys.exit(1)
        
        latest_exp = max(exp_dirs, key=lambda p: p.stat().st_mtime)
        logger.info(f"Using experiment: {latest_exp}")
        
        from src.visualization.interactive_dashboard import create_dashboard
        dashboard = create_dashboard(experiment_dir=latest_exp, port=args.port)
        dashboard.run(debug=False)
        return
    
    # Load configuration
    config = load_config(args.config)
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
    
    # Initialize live monitor for dashboard
    monitor = None
    if not args.no_dashboard:
        logger.info("\nInitializing live training monitor...")
        monitor = create_live_monitor(exp_dir, auto_start=True)
        monitor.set_total_epochs(config['training']['epochs'])
        
        # Launch dashboard in background thread
        dashboard_thread = threading.Thread(
            target=launch_dashboard_async,
            args=(exp_dir, args.port),
            daemon=True
        )
        dashboard_thread.start()
    
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
    
    train_loader, test_loader = create_dataloaders(
        train_generator=train_generator,
        test_generator=test_generator,
        batch_size=config['training']['batch_size'],
        normalize=True,
        device='cpu'
    )
    
    # Keep datasets for later visualization
    train_dataset = FrequencyExtractionDataset(train_generator, normalize=True)
    test_dataset = FrequencyExtractionDataset(test_generator, normalize=True)
    
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
    
    # Train with live monitoring
    history = trainer.train()
    
    # Update monitor with training history
    if monitor:
        for epoch_idx, (train_loss, val_loss, lr) in enumerate(
            zip(history['train_loss'], history['val_loss'], history['learning_rate'])
        ):
            monitor.update_epoch(
                epoch=epoch_idx + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                learning_rate=lr
            )
    
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
    
    # Update monitor with test results
    if monitor:
        monitor.update_test_results(
            overall_metrics=test_results['overall'],
            per_frequency_metrics=test_results['per_frequency']
        )
    
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
    
    # Create all static visualizations
    create_all_visualizations(
        test_dataset=test_dataset,
        predictions_dict=test_predictions_dict,
        frequencies=config['data']['frequencies'],
        history=history,
        train_metrics=train_results,
        test_metrics=test_results,
        save_dir=exp_dir / 'plots'
    )
    
    # Export data for dashboard
    if monitor:
        logger.info("\nExporting data for dashboard...")
        
        # Export training history
        DashboardDataExporter.export_training_history(
            history, exp_dir / 'training_history.json'
        )
        
        # Export test results
        DashboardDataExporter.export_test_results(
            test_results, exp_dir / 'test_results.json'
        )
        
        # Export predictions
        time = test_dataset.generator.time_vector
        mixed_signal = test_dataset.mixed_signal
        targets_dict = test_dataset.targets
        
        DashboardDataExporter.export_predictions(
            test_predictions_dict, targets_dict, time, mixed_signal, exp_dir
        )
        
        # Update final predictions in monitor
        for freq_idx in range(len(config['data']['frequencies'])):
            monitor.update_predictions(
                freq_idx, time, targets_dict[freq_idx],
                test_predictions_dict[freq_idx], mixed_signal
            )
        
        # Stop monitor
        monitor.stop()
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    logger.info(f"\nResults saved to: {exp_dir}")
    logger.info(f"- Plots: {exp_dir / 'plots'}")
    logger.info(f"- Checkpoints: {exp_dir / 'checkpoints'}")
    
    if not args.no_dashboard:
        logger.info(f"\n🌐 Interactive Dashboard: http://localhost:{args.port}")
        logger.info("   (Dashboard is running in the background)")
    
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
    
    if not args.no_dashboard:
        logger.info("\n💡 Tip: Keep this terminal open to maintain dashboard access")
        logger.info("Press Ctrl+C to stop the dashboard and exit")
        
        try:
            # Keep main thread alive for dashboard
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\n\nShutting down...")
            logger.info("Experiment completed successfully!")


if __name__ == '__main__':
    main()

