#!/usr/bin/env python3
"""
Comprehensive Cost Analysis Report Generator
Standalone script for customers to analyze costs and get optimization recommendations.

Usage:
    python cost_analysis_report.py [--experiment-dir PATH] [--config PATH]

Author: Professional ML Engineering Team
Date: 2025
"""

import torch
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.lstm_extractor import create_model
from src.evaluation.cost_analysis import create_cost_analyzer
from src.visualization.cost_visualizer import create_cost_visualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cost_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate comprehensive cost analysis report',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze latest experiment
    python cost_analysis_report.py
    
    # Analyze specific experiment
    python cost_analysis_report.py --experiment-dir experiments/lstm_frequency_extraction_20251118_002838
    
    # Use custom config
    python cost_analysis_report.py --config config/config_production.yaml
        """
    )
    
    parser.add_argument(
        '--experiment-dir',
        type=str,
        default=None,
        help='Path to experiment directory (default: latest in experiments/)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for cost analysis (default: experiment_dir/cost_analysis/)'
    )
    
    parser.add_argument(
        '--training-time',
        type=float,
        default=None,
        help='Training time in seconds (optional, will estimate if not provided)'
    )
    
    return parser.parse_args()


def find_latest_experiment() -> Path:
    """Find the latest experiment directory."""
    experiments_dir = Path('experiments')
    
    if not experiments_dir.exists():
        raise FileNotFoundError("No experiments directory found. Please run training first.")
    
    experiment_dirs = sorted(
        [d for d in experiments_dir.iterdir() if d.is_dir()],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    
    if not experiment_dirs:
        raise FileNotFoundError("No experiment directories found. Please run training first.")
    
    latest = experiment_dirs[0]
    logger.info(f"Using latest experiment: {latest}")
    
    return latest


def load_experiment_config(exp_dir: Path) -> dict:
    """Load configuration from experiment directory."""
    config_path = exp_dir / 'config.yaml'
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found in experiment: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def load_trained_model(exp_dir: Path, config: dict, device: torch.device):
    """Load trained model from checkpoint."""
    checkpoint_path = exp_dir / 'checkpoints' / 'best_model.pt'
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    # Create model
    model = create_model(config['model'])
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded from: {checkpoint_path}")
    
    # Get training info from checkpoint
    training_info = {
        'final_mse': checkpoint.get('val_loss', 0.001),
        'epochs_trained': checkpoint.get('epoch', 10)
    }
    
    return model, training_info


def estimate_training_time(config: dict, device: torch.device) -> float:
    """
    Estimate training time based on configuration and device.
    
    This is a rough estimate. For accurate analysis, provide actual training time.
    """
    # Base time estimates (seconds per epoch)
    base_times = {
        'cuda': 30,
        'mps': 50,
        'cpu': 120
    }
    
    device_type = 'cuda' if device.type == 'cuda' else ('mps' if device.type == 'mps' else 'cpu')
    base_time = base_times[device_type]
    
    # Adjust for model size
    hidden_size = config['model'].get('hidden_size', 128)
    size_factor = hidden_size / 128
    
    # Adjust for batch size
    batch_size = config['training'].get('batch_size', 32)
    batch_factor = 32 / batch_size
    
    epochs = config['training'].get('epochs', 50)
    
    estimated_time = base_time * size_factor * batch_factor * epochs
    
    logger.warning(f"Training time estimated at {estimated_time:.0f} seconds. "
                  "For accurate analysis, provide actual training time with --training-time")
    
    return estimated_time


def create_sample_input(config: dict, device: torch.device) -> torch.Tensor:
    """Create sample input tensor for inference benchmarking."""
    input_size = config['model'].get('input_size', 5)
    sample = torch.randn(1, 1, input_size).to(device)
    return sample


def generate_cost_analysis_report(
    exp_dir: Path,
    config: dict,
    output_dir: Path,
    training_time_seconds: float
):
    """
    Generate comprehensive cost analysis report.
    
    Args:
        exp_dir: Experiment directory
        config: Configuration dictionary
        output_dir: Output directory for reports
        training_time_seconds: Actual training time in seconds
    """
    logger.info("\n" + "="*80)
    logger.info("COMPREHENSIVE COST ANALYSIS REPORT")
    logger.info("="*80)
    
    # Setup device
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
    
    logger.info(f"Device: {device}")
    
    # Load model
    model, training_info = load_trained_model(exp_dir, config, device)
    
    # Create cost analyzer
    analyzer = create_cost_analyzer(model, device)
    
    # Create sample input for inference benchmarking
    sample_input = create_sample_input(config, device)
    
    # Perform cost analysis
    logger.info("\nPerforming cost analysis...")
    cost_breakdown = analyzer.analyze_costs(
        training_time_seconds=training_time_seconds,
        sample_input=sample_input,
        final_mse=training_info['final_mse']
    )
    
    # Generate optimization recommendations
    logger.info("\nGenerating optimization recommendations...")
    recommendations = analyzer.generate_recommendations(
        breakdown=cost_breakdown,
        current_config=config
    )
    
    # Print recommendations
    analyzer.print_recommendations(recommendations)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export analysis to JSON
    json_path = output_dir / 'cost_analysis.json'
    analyzer.export_analysis(cost_breakdown, recommendations, json_path)
    
    # Create visualizations
    logger.info("\nCreating visualizations...")
    visualizer = create_cost_visualizer()
    
    # Comprehensive dashboard
    dashboard_path = output_dir / 'cost_analysis_dashboard.png'
    visualizer.create_comprehensive_cost_dashboard(
        breakdown=cost_breakdown,
        recommendations=recommendations,
        save_path=dashboard_path
    )
    
    # Detailed comparison chart
    comparison_path = output_dir / 'cost_comparison_detailed.png'
    visualizer.create_cost_comparison_chart(
        breakdown=cost_breakdown,
        save_path=comparison_path
    )
    
    # Generate summary report
    summary_path = output_dir / 'COST_ANALYSIS_SUMMARY.md'
    generate_markdown_summary(
        cost_breakdown,
        recommendations,
        summary_path,
        exp_dir
    )
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("COST ANALYSIS COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    logger.info(f"\nOutputs saved to: {output_dir}")
    logger.info(f"- Dashboard:     {dashboard_path}")
    logger.info(f"- Comparison:    {comparison_path}")
    logger.info(f"- JSON Data:     {json_path}")
    logger.info(f"- Summary:       {summary_path}")
    logger.info("\n" + "="*80)
    
    # Print key insights
    print_key_insights(cost_breakdown, recommendations)


def generate_markdown_summary(
    breakdown,
    recommendations,
    save_path: Path,
    exp_dir: Path
):
    """Generate markdown summary report."""
    content = f"""# Cost Analysis Summary Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Experiment:** {exp_dir.name}  
**Device:** {breakdown.device_type}

---

## 💰 Cost Breakdown

### Training Costs
- **Training Time:** {breakdown.training_time_hours:.2f} hours ({breakdown.training_time_seconds/60:.1f} minutes)
- **Energy Consumption:** {breakdown.training_energy_kwh:.4f} kWh
- **Local Cost:** ${breakdown.training_cost_usd:.4f}

### Cloud Computing Costs
| Provider | Instance Type | Cost (USD) |
|----------|--------------|------------|
| AWS | P3.2xlarge (V100) | ${breakdown.aws_gpu_training_cost_usd:.2f} |
| Azure | NC6 V3 (V100) | ${breakdown.azure_gpu_training_cost_usd:.2f} |
| GCP | V100 Instance | ${breakdown.gcp_gpu_training_cost_usd:.2f} |

💡 **Recommendation:** {"Local training is most cost-effective" if breakdown.training_cost_usd < min(breakdown.aws_gpu_training_cost_usd, breakdown.azure_gpu_training_cost_usd, breakdown.gcp_gpu_training_cost_usd) else "Consider cloud training for faster results"}

---

## 🚀 Inference Costs

- **Average Inference Time:** {breakdown.avg_inference_time_ms:.3f} ms
- **Throughput:** {breakdown.inference_throughput_samples_per_sec:.1f} samples/sec
- **Cost per 1,000 samples:** ${breakdown.inference_cost_per_1000_samples_usd:.6f}
- **Cost per 1,000,000 samples:** ${breakdown.aws_cpu_inference_cost_per_million_usd:.2f}

---

## 💾 Resource Usage

- **Model Size:** {breakdown.model_size_mb:.2f} MB
- **Total Parameters:** {breakdown.total_parameters:,}
- **Peak Memory Usage:** {breakdown.peak_memory_mb:.1f} MB
- **Average Memory Usage:** {breakdown.avg_memory_mb:.1f} MB

---

## 🌍 Environmental Impact

- **Carbon Footprint:** {breakdown.carbon_footprint_kg_co2:.4f} kg CO₂
- **Equivalent to:**
  - 🚗 {breakdown.carbon_footprint_kg_co2 / 0.404:.2f} miles driven
  - 🌳 {breakdown.carbon_footprint_kg_co2 / 0.021 / 30:.1f} tree-months to absorb

---

## 📊 Efficiency Metrics

- **Efficiency Score:** {breakdown.efficiency_score:.1f}/100 {"🏆" if breakdown.efficiency_score >= 80 else "⚡" if breakdown.efficiency_score >= 60 else "⚠️"}
- **Cost per MSE Point:** ${breakdown.cost_per_mse_point:.4f}

---

## 🎯 Top Optimization Recommendations

"""
    
    # Add top 5 recommendations
    for i, rec in enumerate(recommendations[:5], 1):
        priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        content += f"""
### {i}. [{priority_emoji[rec.priority]} {rec.priority.upper()}] {rec.category.capitalize()}

**Recommendation:** {rec.recommendation}

**Expected Improvement:** {rec.expected_improvement}

**Estimated Cost Reduction:** {rec.estimated_cost_reduction:.1f}%

**Implementation Effort:** {rec.implementation_effort}

"""
        
        if rec.code_example:
            content += f"""**Code Example:**
```python
{rec.code_example.strip()}
```

"""
    
    content += f"""---

## 📈 ROI Analysis

- **Total Investment (Training):** ${breakdown.training_cost_usd:.4f}
- **Potential Savings (from top 3 recommendations):** {sum(r.estimated_cost_reduction for r in recommendations[:3]):.1f}%
- **Projected Savings:** ${breakdown.training_cost_usd * sum(r.estimated_cost_reduction for r in recommendations[:3]) / 100:.4f}

---

## 🔍 Detailed Analysis

For complete analysis including:
- Interactive cost breakdown dashboard
- Cloud provider comparison
- Resource usage charts
- Environmental impact visualization
- Recommendations matrix

See:
- `cost_analysis_dashboard.png`
- `cost_comparison_detailed.png`
- `cost_analysis.json`

---

## 📞 Next Steps

1. **Review Recommendations:** Prioritize high-impact, low-effort optimizations
2. **Implement Changes:** Start with "Quick Wins" (high priority, easy implementation)
3. **Measure Impact:** Re-run cost analysis after optimizations
4. **Iterate:** Continuously optimize based on new insights

---

*Generated by LSTM Frequency Extraction Cost Analysis System*  
*For questions or support, refer to the documentation.*
"""
    
    with open(save_path, 'w') as f:
        f.write(content)
    
    logger.info(f"Summary report saved to: {save_path}")


def print_key_insights(breakdown, recommendations):
    """Print key insights to console."""
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    # Cost insights
    print("\n💰 COST INSIGHTS:")
    print(f"  - Training on {breakdown.device_type}: ${breakdown.training_cost_usd:.4f}")
    
    cheapest_cloud = min(
        ('AWS', breakdown.aws_gpu_training_cost_usd),
        ('Azure', breakdown.azure_gpu_training_cost_usd),
        ('GCP', breakdown.gcp_gpu_training_cost_usd),
        key=lambda x: x[1]
    )
    print(f"  - Cheapest cloud option: {cheapest_cloud[0]} at ${cheapest_cloud[1]:.2f}")
    
    if breakdown.training_cost_usd < cheapest_cloud[1]:
        savings = ((cheapest_cloud[1] - breakdown.training_cost_usd) / cheapest_cloud[1]) * 100
        print(f"  - 💡 Local training saves {savings:.1f}% vs cloud")
    
    # Efficiency insights
    print("\n⚡ EFFICIENCY INSIGHTS:")
    print(f"  - Overall efficiency score: {breakdown.efficiency_score:.1f}/100")
    
    if breakdown.efficiency_score >= 80:
        print("  - ✅ Excellent! Model is highly efficient")
    elif breakdown.efficiency_score >= 60:
        print("  - ⚡ Good, but room for improvement")
    else:
        print("  - ⚠️  Consider optimization recommendations")
    
    # Top recommendation
    if recommendations:
        print("\n🎯 TOP RECOMMENDATION:")
        top = recommendations[0]
        print(f"  - [{top.priority.upper()}] {top.recommendation}")
        print(f"  - Expected cost reduction: {top.estimated_cost_reduction:.1f}%")
        print(f"  - Implementation: {top.implementation_effort}")
    
    # Environmental
    print("\n🌍 ENVIRONMENTAL IMPACT:")
    print(f"  - Carbon footprint: {breakdown.carbon_footprint_kg_co2:.4f} kg CO₂")
    
    if breakdown.carbon_footprint_kg_co2 < 0.1:
        print("  - ✅ Low environmental impact")
    else:
        print(f"  - 🌳 Plant {breakdown.carbon_footprint_kg_co2 / 0.021 / 30:.0f} tree-months to offset")
    
    print("\n" + "="*80)


def main():
    """Main execution function."""
    args = parse_arguments()
    
    try:
        # Find experiment directory
        if args.experiment_dir:
            exp_dir = Path(args.experiment_dir)
            if not exp_dir.exists():
                raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
        else:
            exp_dir = find_latest_experiment()
        
        # Load configuration
        if Path(args.config).exists():
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Using config: {args.config}")
        else:
            config = load_experiment_config(exp_dir)
            logger.info(f"Using experiment config: {exp_dir / 'config.yaml'}")
        
        # Setup output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = exp_dir / 'cost_analysis'
        
        # Get training time
        if args.training_time:
            training_time = args.training_time
            logger.info(f"Using provided training time: {training_time:.0f} seconds")
        else:
            # Try to estimate or use default
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            training_time = estimate_training_time(config, device)
        
        # Generate report
        generate_cost_analysis_report(
            exp_dir=exp_dir,
            config=config,
            output_dir=output_dir,
            training_time_seconds=training_time
        )
        
    except Exception as e:
        logger.error(f"Error generating cost analysis: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

