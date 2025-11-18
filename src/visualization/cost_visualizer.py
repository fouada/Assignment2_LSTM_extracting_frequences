"""
Cost Analysis Visualization Module
Creates professional visualizations for cost breakdown and optimization recommendations.

Author: Professional ML Engineering Team
Date: 2025
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
from typing import List, Dict
from pathlib import Path
import logging

from ..evaluation.cost_analysis import CostBreakdown, OptimizationRecommendation

logger = logging.getLogger(__name__)


class CostVisualizer:
    """
    Professional visualization for cost analysis.
    
    Creates:
    - Cost breakdown charts
    - Cloud provider comparison
    - Resource usage dashboard
    - Optimization recommendations visualization
    - ROI and efficiency plots
    """
    
    # Professional color scheme
    COLORS = {
        'primary': '#2E86AB',
        'secondary': '#A23B72',
        'success': '#06A77D',
        'warning': '#F18F01',
        'danger': '#C73E1D',
        'neutral': '#6C757D',
        'light': '#E9ECEF',
        'dark': '#212529',
    }
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style to use
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        # Set professional defaults
        plt.rcParams['figure.figsize'] = (16, 10)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 9
    
    def create_comprehensive_cost_dashboard(
        self,
        breakdown: CostBreakdown,
        recommendations: List[OptimizationRecommendation],
        save_path: Path
    ):
        """
        Create comprehensive cost analysis dashboard.
        
        Args:
            breakdown: Cost breakdown data
            recommendations: List of optimization recommendations
            save_path: Path to save the figure
        """
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Cost Breakdown Pie Chart
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_cost_breakdown_pie(ax1, breakdown)
        
        # 2. Cloud Provider Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_cloud_comparison(ax2, breakdown)
        
        # 3. Efficiency Score Gauge
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_efficiency_gauge(ax3, breakdown.efficiency_score)
        
        # 4. Resource Usage Bars
        ax4 = fig.add_subplot(gs[1, :2])
        self._plot_resource_usage(ax4, breakdown)
        
        # 5. Environmental Impact
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_environmental_impact(ax5, breakdown)
        
        # 6. Recommendations Priority Matrix
        ax6 = fig.add_subplot(gs[2, :])
        self._plot_recommendations_matrix(ax6, recommendations)
        
        # Main title
        fig.suptitle(
            'Comprehensive Cost Analysis & Optimization Dashboard',
            fontsize=16,
            fontweight='bold',
            y=0.98
        )
        
        # Save
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Cost dashboard saved to: {save_path}")
    
    def _plot_cost_breakdown_pie(self, ax, breakdown: CostBreakdown):
        """Plot cost breakdown as pie chart."""
        # Calculate component costs
        training_cost = breakdown.training_cost_usd
        inference_cost = breakdown.inference_cost_per_1000_samples_usd * 10  # Scaled for visibility
        
        if training_cost == 0:
            training_cost = 0.001  # Avoid zero division
        
        sizes = [training_cost, inference_cost]
        labels = ['Training Cost\n(Local)', 'Inference Cost\n(per 10K samples)']
        colors = [self.COLORS['primary'], self.COLORS['secondary']]
        explode = (0.05, 0.05)
        
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            explode=explode,
            shadow=True
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('Cost Breakdown', fontweight='bold', pad=10)
    
    def _plot_cloud_comparison(self, ax, breakdown: CostBreakdown):
        """Plot cloud provider cost comparison."""
        providers = ['AWS\nP3.2xlarge', 'Azure\nNC6 V3', 'GCP\nV100', 'Local']
        costs = [
            breakdown.aws_gpu_training_cost_usd,
            breakdown.azure_gpu_training_cost_usd,
            breakdown.gcp_gpu_training_cost_usd,
            breakdown.training_cost_usd
        ]
        
        colors_list = [
            self.COLORS['warning'],
            self.COLORS['primary'],
            self.COLORS['success'],
            self.COLORS['neutral']
        ]
        
        bars = ax.bar(providers, costs, color=colors_list, alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'${height:.2f}',
                ha='center',
                va='bottom',
                fontweight='bold',
                fontsize=9
            )
        
        ax.set_ylabel('Training Cost (USD)', fontweight='bold')
        ax.set_title('Cloud Provider Cost Comparison', fontweight='bold', pad=10)
        ax.grid(axis='y', alpha=0.3)
        
        # Highlight cheapest option
        min_idx = np.argmin(costs)
        bars[min_idx].set_edgecolor(self.COLORS['success'])
        bars[min_idx].set_linewidth(3)
    
    def _plot_efficiency_gauge(self, ax, efficiency_score: float):
        """Plot efficiency score as gauge chart."""
        # Create gauge
        theta = np.linspace(0, np.pi, 100)
        
        # Color zones
        colors_zones = [
            (0, 40, self.COLORS['danger']),
            (40, 70, self.COLORS['warning']),
            (70, 100, self.COLORS['success'])
        ]
        
        for start, end, color in colors_zones:
            theta_zone = np.linspace(start * np.pi / 100, end * np.pi / 100, 50)
            ax.fill_between(
                theta_zone,
                0.8,
                1.0,
                color=color,
                alpha=0.3
            )
        
        # Draw outer arc
        ax.plot(theta, np.ones_like(theta), 'k-', linewidth=2)
        ax.plot(theta, 0.8 * np.ones_like(theta), 'k-', linewidth=2)
        
        # Draw needle
        needle_angle = efficiency_score * np.pi / 100
        ax.arrow(
            0, 0,
            0.7 * np.cos(needle_angle),
            0.7 * np.sin(needle_angle),
            head_width=0.08,
            head_length=0.1,
            fc=self.COLORS['dark'],
            ec=self.COLORS['dark'],
            linewidth=2
        )
        
        # Center circle
        circle = plt.Circle((0, 0), 0.1, color=self.COLORS['dark'])
        ax.add_patch(circle)
        
        # Score text
        ax.text(
            0, -0.3,
            f'{efficiency_score:.1f}/100',
            ha='center',
            va='center',
            fontsize=16,
            fontweight='bold',
            color=self.COLORS['dark']
        )
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.5, 1.2)
        ax.axis('off')
        ax.set_title('Efficiency Score', fontweight='bold', pad=10)
    
    def _plot_resource_usage(self, ax, breakdown: CostBreakdown):
        """Plot resource usage comparison."""
        metrics = [
            'Model Size\n(MB)',
            'Peak Memory\n(MB)',
            'Inference Time\n(ms)',
            'Training Time\n(hours)'
        ]
        
        values = [
            breakdown.model_size_mb,
            breakdown.peak_memory_mb / 100,  # Scale for visibility
            breakdown.avg_inference_time_ms,
            breakdown.training_time_hours * 10  # Scale for visibility
        ]
        
        benchmarks = [5.0, 15.0, 0.5, 2.0]  # Example benchmarks
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, values, width, label='Current',
                       color=self.COLORS['primary'], alpha=0.8)
        bars2 = ax.bar(x + width/2, benchmarks, width, label='Benchmark',
                       color=self.COLORS['neutral'], alpha=0.6)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height,
                    f'{height:.2f}',
                    ha='center',
                    va='bottom',
                    fontsize=8
                )
        
        ax.set_ylabel('Value', fontweight='bold')
        ax.set_title('Resource Usage vs Benchmarks', fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_environmental_impact(self, ax, breakdown: CostBreakdown):
        """Plot environmental impact visualization."""
        # Convert to equivalents for better understanding
        carbon_kg = breakdown.carbon_footprint_kg_co2
        
        # Equivalents
        tree_months = carbon_kg / 0.021  # A tree absorbs ~21g CO2 per day
        car_miles = carbon_kg / 0.404  # Average car emits 404g CO2 per mile
        
        categories = ['CO₂\n(kg)', 'Tree-Months\nto Absorb', 'Car Miles\nEquivalent']
        values = [carbon_kg, tree_months / 30, car_miles]
        colors_list = [self.COLORS['danger'], self.COLORS['success'], self.COLORS['warning']]
        
        bars = ax.bar(categories, values, color=colors_list, alpha=0.7)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{height:.2f}',
                ha='center',
                va='bottom',
                fontweight='bold',
                fontsize=9
            )
        
        ax.set_title('Environmental Impact', fontweight='bold', pad=10)
        ax.set_ylabel('Value', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_recommendations_matrix(self, ax, recommendations: List[OptimizationRecommendation]):
        """Plot recommendations as priority/impact matrix."""
        if not recommendations:
            ax.text(0.5, 0.5, 'No recommendations available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            return
        
        # Prepare data
        priority_map = {'high': 3, 'medium': 2, 'low': 1}
        effort_map = {'easy': 1, 'moderate': 2, 'complex': 3}
        
        x_values = [effort_map[rec.implementation_effort] for rec in recommendations]
        y_values = [priority_map[rec.priority] for rec in recommendations]
        sizes = [rec.estimated_cost_reduction * 10 for rec in recommendations]
        
        # Category colors
        category_colors = {
            'training': self.COLORS['primary'],
            'model': self.COLORS['secondary'],
            'inference': self.COLORS['success'],
            'deployment': self.COLORS['warning']
        }
        colors_list = [category_colors.get(rec.category, self.COLORS['neutral'])
                      for rec in recommendations]
        
        # Scatter plot
        scatter = ax.scatter(
            x_values,
            y_values,
            s=sizes,
            c=colors_list,
            alpha=0.6,
            edgecolors='black',
            linewidth=2
        )
        
        # Add labels with arrows
        for i, rec in enumerate(recommendations):
            ax.annotate(
                f"{i+1}",
                (x_values[i], y_values[i]),
                xytext=(0, 0),
                textcoords='offset points',
                ha='center',
                va='center',
                fontweight='bold',
                fontsize=10,
                color='white'
            )
        
        ax.set_xlabel('Implementation Effort', fontweight='bold')
        ax.set_ylabel('Priority', fontweight='bold')
        ax.set_title('Optimization Recommendations Matrix\n(Size = Cost Reduction %)', 
                    fontweight='bold', pad=10)
        
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['Easy', 'Moderate', 'Complex'])
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(['Low', 'Medium', 'High'])
        
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.5, 3.5)
        ax.set_ylim(0.5, 3.5)
        
        # Add quadrant labels
        ax.text(1.5, 2.5, 'Quick Wins', ha='center', va='center',
               fontsize=10, alpha=0.3, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='green', alpha=0.1))
        ax.text(2.5, 2.5, 'Strategic\nProjects', ha='center', va='center',
               fontsize=10, alpha=0.3, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='blue', alpha=0.1))
        
        # Legend
        legend_elements = [
            mpatches.Patch(color=color, label=cat.capitalize())
            for cat, color in category_colors.items()
        ]
        ax.legend(handles=legend_elements, loc='upper right', 
                 title='Category', framealpha=0.9)
        
        # Add recommendation list below
        rec_text = '\n'.join([
            f"{i+1}. [{rec.priority.upper()}] {rec.recommendation[:80]}..."
            for i, rec in enumerate(recommendations[:5])  # Show top 5
        ])
        
        ax.text(
            0.02, -0.15,
            'Top Recommendations:\n' + rec_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        )
    
    def create_cost_comparison_chart(
        self,
        breakdown: CostBreakdown,
        save_path: Path
    ):
        """
        Create detailed cost comparison chart.
        
        Args:
            breakdown: Cost breakdown data
            save_path: Path to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # 1. Training cost breakdown
        ax1 = axes[0, 0]
        training_components = {
            'Energy': breakdown.training_energy_kwh * breakdown.ELECTRICITY_COST_PER_KWH
            if hasattr(breakdown, 'ELECTRICITY_COST_PER_KWH') else breakdown.training_cost_usd * 0.8,
            'Compute': breakdown.training_cost_usd * 0.2,
        }
        ax1.bar(training_components.keys(), training_components.values(),
               color=[self.COLORS['primary'], self.COLORS['secondary']], alpha=0.7)
        ax1.set_title('Training Cost Components', fontweight='bold')
        ax1.set_ylabel('Cost (USD)', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Inference cost scaling
        ax2 = axes[0, 1]
        sample_counts = np.array([1e3, 1e4, 1e5, 1e6])
        inference_costs = sample_counts * breakdown.inference_cost_per_1000_samples_usd / 1000
        ax2.plot(sample_counts, inference_costs, marker='o', linewidth=2,
                color=self.COLORS['primary'], markersize=8)
        ax2.set_xlabel('Number of Samples', fontweight='bold')
        ax2.set_ylabel('Inference Cost (USD)', fontweight='bold')
        ax2.set_title('Inference Cost Scaling', fontweight='bold')
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)
        
        # 3. Memory usage breakdown
        ax3 = axes[1, 0]
        memory_components = {
            'Model': breakdown.model_size_mb,
            'Peak\nUsage': breakdown.peak_memory_mb,
            'Average\nUsage': breakdown.avg_memory_mb
        }
        bars = ax3.bar(memory_components.keys(), memory_components.values(),
                      color=[self.COLORS['success'], self.COLORS['warning'], self.COLORS['primary']],
                      alpha=0.7)
        ax3.set_ylabel('Memory (MB)', fontweight='bold')
        ax3.set_title('Memory Usage Breakdown', fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontweight='bold')
        
        # 4. Cost per performance metric
        ax4 = axes[1, 1]
        metrics_cost = {
            'Cost per\nMSE point': breakdown.cost_per_mse_point,
            'Cost per\n1K inferences': breakdown.inference_cost_per_1000_samples_usd * 1000,
            'Cost per\nParameter\n(×10⁶)': (breakdown.training_cost_usd / breakdown.total_parameters) * 1e6
        }
        bars = ax4.bar(metrics_cost.keys(), metrics_cost.values(),
                      color=[self.COLORS['danger'], self.COLORS['warning'], self.COLORS['success']],
                      alpha=0.7)
        ax4.set_ylabel('Cost (USD)', fontweight='bold')
        ax4.set_title('Cost Efficiency Metrics', fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:.4f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        plt.suptitle('Detailed Cost Analysis', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Cost comparison chart saved to: {save_path}")


def create_cost_visualizer() -> CostVisualizer:
    """
    Factory function to create cost visualizer.
    
    Returns:
        CostVisualizer instance
    """
    return CostVisualizer()

