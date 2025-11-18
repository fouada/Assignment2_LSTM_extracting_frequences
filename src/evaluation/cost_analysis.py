"""
Comprehensive Cost Analysis Module
Provides detailed cost breakdown and optimization recommendations for LSTM deployment.

Author: Professional ML Engineering Team
Date: 2025
"""

import torch
import time
import psutil
import platform
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import logging
import json
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class CostBreakdown:
    """Detailed cost breakdown structure."""
    
    # Training costs
    training_time_seconds: float
    training_time_hours: float
    training_energy_kwh: float
    training_cost_usd: float
    
    # Inference costs
    avg_inference_time_ms: float
    inference_throughput_samples_per_sec: float
    inference_cost_per_1000_samples_usd: float
    
    # Resource usage
    peak_memory_mb: float
    avg_memory_mb: float
    model_size_mb: float
    total_parameters: int
    
    # Cloud computing estimates
    aws_gpu_training_cost_usd: float
    aws_cpu_inference_cost_per_million_usd: float
    azure_gpu_training_cost_usd: float
    gcp_gpu_training_cost_usd: float
    
    # Environmental impact
    carbon_footprint_kg_co2: float
    
    # Efficiency metrics
    cost_per_mse_point: float
    efficiency_score: float
    
    # Additional metadata
    device_type: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class OptimizationRecommendation:
    """Single optimization recommendation."""
    
    category: str  # "model", "training", "inference", "deployment"
    priority: str  # "high", "medium", "low"
    recommendation: str
    expected_improvement: str
    implementation_effort: str  # "easy", "moderate", "complex"
    estimated_cost_reduction: float  # percentage
    code_example: Optional[str] = None


class CostAnalyzer:
    """
    Comprehensive cost analysis and optimization recommendation system.
    
    Analyzes:
    - Training costs (time, energy, cloud computing)
    - Inference costs (latency, throughput)
    - Resource usage (memory, storage)
    - Environmental impact
    - Cost-effectiveness metrics
    
    Provides:
    - Detailed cost breakdown
    - Optimization recommendations
    - Comparison with benchmarks
    - ROI analysis
    """
    
    # Industry standard pricing (2025 estimates)
    CLOUD_PRICING = {
        'aws': {
            'p3.2xlarge': 3.06,  # $/hour (V100 GPU)
            'g4dn.xlarge': 0.526,  # $/hour (T4 GPU)
            'lambda_inference': 0.0000167,  # $/request for 1GB-s
        },
        'azure': {
            'nc6': 0.90,  # $/hour (K80 GPU)
            'nc6_v3': 3.06,  # $/hour (V100 GPU)
        },
        'gcp': {
            'n1-highmem-8-v100': 2.48,  # $/hour (V100 GPU)
            'n1-standard-4-t4': 0.35,  # $/hour (T4 GPU)
        }
    }
    
    # Energy costs (average US rates)
    ELECTRICITY_COST_PER_KWH = 0.13  # USD
    
    # Average power consumption estimates (Watts)
    POWER_CONSUMPTION = {
        'cpu': 65,
        'cuda': 250,  # NVIDIA V100/A100
        'mps': 50,  # Apple Silicon
    }
    
    # Carbon intensity (kg CO2 per kWh - US average)
    CARBON_INTENSITY = 0.42
    
    def __init__(self, model: torch.nn.Module, device: torch.device):
        """
        Initialize cost analyzer.
        
        Args:
            model: PyTorch model to analyze
            device: Device used for computation
        """
        self.model = model
        self.device = device
        self.device_type = self._get_device_type()
        
        # System info
        self.system_info = self._collect_system_info()
        
        # Benchmarks for comparison
        self.benchmarks = {
            'training_time_hours': 0.2,  # 12 minutes
            'inference_time_ms': 0.5,
            'model_size_mb': 5.0,
            'peak_memory_mb': 1500,
        }
        
        logger.info(f"CostAnalyzer initialized for device: {self.device_type}")
    
    def _get_device_type(self) -> str:
        """Get device type string."""
        if self.device.type == 'cuda':
            return 'cuda'
        elif self.device.type == 'mps':
            return 'mps'
        else:
            return 'cpu'
    
    def _collect_system_info(self) -> Dict:
        """Collect system information."""
        info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'cpu_count': psutil.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total / (1024**3),
        }
        
        if self.device.type == 'cuda':
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        return info
    
    def calculate_model_size(self) -> Tuple[float, int]:
        """
        Calculate model size and parameter count.
        
        Returns:
            Tuple of (size_in_mb, total_parameters)
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        
        # Calculate size (assuming float32 = 4 bytes)
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        total_size_mb = (param_size + buffer_size) / (1024**2)
        
        logger.info(f"Model size: {total_size_mb:.2f} MB")
        logger.info(f"Total parameters: {total_params:,}")
        
        return total_size_mb, total_params
    
    def measure_inference_time(
        self,
        sample_input: torch.Tensor,
        num_warmup: int = 10,
        num_iterations: int = 100
    ) -> Tuple[float, float]:
        """
        Measure inference time with warmup.
        
        Args:
            sample_input: Sample input tensor
            num_warmup: Number of warmup iterations
            num_iterations: Number of measurement iterations
        
        Returns:
            Tuple of (avg_time_ms, throughput_samples_per_sec)
        """
        self.model.eval()
        
        with torch.no_grad():
            # Warmup
            for _ in range(num_warmup):
                _ = self.model(sample_input, reset_state=True)
            
            # Measure
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            for _ in range(num_iterations):
                _ = self.model(sample_input, reset_state=True)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
        
        total_time = end_time - start_time
        avg_time_ms = (total_time / num_iterations) * 1000
        throughput = num_iterations / total_time
        
        logger.info(f"Average inference time: {avg_time_ms:.3f} ms")
        logger.info(f"Throughput: {throughput:.1f} samples/sec")
        
        return avg_time_ms, throughput
    
    def estimate_training_energy(self, training_time_hours: float) -> float:
        """
        Estimate energy consumption for training.
        
        Args:
            training_time_hours: Total training time in hours
        
        Returns:
            Energy consumption in kWh
        """
        power_watts = self.POWER_CONSUMPTION.get(self.device_type, 100)
        energy_kwh = (power_watts * training_time_hours) / 1000
        
        logger.info(f"Estimated energy consumption: {energy_kwh:.4f} kWh")
        
        return energy_kwh
    
    def calculate_cloud_costs(
        self,
        training_time_hours: float,
        num_inference_samples: int = 1_000_000
    ) -> Dict[str, float]:
        """
        Calculate cloud computing costs.
        
        Args:
            training_time_hours: Training duration in hours
            num_inference_samples: Number of inference samples for cost estimation
        
        Returns:
            Dictionary with cloud provider costs
        """
        costs = {}
        
        # AWS costs
        costs['aws_gpu_training'] = training_time_hours * self.CLOUD_PRICING['aws']['p3.2xlarge']
        costs['aws_gpu_training_budget'] = training_time_hours * self.CLOUD_PRICING['aws']['g4dn.xlarge']
        costs['aws_inference_per_million'] = (num_inference_samples / 1000) * self.CLOUD_PRICING['aws']['lambda_inference']
        
        # Azure costs
        costs['azure_gpu_training'] = training_time_hours * self.CLOUD_PRICING['azure']['nc6_v3']
        
        # GCP costs
        costs['gcp_gpu_training'] = training_time_hours * self.CLOUD_PRICING['gcp']['n1-highmem-8-v100']
        costs['gcp_gpu_training_budget'] = training_time_hours * self.CLOUD_PRICING['gcp']['n1-standard-4-t4']
        
        logger.info(f"AWS GPU training cost: ${costs['aws_gpu_training']:.2f}")
        logger.info(f"Azure GPU training cost: ${costs['azure_gpu_training']:.2f}")
        logger.info(f"GCP GPU training cost: ${costs['gcp_gpu_training']:.2f}")
        
        return costs
    
    def analyze_costs(
        self,
        training_time_seconds: float,
        sample_input: torch.Tensor,
        final_mse: float,
        peak_memory_mb: Optional[float] = None,
        avg_memory_mb: Optional[float] = None
    ) -> CostBreakdown:
        """
        Perform comprehensive cost analysis.
        
        Args:
            training_time_seconds: Total training time in seconds
            sample_input: Sample input for inference benchmarking
            final_mse: Final MSE achieved by the model
            peak_memory_mb: Peak memory usage during training
            avg_memory_mb: Average memory usage during training
        
        Returns:
            CostBreakdown object with detailed analysis
        """
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE COST ANALYSIS")
        logger.info("="*80)
        
        # Training costs
        training_time_hours = training_time_seconds / 3600
        training_energy_kwh = self.estimate_training_energy(training_time_hours)
        training_cost_local = training_energy_kwh * self.ELECTRICITY_COST_PER_KWH
        
        # Model size
        model_size_mb, total_params = self.calculate_model_size()
        
        # Inference costs
        avg_inference_ms, throughput = self.measure_inference_time(sample_input)
        inference_cost_per_1000 = (avg_inference_ms * 1000) * self.CLOUD_PRICING['aws']['lambda_inference']
        
        # Memory usage
        if peak_memory_mb is None:
            peak_memory_mb = psutil.virtual_memory().used / (1024**2)
        if avg_memory_mb is None:
            avg_memory_mb = peak_memory_mb * 0.8
        
        # Cloud costs
        cloud_costs = self.calculate_cloud_costs(training_time_hours)
        
        # Environmental impact
        carbon_footprint = training_energy_kwh * self.CARBON_INTENSITY
        
        # Efficiency metrics
        cost_per_mse_point = training_cost_local / max(final_mse, 1e-6)
        efficiency_score = self._calculate_efficiency_score(
            training_time_hours, model_size_mb, final_mse, avg_inference_ms
        )
        
        breakdown = CostBreakdown(
            training_time_seconds=training_time_seconds,
            training_time_hours=training_time_hours,
            training_energy_kwh=training_energy_kwh,
            training_cost_usd=training_cost_local,
            avg_inference_time_ms=avg_inference_ms,
            inference_throughput_samples_per_sec=throughput,
            inference_cost_per_1000_samples_usd=inference_cost_per_1000,
            peak_memory_mb=peak_memory_mb,
            avg_memory_mb=avg_memory_mb,
            model_size_mb=model_size_mb,
            total_parameters=total_params,
            aws_gpu_training_cost_usd=cloud_costs['aws_gpu_training'],
            aws_cpu_inference_cost_per_million_usd=cloud_costs['aws_inference_per_million'],
            azure_gpu_training_cost_usd=cloud_costs['azure_gpu_training'],
            gcp_gpu_training_cost_usd=cloud_costs['gcp_gpu_training'],
            carbon_footprint_kg_co2=carbon_footprint,
            cost_per_mse_point=cost_per_mse_point,
            efficiency_score=efficiency_score,
            device_type=self.device_type
        )
        
        self._print_cost_summary(breakdown)
        
        return breakdown
    
    def _calculate_efficiency_score(
        self,
        training_hours: float,
        model_size_mb: float,
        final_mse: float,
        inference_ms: float
    ) -> float:
        """
        Calculate overall efficiency score (0-100).
        
        Higher is better. Considers:
        - Training time vs benchmark
        - Model size vs benchmark
        - Accuracy (MSE)
        - Inference speed vs benchmark
        """
        # Normalize each metric (lower is better for time/size, higher is better for accuracy)
        training_score = max(0, 100 * (1 - training_hours / self.benchmarks['training_time_hours']))
        size_score = max(0, 100 * (1 - model_size_mb / self.benchmarks['model_size_mb']))
        accuracy_score = max(0, 100 * (1 - min(final_mse, 0.01) / 0.01))
        inference_score = max(0, 100 * (1 - inference_ms / self.benchmarks['inference_time_ms']))
        
        # Weighted average
        weights = {'training': 0.25, 'size': 0.15, 'accuracy': 0.40, 'inference': 0.20}
        efficiency = (
            weights['training'] * training_score +
            weights['size'] * size_score +
            weights['accuracy'] * accuracy_score +
            weights['inference'] * inference_score
        )
        
        return efficiency
    
    def _print_cost_summary(self, breakdown: CostBreakdown):
        """Print formatted cost summary."""
        logger.info("\n" + "-"*80)
        logger.info("TRAINING COSTS")
        logger.info("-"*80)
        logger.info(f"Training Time:       {breakdown.training_time_hours:.2f} hours ({breakdown.training_time_seconds/60:.1f} minutes)")
        logger.info(f"Energy Consumption:  {breakdown.training_energy_kwh:.4f} kWh")
        logger.info(f"Local Cost:          ${breakdown.training_cost_usd:.4f}")
        logger.info(f"AWS GPU Cost:        ${breakdown.aws_gpu_training_cost_usd:.2f}")
        logger.info(f"Azure GPU Cost:      ${breakdown.azure_gpu_training_cost_usd:.2f}")
        logger.info(f"GCP GPU Cost:        ${breakdown.gcp_gpu_training_cost_usd:.2f}")
        
        logger.info("\n" + "-"*80)
        logger.info("INFERENCE COSTS")
        logger.info("-"*80)
        logger.info(f"Avg Inference Time:  {breakdown.avg_inference_time_ms:.3f} ms")
        logger.info(f"Throughput:          {breakdown.inference_throughput_samples_per_sec:.1f} samples/sec")
        logger.info(f"Cost per 1K samples: ${breakdown.inference_cost_per_1000_samples_usd:.6f}")
        logger.info(f"Cost per 1M samples: ${breakdown.aws_cpu_inference_cost_per_million_usd:.2f} (AWS)")
        
        logger.info("\n" + "-"*80)
        logger.info("RESOURCE USAGE")
        logger.info("-"*80)
        logger.info(f"Model Size:          {breakdown.model_size_mb:.2f} MB")
        logger.info(f"Total Parameters:    {breakdown.total_parameters:,}")
        logger.info(f"Peak Memory:         {breakdown.peak_memory_mb:.1f} MB")
        logger.info(f"Avg Memory:          {breakdown.avg_memory_mb:.1f} MB")
        
        logger.info("\n" + "-"*80)
        logger.info("ENVIRONMENTAL & EFFICIENCY")
        logger.info("-"*80)
        logger.info(f"Carbon Footprint:    {breakdown.carbon_footprint_kg_co2:.4f} kg CO₂")
        logger.info(f"Efficiency Score:    {breakdown.efficiency_score:.1f}/100")
        logger.info(f"Cost per MSE point:  ${breakdown.cost_per_mse_point:.2f}")
        logger.info("="*80)
    
    def generate_recommendations(
        self,
        breakdown: CostBreakdown,
        current_config: Dict
    ) -> List[OptimizationRecommendation]:
        """
        Generate personalized optimization recommendations.
        
        Args:
            breakdown: Cost breakdown from analysis
            current_config: Current model/training configuration
        
        Returns:
            List of prioritized recommendations
        """
        recommendations = []
        
        # Analyze training time
        if breakdown.training_time_hours > self.benchmarks['training_time_hours'] * 1.5:
            recommendations.append(OptimizationRecommendation(
                category="training",
                priority="high",
                recommendation="Reduce training time by optimizing batch size and learning rate",
                expected_improvement="20-30% faster training",
                implementation_effort="easy",
                estimated_cost_reduction=25.0,
                code_example="""
# Increase batch size (if memory allows)
config['training']['batch_size'] = 64  # from 32

# Use learning rate finder for optimal LR
config['training']['learning_rate'] = 0.003  # optimized value
"""
            ))
        
        # Analyze model size
        if breakdown.model_size_mb > self.benchmarks['model_size_mb'] * 1.2:
            recommendations.append(OptimizationRecommendation(
                category="model",
                priority="medium",
                recommendation="Consider model compression techniques to reduce size",
                expected_improvement="30-50% size reduction with <2% accuracy loss",
                implementation_effort="moderate",
                estimated_cost_reduction=15.0,
                code_example="""
# Option 1: Reduce hidden size
config['model']['hidden_size'] = 96  # from 128

# Option 2: Use model quantization (post-training)
model_int8 = torch.quantization.quantize_dynamic(
    model, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8
)
"""
            ))
        
        # Analyze inference time
        if breakdown.avg_inference_time_ms > self.benchmarks['inference_time_ms'] * 1.5:
            recommendations.append(OptimizationRecommendation(
                category="inference",
                priority="high",
                recommendation="Optimize inference with batching and/or model optimization",
                expected_improvement="50-70% faster inference",
                implementation_effort="easy",
                estimated_cost_reduction=40.0,
                code_example="""
# Batch inference (process multiple samples together)
def batch_inference(model, inputs, batch_size=32):
    results = []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i+batch_size]
        with torch.no_grad():
            outputs = model(batch)
        results.extend(outputs)
    return results
"""
            ))
        
        # Memory optimization
        if breakdown.peak_memory_mb > self.benchmarks['peak_memory_mb']:
            recommendations.append(OptimizationRecommendation(
                category="training",
                priority="medium",
                recommendation="Reduce memory usage with gradient accumulation",
                expected_improvement="50% memory reduction",
                implementation_effort="moderate",
                estimated_cost_reduction=10.0,
                code_example="""
# Gradient accumulation
accumulation_steps = 4
for i, (inputs, targets) in enumerate(train_loader):
    outputs = model(inputs)
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
"""
            ))
        
        # Cloud deployment recommendations
        if breakdown.aws_gpu_training_cost_usd > 1.0:
            recommendations.append(OptimizationRecommendation(
                category="deployment",
                priority="medium",
                recommendation="Use spot instances or preemptible VMs for training",
                expected_improvement="60-80% cost reduction for cloud training",
                implementation_effort="easy",
                estimated_cost_reduction=70.0,
                code_example="""
# AWS Spot Instance example
# - Use g4dn.xlarge instead of p3.2xlarge
# - Estimated cost: $0.16/hour (70% savings)
# - Training time may increase slightly due to lower-end GPU

# Implement checkpointing for spot instance interruptions
if checkpoint_exists:
    model.load_state_dict(torch.load('checkpoint.pt'))
"""
            ))
        
        # Mixed precision training
        if self.device_type == 'cuda':
            recommendations.append(OptimizationRecommendation(
                category="training",
                priority="high",
                recommendation="Enable mixed precision training (FP16) for faster training",
                expected_improvement="2-3x faster training with same accuracy",
                implementation_effort="easy",
                estimated_cost_reduction=50.0,
                code_example="""
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for inputs, targets in train_loader:
    optimizer.zero_grad()
    
    # Enable automatic mixed precision
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
"""
            ))
        
        # Environmental recommendations
        if breakdown.carbon_footprint_kg_co2 > 0.1:
            recommendations.append(OptimizationRecommendation(
                category="deployment",
                priority="low",
                recommendation="Consider training during off-peak hours or using renewable energy",
                expected_improvement="Reduce carbon footprint by 30-50%",
                implementation_effort="easy",
                estimated_cost_reduction=5.0,
                code_example="""
# Schedule training during off-peak hours (lower carbon intensity)
# Use cloud regions with renewable energy:
# - AWS: us-west-2 (Oregon) - 85% renewable
# - GCP: us-central1 (Iowa) - 95% carbon-free
# - Azure: North Europe - high renewable percentage
"""
            ))
        
        # Early stopping optimization
        if 'early_stopping_patience' in current_config.get('training', {}):
            patience = current_config['training']['early_stopping_patience']
            if patience > 5:
                recommendations.append(OptimizationRecommendation(
                    category="training",
                    priority="low",
                    recommendation="Reduce early stopping patience to save training time",
                    expected_improvement="10-20% faster training convergence",
                    implementation_effort="easy",
                    estimated_cost_reduction=15.0,
                    code_example="""
# Reduce patience from 10 to 5 epochs
config['training']['early_stopping_patience'] = 5

# Add min_delta for more aggressive early stopping
config['training']['early_stopping_min_delta'] = 0.0001
"""
                ))
        
        # Sort by priority and cost reduction
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(
            key=lambda x: (priority_order[x.priority], -x.estimated_cost_reduction)
        )
        
        logger.info(f"\nGenerated {len(recommendations)} optimization recommendations")
        
        return recommendations
    
    def print_recommendations(self, recommendations: List[OptimizationRecommendation]):
        """Print recommendations in a formatted way."""
        logger.info("\n" + "="*80)
        logger.info("OPTIMIZATION RECOMMENDATIONS")
        logger.info("="*80)
        
        for i, rec in enumerate(recommendations, 1):
            logger.info(f"\n{i}. [{rec.priority.upper()}] {rec.category.upper()}")
            logger.info(f"   Recommendation: {rec.recommendation}")
            logger.info(f"   Expected Improvement: {rec.expected_improvement}")
            logger.info(f"   Estimated Cost Reduction: {rec.estimated_cost_reduction:.1f}%")
            logger.info(f"   Implementation Effort: {rec.implementation_effort}")
            
            if rec.code_example:
                logger.info(f"   Code Example:")
                for line in rec.code_example.strip().split('\n'):
                    logger.info(f"   {line}")
        
        logger.info("\n" + "="*80)
    
    def export_analysis(
        self,
        breakdown: CostBreakdown,
        recommendations: List[OptimizationRecommendation],
        save_path: Path
    ):
        """
        Export analysis to JSON file.
        
        Args:
            breakdown: Cost breakdown
            recommendations: List of recommendations
            save_path: Path to save JSON file
        """
        data = {
            'cost_breakdown': {
                'training': {
                    'time_hours': breakdown.training_time_hours,
                    'energy_kwh': breakdown.training_energy_kwh,
                    'cost_local_usd': breakdown.training_cost_usd,
                    'cost_aws_usd': breakdown.aws_gpu_training_cost_usd,
                    'cost_azure_usd': breakdown.azure_gpu_training_cost_usd,
                    'cost_gcp_usd': breakdown.gcp_gpu_training_cost_usd,
                },
                'inference': {
                    'avg_time_ms': breakdown.avg_inference_time_ms,
                    'throughput_samples_per_sec': breakdown.inference_throughput_samples_per_sec,
                    'cost_per_1000_samples_usd': breakdown.inference_cost_per_1000_samples_usd,
                },
                'resources': {
                    'model_size_mb': breakdown.model_size_mb,
                    'total_parameters': breakdown.total_parameters,
                    'peak_memory_mb': breakdown.peak_memory_mb,
                    'avg_memory_mb': breakdown.avg_memory_mb,
                },
                'environmental': {
                    'carbon_footprint_kg_co2': breakdown.carbon_footprint_kg_co2,
                },
                'efficiency': {
                    'efficiency_score': breakdown.efficiency_score,
                    'cost_per_mse_point': breakdown.cost_per_mse_point,
                },
                'system': {
                    'device_type': breakdown.device_type,
                    'timestamp': breakdown.timestamp,
                }
            },
            'recommendations': [
                {
                    'category': rec.category,
                    'priority': rec.priority,
                    'recommendation': rec.recommendation,
                    'expected_improvement': rec.expected_improvement,
                    'implementation_effort': rec.implementation_effort,
                    'estimated_cost_reduction': rec.estimated_cost_reduction,
                    'code_example': rec.code_example,
                }
                for rec in recommendations
            ],
            'system_info': self.system_info
        }
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Cost analysis exported to: {save_path}")


def create_cost_analyzer(model: torch.nn.Module, device: torch.device) -> CostAnalyzer:
    """
    Factory function to create cost analyzer.
    
    Args:
        model: PyTorch model
        device: Computation device
    
    Returns:
        CostAnalyzer instance
    """
    return CostAnalyzer(model, device)

