"""
ISO/IEC 25010 Quality Metrics Collector
Automated collection and reporting of quality metrics
"""

import time
import psutil
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np


class QualityMetricsCollector:
    """
    Collects and reports ISO/IEC 25010 quality metrics
    
    Tracks:
    - Performance metrics
    - Reliability metrics
    - Resource utilization
    - System health
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize metrics collector
        
        Args:
            output_dir: Directory for metrics reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {
            'performance': {},
            'reliability': {},
            'security': {},
            'usability': {},
            'maintainability': {},
            'portability': {},
            'functional_suitability': {},
            'compatibility': {}
        }
        
        self.start_time = time.time()
        self.error_count = 0
        self.warning_count = 0
        
    # ========================
    # Performance Metrics
    # ========================
    
    def collect_performance_metrics(
        self,
        operation_name: str,
        execution_time: float,
        memory_used: Optional[float] = None,
        gpu_memory_used: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Collect performance metrics for an operation
        
        Args:
            operation_name: Name of the operation
            execution_time: Time taken in seconds
            memory_used: Memory used in MB
            gpu_memory_used: GPU memory used in MB
            
        Returns:
            Performance metrics dictionary
        """
        metrics = {
            'operation': operation_name,
            'timestamp': datetime.now().isoformat(),
            'time_behaviour': {
                'execution_time': execution_time,
                'throughput': 1.0 / execution_time if execution_time > 0 else 0
            },
            'resource_utilization': {
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_mb': memory_used or self._get_memory_usage(),
                'memory_percent': psutil.virtual_memory().percent
            }
        }
        
        if gpu_memory_used is not None:
            metrics['resource_utilization']['gpu_memory_mb'] = gpu_memory_used
            
        self.metrics['performance'][operation_name] = metrics
        return metrics
    
    def measure_latency(self, func, *args, **kwargs) -> tuple:
        """
        Measure function latency
        
        Args:
            func: Function to measure
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Tuple of (result, execution_time)
        """
        start = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start
        
        return result, execution_time
    
    # ========================
    # Reliability Metrics
    # ========================
    
    def collect_reliability_metrics(self) -> Dict[str, Any]:
        """
        Collect reliability metrics
        
        Returns:
            Reliability metrics dictionary
        """
        uptime = time.time() - self.start_time
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'maturity': {
                'error_count': self.error_count,
                'warning_count': self.warning_count,
                'uptime_seconds': uptime,
                'uptime_hours': uptime / 3600
            },
            'availability': {
                'uptime_percentage': 100.0,  # Calculated if we track downtime
                'mtbf_hours': uptime / 3600 if self.error_count == 0 else (uptime / 3600) / self.error_count
            },
            'fault_tolerance': {
                'recovered_errors': 0,  # Track with error handling
                'unrecovered_errors': self.error_count
            }
        }
        
        self.metrics['reliability'] = metrics
        return metrics
    
    def record_error(self, error_type: str, severity: str = 'ERROR'):
        """
        Record an error occurrence
        
        Args:
            error_type: Type of error
            severity: Severity level (ERROR, WARNING, CRITICAL)
        """
        if severity == 'ERROR' or severity == 'CRITICAL':
            self.error_count += 1
        elif severity == 'WARNING':
            self.warning_count += 1
            
        if 'error_log' not in self.metrics['reliability']:
            self.metrics['reliability']['error_log'] = []
            
        self.metrics['reliability']['error_log'].append({
            'timestamp': datetime.now().isoformat(),
            'type': error_type,
            'severity': severity
        })
    
    # ========================
    # Security Metrics
    # ========================
    
    def collect_security_metrics(
        self,
        validation_checks: int,
        validation_failures: int,
        auth_attempts: int = 0,
        auth_failures: int = 0
    ) -> Dict[str, Any]:
        """
        Collect security metrics
        
        Args:
            validation_checks: Number of validation checks performed
            validation_failures: Number of validation failures
            auth_attempts: Authentication attempts
            auth_failures: Failed authentication attempts
            
        Returns:
            Security metrics dictionary
        """
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'integrity': {
                'validation_checks': validation_checks,
                'validation_failures': validation_failures,
                'validation_success_rate': 
                    (validation_checks - validation_failures) / validation_checks * 100
                    if validation_checks > 0 else 0
            },
            'accountability': {
                'audit_logs_enabled': True,  # Track if logging is enabled
                'tracked_operations': validation_checks
            }
        }
        
        if auth_attempts > 0:
            metrics['authenticity'] = {
                'auth_attempts': auth_attempts,
                'auth_failures': auth_failures,
                'auth_success_rate': 
                    (auth_attempts - auth_failures) / auth_attempts * 100
            }
        
        self.metrics['security'] = metrics
        return metrics
    
    # ========================
    # Usability Metrics
    # ========================
    
    def collect_usability_metrics(
        self,
        command_success_count: int,
        command_failure_count: int,
        avg_task_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Collect usability metrics
        
        Args:
            command_success_count: Successful commands
            command_failure_count: Failed commands
            avg_task_time: Average time to complete tasks
            
        Returns:
            Usability metrics dictionary
        """
        total_commands = command_success_count + command_failure_count
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'operability': {
                'command_success_count': command_success_count,
                'command_failure_count': command_failure_count,
                'success_rate': 
                    command_success_count / total_commands * 100
                    if total_commands > 0 else 0
            }
        }
        
        if avg_task_time is not None:
            metrics['learnability'] = {
                'avg_task_completion_time': avg_task_time,
                'efficiency_score': 100 / avg_task_time if avg_task_time > 0 else 0
            }
        
        self.metrics['usability'] = metrics
        return metrics
    
    # ========================
    # Maintainability Metrics
    # ========================
    
    def collect_maintainability_metrics(
        self,
        test_coverage: float,
        code_complexity: float,
        documentation_coverage: float
    ) -> Dict[str, Any]:
        """
        Collect maintainability metrics
        
        Args:
            test_coverage: Test coverage percentage
            code_complexity: Average cyclomatic complexity
            documentation_coverage: Documentation coverage percentage
            
        Returns:
            Maintainability metrics dictionary
        """
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'testability': {
                'test_coverage_percent': test_coverage,
                'coverage_status': 'GOOD' if test_coverage >= 80 else 'NEEDS_IMPROVEMENT'
            },
            'analysability': {
                'code_complexity': code_complexity,
                'complexity_status': 'GOOD' if code_complexity <= 10 else 'HIGH',
                'documentation_coverage': documentation_coverage
            },
            'modifiability': {
                'modularity_score': 85.0,  # Can be calculated from code structure
                'coupling_score': 7.0  # Can be calculated from dependencies
            }
        }
        
        self.metrics['maintainability'] = metrics
        return metrics
    
    # ========================
    # Functional Suitability
    # ========================
    
    def collect_functional_metrics(
        self,
        implemented_features: int,
        required_features: int,
        test_pass_rate: float
    ) -> Dict[str, Any]:
        """
        Collect functional suitability metrics
        
        Args:
            implemented_features: Number of implemented features
            required_features: Number of required features
            test_pass_rate: Test pass rate percentage
            
        Returns:
            Functional metrics dictionary
        """
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'functional_completeness': {
                'implemented_features': implemented_features,
                'required_features': required_features,
                'completeness_percent': 
                    implemented_features / required_features * 100
                    if required_features > 0 else 0
            },
            'functional_correctness': {
                'test_pass_rate': test_pass_rate,
                'defect_count': self.error_count,
                'correctness_status': 'GOOD' if test_pass_rate >= 95 else 'NEEDS_IMPROVEMENT'
            }
        }
        
        self.metrics['functional_suitability'] = metrics
        return metrics
    
    # ========================
    # Report Generation
    # ========================
    
    def generate_compliance_report(self, format: str = 'json') -> Path:
        """
        Generate ISO/IEC 25010 compliance report
        
        Args:
            format: Output format ('json' or 'yaml')
            
        Returns:
            Path to generated report
        """
        report = {
            'report_metadata': {
                'standard': 'ISO/IEC 25010:2011',
                'generated_at': datetime.now().isoformat(),
                'report_version': '1.0',
                'system': 'LSTM Frequency Extraction System'
            },
            'quality_characteristics': self.metrics,
            'overall_scores': self._calculate_overall_scores(),
            'compliance_status': self._assess_compliance()
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'json':
            report_path = self.output_dir / f'compliance_report_{timestamp}.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
        else:
            report_path = self.output_dir / f'compliance_report_{timestamp}.yaml'
            with open(report_path, 'w') as f:
                yaml.dump(report, f, default_flow_style=False)
        
        # Also generate a summary
        self._generate_summary_report(report)
        
        return report_path
    
    def _calculate_overall_scores(self) -> Dict[str, float]:
        """Calculate overall quality scores"""
        scores = {}
        
        # Performance score
        if self.metrics['performance']:
            avg_cpu = np.mean([
                m['resource_utilization']['cpu_percent'] 
                for m in self.metrics['performance'].values()
            ])
            scores['performance_efficiency'] = min(100, 100 - avg_cpu)
        
        # Reliability score
        if self.metrics.get('reliability'):
            error_rate = self.error_count / max(1, time.time() - self.start_time)
            scores['reliability'] = max(0, 100 - error_rate * 100)
        
        # Security score
        if self.metrics.get('security'):
            val_success = self.metrics['security']['integrity']['validation_success_rate']
            scores['security'] = val_success
        
        # Usability score
        if self.metrics.get('usability'):
            scores['usability'] = self.metrics['usability']['operability']['success_rate']
        
        # Maintainability score
        if self.metrics.get('maintainability'):
            scores['maintainability'] = self.metrics['maintainability']['testability']['test_coverage_percent']
        
        # Overall score
        if scores:
            scores['overall'] = np.mean(list(scores.values()))
        
        return scores
    
    def _assess_compliance(self) -> Dict[str, str]:
        """Assess compliance status for each characteristic"""
        scores = self._calculate_overall_scores()
        
        def get_status(score: float) -> str:
            if score >= 90:
                return 'EXCELLENT'
            elif score >= 75:
                return 'GOOD'
            elif score >= 60:
                return 'ACCEPTABLE'
            else:
                return 'NEEDS_IMPROVEMENT'
        
        return {
            characteristic: get_status(score)
            for characteristic, score in scores.items()
        }
    
    def _generate_summary_report(self, report: Dict[str, Any]):
        """Generate human-readable summary report"""
        summary_path = self.output_dir / 'compliance_summary.txt'
        
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ISO/IEC 25010 COMPLIANCE REPORT SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Generated: {report['report_metadata']['generated_at']}\n")
            f.write(f"System: {report['report_metadata']['system']}\n\n")
            
            f.write("OVERALL SCORES:\n")
            f.write("-" * 80 + "\n")
            for characteristic, score in report['overall_scores'].items():
                status = report['compliance_status'].get(characteristic, 'N/A')
                f.write(f"{characteristic:30s}: {score:6.2f}% - {status}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("RECOMMENDATIONS:\n")
            f.write("-" * 80 + "\n")
            
            for characteristic, status in report['compliance_status'].items():
                if status == 'NEEDS_IMPROVEMENT':
                    f.write(f"⚠️  {characteristic}: Requires immediate attention\n")
                elif status == 'ACCEPTABLE':
                    f.write(f"ℹ️  {characteristic}: Room for improvement\n")
            
            f.write("\n" + "=" * 80 + "\n")
    
    # ========================
    # Utility Methods
    # ========================
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def reset(self):
        """Reset all metrics"""
        self.metrics = {
            'performance': {},
            'reliability': {},
            'security': {},
            'usability': {},
            'maintainability': {},
            'portability': {},
            'functional_suitability': {},
            'compatibility': {}
        }
        self.start_time = time.time()
        self.error_count = 0
        self.warning_count = 0


# Context manager for performance measurement
class PerformanceTimer:
    """Context manager for measuring operation performance"""
    
    def __init__(
        self,
        collector: QualityMetricsCollector,
        operation_name: str
    ):
        self.collector = collector
        self.operation_name = operation_name
        self.start_time = None
        self.start_memory = None
        
    def __enter__(self):
        self.start_time = time.time()
        self.start_memory = self.collector._get_memory_usage()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        execution_time = time.time() - self.start_time
        memory_used = self.collector._get_memory_usage() - self.start_memory
        
        self.collector.collect_performance_metrics(
            self.operation_name,
            execution_time,
            memory_used
        )
        
        if exc_type is not None:
            self.collector.record_error(
                str(exc_type.__name__),
                'CRITICAL' if exc_type == Exception else 'ERROR'
            )
        
        return False  # Don't suppress exceptions

