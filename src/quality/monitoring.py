"""
Performance and Reliability Monitoring
ISO/IEC 25010 Compliance
"""

import time
import psutil
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from collections import deque
import json

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Performance monitoring for ISO/IEC 25010 compliance
    
    Tracks:
    - Time behaviour: Execution times, latency
    - Resource utilization: CPU, memory, GPU
    - Capacity: Throughput, scalability
    """
    
    def __init__(self, history_size: int = 1000):
        """
        Initialize performance monitor
        
        Args:
            history_size: Number of measurements to keep in history
        """
        self.history_size = history_size
        self.measurements = {
            'execution_times': deque(maxlen=history_size),
            'memory_usage': deque(maxlen=history_size),
            'cpu_usage': deque(maxlen=history_size),
            'gpu_usage': deque(maxlen=history_size) if TORCH_AVAILABLE else None
        }
        self.operation_stats = {}
        
    def measure_operation(
        self,
        operation_name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> tuple:
        """
        Measure operation performance
        
        Args:
            operation_name: Name of operation
            func: Function to measure
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Tuple of (result, metrics)
        """
        # Baseline measurements
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_cpu = psutil.cpu_percent(interval=0.1)
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.synchronize()
            start_gpu_mem = torch.cuda.memory_allocated() / 1024**2
        else:
            start_gpu_mem = None
        
        # Execute operation
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
            logger.error(f"Operation {operation_name} failed: {e}")
        
        # Final measurements
        end_time = time.time()
        end_memory = self._get_memory_usage()
        end_cpu = psutil.cpu_percent(interval=0.1)
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.synchronize()
            end_gpu_mem = torch.cuda.memory_allocated() / 1024**2
        else:
            end_gpu_mem = None
        
        # Calculate metrics
        metrics = {
            'operation': operation_name,
            'timestamp': datetime.now().isoformat(),
            'execution_time': end_time - start_time,
            'memory_delta_mb': end_memory - start_memory,
            'cpu_percent': (start_cpu + end_cpu) / 2,
            'success': success,
            'error': error
        }
        
        if start_gpu_mem is not None and end_gpu_mem is not None:
            metrics['gpu_memory_delta_mb'] = end_gpu_mem - start_gpu_mem
            metrics['gpu_memory_mb'] = end_gpu_mem
        
        # Record measurements
        self.measurements['execution_times'].append(metrics['execution_time'])
        self.measurements['memory_usage'].append(end_memory)
        self.measurements['cpu_usage'].append(metrics['cpu_percent'])
        
        if end_gpu_mem is not None:
            self.measurements['gpu_usage'].append(end_gpu_mem)
        
        # Update operation statistics
        if operation_name not in self.operation_stats:
            self.operation_stats[operation_name] = {
                'count': 0,
                'total_time': 0.0,
                'min_time': float('inf'),
                'max_time': 0.0,
                'failures': 0
            }
        
        stats = self.operation_stats[operation_name]
        stats['count'] += 1
        
        if success:
            stats['total_time'] += metrics['execution_time']
            stats['min_time'] = min(stats['min_time'], metrics['execution_time'])
            stats['max_time'] = max(stats['max_time'], metrics['execution_time'])
        else:
            stats['failures'] += 1
        
        return result, metrics
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """
        Get statistics for an operation
        
        Args:
            operation_name: Name of operation
            
        Returns:
            Operation statistics
        """
        if operation_name not in self.operation_stats:
            return {}
        
        stats = self.operation_stats[operation_name]
        
        return {
            'operation': operation_name,
            'total_calls': stats['count'],
            'successful_calls': stats['count'] - stats['failures'],
            'failed_calls': stats['failures'],
            'success_rate': (stats['count'] - stats['failures']) / stats['count'] * 100
                if stats['count'] > 0 else 0,
            'total_time_seconds': stats['total_time'],
            'average_time_seconds': stats['total_time'] / (stats['count'] - stats['failures'])
                if (stats['count'] - stats['failures']) > 0 else 0,
            'min_time_seconds': stats['min_time'] if stats['min_time'] != float('inf') else 0,
            'max_time_seconds': stats['max_time']
        }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Get current system metrics
        
        Returns:
            System metrics dictionary
        """
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu': {
                'percent': psutil.cpu_percent(interval=0.1),
                'count': psutil.cpu_count(),
                'per_cpu': psutil.cpu_percent(interval=0.1, percpu=True)
            },
            'memory': {
                'total_mb': psutil.virtual_memory().total / 1024**2,
                'available_mb': psutil.virtual_memory().available / 1024**2,
                'percent': psutil.virtual_memory().percent,
                'used_mb': psutil.virtual_memory().used / 1024**2
            },
            'disk': {
                'total_gb': psutil.disk_usage('/').total / 1024**3,
                'used_gb': psutil.disk_usage('/').used / 1024**3,
                'free_gb': psutil.disk_usage('/').free / 1024**3,
                'percent': psutil.disk_usage('/').percent
            }
        }
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            metrics['gpu'] = {
                'available': True,
                'device_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'device_name': torch.cuda.get_device_name(),
                'memory_allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                'memory_reserved_mb': torch.cuda.memory_reserved() / 1024**2,
                'memory_total_mb': torch.cuda.get_device_properties(0).total_memory / 1024**2
            }
        else:
            metrics['gpu'] = {'available': False}
        
        return metrics
    
    def check_resource_limits(
        self,
        max_cpu_percent: float = 90.0,
        max_memory_percent: float = 85.0,
        max_disk_percent: float = 90.0
    ) -> Dict[str, Any]:
        """
        Check if resource usage is within limits
        
        Args:
            max_cpu_percent: Maximum CPU usage percentage
            max_memory_percent: Maximum memory usage percentage
            max_disk_percent: Maximum disk usage percentage
            
        Returns:
            Dictionary with limit check results
        """
        metrics = self.get_system_metrics()
        
        checks = {
            'timestamp': metrics['timestamp'],
            'cpu_ok': metrics['cpu']['percent'] < max_cpu_percent,
            'memory_ok': metrics['memory']['percent'] < max_memory_percent,
            'disk_ok': metrics['disk']['percent'] < max_disk_percent,
            'all_ok': True
        }
        
        checks['all_ok'] = checks['cpu_ok'] and checks['memory_ok'] and checks['disk_ok']
        
        if not checks['cpu_ok']:
            logger.warning(
                f"CPU usage {metrics['cpu']['percent']:.1f}% "
                f"exceeds limit {max_cpu_percent}%"
            )
        
        if not checks['memory_ok']:
            logger.warning(
                f"Memory usage {metrics['memory']['percent']:.1f}% "
                f"exceeds limit {max_memory_percent}%"
            )
        
        if not checks['disk_ok']:
            logger.warning(
                f"Disk usage {metrics['disk']['percent']:.1f}% "
                f"exceeds limit {max_disk_percent}%"
            )
        
        return checks
    
    def estimate_capacity(self, sample_size: int, total_size: int) -> Dict[str, Any]:
        """
        Estimate system capacity for workload
        
        Args:
            sample_size: Size of sample processed
            total_size: Total size to process
            
        Returns:
            Capacity estimates
        """
        if not self.measurements['execution_times']:
            return {'error': 'No measurements available'}
        
        avg_time = sum(self.measurements['execution_times']) / len(self.measurements['execution_times'])
        estimated_total_time = (total_size / sample_size) * avg_time
        
        avg_memory = sum(self.measurements['memory_usage']) / len(self.measurements['memory_usage'])
        available_memory = psutil.virtual_memory().available / 1024**2
        
        return {
            'sample_size': sample_size,
            'total_size': total_size,
            'estimated_time_seconds': estimated_total_time,
            'estimated_time_minutes': estimated_total_time / 60,
            'average_throughput': sample_size / avg_time if avg_time > 0 else 0,
            'memory_sufficient': avg_memory < available_memory * 0.8,
            'estimated_memory_mb': avg_memory,
            'available_memory_mb': available_memory
        }
    
    def _get_memory_usage(self) -> float:
        """Get current process memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def export_metrics(self, output_path: Path):
        """
        Export metrics to file
        
        Args:
            output_path: Path for metrics export
        """
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'operation_statistics': {
                name: self.get_operation_stats(name)
                for name in self.operation_stats.keys()
            },
            'system_metrics': self.get_system_metrics(),
            'measurement_history_size': {
                'execution_times': len(self.measurements['execution_times']),
                'memory_usage': len(self.measurements['memory_usage']),
                'cpu_usage': len(self.measurements['cpu_usage'])
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Metrics exported to {output_path}")


class ReliabilityMonitor:
    """
    Reliability monitoring for ISO/IEC 25010 compliance
    
    Tracks:
    - Maturity: Error rates, stability
    - Availability: Uptime, downtime
    - Fault tolerance: Error recovery
    - Recoverability: Recovery time
    """
    
    def __init__(self, log_dir: Path):
        """
        Initialize reliability monitor
        
        Args:
            log_dir: Directory for reliability logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.start_time = datetime.now()
        self.error_log = []
        self.downtime_periods = []
        self.recovery_times = []
        self.health_checks = deque(maxlen=100)
        
        self.is_available = True
        self.current_downtime_start = None
        
    def log_error(
        self,
        error_type: str,
        severity: str,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Log error occurrence
        
        Args:
            error_type: Type of error
            severity: Severity level (LOW, MEDIUM, HIGH, CRITICAL)
            message: Error message
            context: Additional context
        """
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': error_type,
            'severity': severity,
            'message': message,
            'context': context or {}
        }
        
        self.error_log.append(error_entry)
        
        # Log to file
        error_file = self.log_dir / 'errors.jsonl'
        with open(error_file, 'a') as f:
            f.write(json.dumps(error_entry) + '\n')
        
        logger.error(f"[{severity}] {error_type}: {message}")
        
        # Mark as unavailable if critical
        if severity == 'CRITICAL':
            self.mark_unavailable()
    
    def mark_unavailable(self):
        """Mark system as unavailable"""
        if self.is_available:
            self.is_available = False
            self.current_downtime_start = datetime.now()
            logger.warning("System marked as UNAVAILABLE")
    
    def mark_available(self):
        """Mark system as available (recovered)"""
        if not self.is_available and self.current_downtime_start:
            self.is_available = True
            recovery_time = (datetime.now() - self.current_downtime_start).total_seconds()
            
            self.downtime_periods.append({
                'start': self.current_downtime_start.isoformat(),
                'end': datetime.now().isoformat(),
                'duration_seconds': recovery_time
            })
            
            self.recovery_times.append(recovery_time)
            self.current_downtime_start = None
            
            logger.info(f"System RECOVERED in {recovery_time:.2f} seconds")
    
    def perform_health_check(self, checks: Dict[str, bool]) -> bool:
        """
        Perform health check
        
        Args:
            checks: Dictionary of check name to result
            
        Returns:
            True if all checks passed
        """
        result = {
            'timestamp': datetime.now().isoformat(),
            'checks': checks,
            'passed': all(checks.values())
        }
        
        self.health_checks.append(result)
        
        if not result['passed']:
            failed_checks = [name for name, passed in checks.items() if not passed]
            logger.warning(f"Health check failed: {failed_checks}")
            self.mark_unavailable()
        elif not self.is_available:
            self.mark_available()
        
        return result['passed']
    
    def get_uptime_stats(self) -> Dict[str, Any]:
        """
        Get uptime statistics
        
        Returns:
            Uptime statistics
        """
        total_uptime = (datetime.now() - self.start_time).total_seconds()
        total_downtime = sum(p['duration_seconds'] for p in self.downtime_periods)
        
        if self.current_downtime_start:
            total_downtime += (datetime.now() - self.current_downtime_start).total_seconds()
        
        uptime_percentage = (1 - total_downtime / total_uptime) * 100 if total_uptime > 0 else 100
        
        return {
            'start_time': self.start_time.isoformat(),
            'current_time': datetime.now().isoformat(),
            'total_uptime_seconds': total_uptime,
            'total_uptime_hours': total_uptime / 3600,
            'total_downtime_seconds': total_downtime,
            'total_downtime_minutes': total_downtime / 60,
            'uptime_percentage': uptime_percentage,
            'downtime_count': len(self.downtime_periods),
            'currently_available': self.is_available
        }
    
    def get_error_stats(self) -> Dict[str, Any]:
        """
        Get error statistics
        
        Returns:
            Error statistics
        """
        if not self.error_log:
            return {
                'total_errors': 0,
                'by_severity': {},
                'by_type': {},
                'error_rate': 0.0
            }
        
        by_severity = {}
        by_type = {}
        
        for error in self.error_log:
            severity = error['severity']
            error_type = error['type']
            
            by_severity[severity] = by_severity.get(severity, 0) + 1
            by_type[error_type] = by_type.get(error_type, 0) + 1
        
        uptime = (datetime.now() - self.start_time).total_seconds()
        error_rate = len(self.error_log) / uptime if uptime > 0 else 0
        
        return {
            'total_errors': len(self.error_log),
            'by_severity': by_severity,
            'by_type': by_type,
            'error_rate_per_hour': error_rate * 3600,
            'recent_errors': self.error_log[-10:] if self.error_log else []
        }
    
    def get_reliability_score(self) -> float:
        """
        Calculate overall reliability score (0-100)
        
        Returns:
            Reliability score
        """
        uptime_stats = self.get_uptime_stats()
        error_stats = self.get_error_stats()
        
        # Components of reliability score
        uptime_score = uptime_stats['uptime_percentage']
        
        # Error score (fewer errors = higher score)
        error_rate = error_stats['error_rate_per_hour']
        error_score = max(0, 100 - error_rate * 10)
        
        # Recovery score
        if self.recovery_times:
            avg_recovery = sum(self.recovery_times) / len(self.recovery_times)
            recovery_score = max(0, 100 - avg_recovery)  # Target <100s recovery
        else:
            recovery_score = 100
        
        # Weighted average
        reliability_score = (
            uptime_score * 0.5 +
            error_score * 0.3 +
            recovery_score * 0.2
        )
        
        return reliability_score
    
    def generate_reliability_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive reliability report
        
        Returns:
            Reliability report
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'reliability_score': self.get_reliability_score(),
            'uptime_stats': self.get_uptime_stats(),
            'error_stats': self.get_error_stats(),
            'recovery_stats': {
                'recovery_count': len(self.recovery_times),
                'average_recovery_time_seconds': 
                    sum(self.recovery_times) / len(self.recovery_times)
                    if self.recovery_times else 0,
                'min_recovery_time_seconds': 
                    min(self.recovery_times) if self.recovery_times else 0,
                'max_recovery_time_seconds': 
                    max(self.recovery_times) if self.recovery_times else 0
            },
            'health_check_stats': {
                'total_checks': len(self.health_checks),
                'passed_checks': sum(1 for check in self.health_checks if check['passed']),
                'failed_checks': sum(1 for check in self.health_checks if not check['passed']),
                'success_rate': 
                    sum(1 for check in self.health_checks if check['passed']) / len(self.health_checks) * 100
                    if self.health_checks else 0
            }
        }
    
    def export_report(self, output_path: Path):
        """
        Export reliability report
        
        Args:
            output_path: Path for report output
        """
        report = self.generate_reliability_report()
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Reliability report exported to {output_path}")

