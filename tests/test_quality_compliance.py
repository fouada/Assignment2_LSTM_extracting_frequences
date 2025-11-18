"""
ISO/IEC 25010 Compliance Test Suite
Tests all quality characteristics
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import yaml
import json

from src.quality.validator import InputValidator, ConfigValidator, ValidationError
from src.quality.security import SecurityManager
from src.quality.metrics_collector import QualityMetricsCollector, PerformanceTimer
from src.quality.monitoring import PerformanceMonitor, ReliabilityMonitor


class TestFunctionalSuitability:
    """Test functional suitability compliance"""
    
    def test_frequency_validation_completeness(self):
        """Test all frequency validation scenarios"""
        # Valid frequencies
        assert InputValidator.validate_frequency(1.0)[0]
        assert InputValidator.validate_frequency(10.5)[0]
        
        # Invalid frequencies
        assert not InputValidator.validate_frequency(-1.0)[0]
        assert not InputValidator.validate_frequency(0)[0]
        assert not InputValidator.validate_frequency("invalid")[0]
    
    def test_frequencies_list_validation(self):
        """Test frequency list validation"""
        # Valid list
        assert InputValidator.validate_frequencies([1.0, 3.0, 5.0])[0]
        
        # Duplicates
        assert not InputValidator.validate_frequencies([1.0, 1.0, 3.0])[0]
        
        # Empty list
        assert not InputValidator.validate_frequencies([])[0]
    
    def test_nyquist_criterion(self):
        """Test Nyquist criterion validation"""
        # Valid: sampling rate > 2 * max frequency
        assert InputValidator.validate_nyquist([1.0, 3.0, 5.0], 1000)[0]
        
        # Invalid: sampling rate too low
        assert not InputValidator.validate_nyquist([1.0, 600.0], 1000)[0]


class TestPerformanceEfficiency:
    """Test performance efficiency compliance"""
    
    def test_performance_measurement(self):
        """Test performance monitoring"""
        monitor = PerformanceMonitor()
        
        def sample_operation():
            return sum(range(1000))
        
        result, metrics = monitor.measure_operation(
            'sample_test',
            sample_operation
        )
        
        assert metrics['success']
        assert metrics['execution_time'] > 0
        assert 'memory_delta_mb' in metrics
    
    def test_resource_monitoring(self):
        """Test resource utilization monitoring"""
        monitor = PerformanceMonitor()
        metrics = monitor.get_system_metrics()
        
        assert 'cpu' in metrics
        assert 'memory' in metrics
        assert 'disk' in metrics
        assert metrics['cpu']['percent'] >= 0
        assert metrics['memory']['percent'] >= 0
    
    def test_capacity_estimation(self):
        """Test capacity estimation"""
        monitor = PerformanceMonitor()
        
        # Simulate some operations
        for _ in range(10):
            monitor.measurements['execution_times'].append(0.1)
            monitor.measurements['memory_usage'].append(100.0)
        
        estimate = monitor.estimate_capacity(100, 1000)
        
        assert 'estimated_time_seconds' in estimate
        assert 'average_throughput' in estimate
        assert estimate['estimated_time_seconds'] > 0


class TestCompatibility:
    """Test compatibility compliance"""
    
    def test_config_file_formats(self):
        """Test configuration file format support"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'test_config.yaml'
            
            config = {
                'data': {
                    'frequencies': [1.0, 3.0, 5.0],
                    'sampling_rate': 1000,
                    'duration': 10.0
                },
                'model': {
                    'input_size': 5,
                    'hidden_size': 128,
                    'num_layers': 2,
                    'output_size': 1
                },
                'training': {
                    'batch_size': 32,
                    'epochs': 10,
                    'learning_rate': 0.001
                }
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            is_valid, error, loaded_config = ConfigValidator.validate_config_file(config_path)
            
            assert is_valid, f"Config validation failed: {error}"
            assert loaded_config is not None


class TestUsability:
    """Test usability compliance"""
    
    def test_error_messages_quality(self):
        """Test error message clarity"""
        # Test that error messages are informative
        is_valid, error = InputValidator.validate_frequency(-1.0)
        assert not is_valid
        assert 'positive' in error.lower()
        assert '-1.0' in error
    
    def test_input_validation_protection(self):
        """Test user error protection"""
        # String sanitization
        is_valid, error, sanitized = InputValidator.sanitize_string(
            "valid_name_123",
            max_length=100
        )
        assert is_valid
        assert sanitized == "valid_name_123"
        
        # Dangerous input
        is_valid, error, _ = InputValidator.sanitize_string(
            "<script>alert('xss')</script>",
            max_length=100
        )
        assert not is_valid
        assert 'dangerous' in error.lower() or 'invalid' in error.lower()
    
    def test_validation_feedback(self):
        """Test validation provides helpful feedback"""
        is_valid, error = InputValidator.validate_batch_size(0)
        assert not is_valid
        assert 'minimum' in error.lower()


class TestReliability:
    """Test reliability compliance"""
    
    def test_error_logging(self):
        """Test error logging functionality"""
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = ReliabilityMonitor(Path(tmpdir))
            
            monitor.log_error(
                'TestError',
                'HIGH',
                'Test error message',
                {'context': 'test'}
            )
            
            stats = monitor.get_error_stats()
            assert stats['total_errors'] == 1
            assert 'HIGH' in stats['by_severity']
    
    def test_availability_tracking(self):
        """Test availability monitoring"""
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = ReliabilityMonitor(Path(tmpdir))
            
            # Simulate downtime
            monitor.mark_unavailable()
            assert not monitor.is_available
            
            # Simulate recovery
            monitor.mark_available()
            assert monitor.is_available
            
            stats = monitor.get_uptime_stats()
            assert stats['downtime_count'] == 1
    
    def test_health_checks(self):
        """Test health check functionality"""
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = ReliabilityMonitor(Path(tmpdir))
            
            # Passing health check
            result = monitor.perform_health_check({
                'cpu': True,
                'memory': True,
                'disk': True
            })
            assert result
            
            # Failing health check
            result = monitor.perform_health_check({
                'cpu': True,
                'memory': False,
                'disk': True
            })
            assert not result
    
    def test_recovery_time_tracking(self):
        """Test recovery time measurement"""
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = ReliabilityMonitor(Path(tmpdir))
            
            monitor.mark_unavailable()
            import time
            time.sleep(0.1)
            monitor.mark_available()
            
            assert len(monitor.recovery_times) == 1
            assert monitor.recovery_times[0] >= 0.1


class TestSecurity:
    """Test security compliance"""
    
    def test_encryption_decryption(self):
        """Test data encryption/decryption"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SecurityManager(Path(tmpdir) / 'audit.log')
            
            original_data = "sensitive information"
            
            # Encrypt
            encrypted = manager.encrypt_data(original_data)
            assert encrypted != original_data.encode()
            
            # Decrypt
            decrypted = manager.decrypt_data(encrypted)
            assert decrypted == original_data
    
    def test_checksum_verification(self):
        """Test data integrity checking"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SecurityManager(Path(tmpdir) / 'audit.log')
            
            data = b"test data"
            checksum = manager.compute_checksum(data)
            
            # Verify correct data
            assert manager.verify_checksum(data, checksum)
            
            # Verify tampered data
            tampered_data = b"test data modified"
            assert not manager.verify_checksum(tampered_data, checksum)
    
    def test_audit_logging(self):
        """Test audit trail functionality"""
        with tempfile.TemporaryDirectory() as tmpdir:
            audit_path = Path(tmpdir) / 'audit.log'
            manager = SecurityManager(audit_path)
            
            manager.log_audit_event(
                'TEST',
                'test_action',
                {'detail': 'test'},
                user='test_user'
            )
            
            logs = manager.get_audit_log()
            assert len(logs) > 0
            assert logs[-1]['action'] == 'test_action'
    
    def test_api_key_generation(self):
        """Test API key generation and validation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SecurityManager(Path(tmpdir) / 'audit.log')
            
            api_key = manager.generate_api_key('user123')
            assert 'user123' in api_key
            
            is_valid, user_id = manager.validate_api_key(api_key)
            assert is_valid
            assert user_id == 'user123'
    
    def test_sensitive_data_sanitization(self):
        """Test sensitive data sanitization in logs"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SecurityManager(Path(tmpdir) / 'audit.log')
            
            data = {
                'username': 'user',
                'password': 'secret123',
                'api_key': 'key123',
                'public_data': 'visible'
            }
            
            sanitized = manager.sanitize_log_data(data)
            
            assert sanitized['username'] == 'user'
            assert sanitized['password'] == '***REDACTED***'
            assert sanitized['api_key'] == '***REDACTED***'
            assert sanitized['public_data'] == 'visible'


class TestMaintainability:
    """Test maintainability compliance"""
    
    def test_input_validation_modularity(self):
        """Test that validation functions are modular"""
        # Each validator should work independently
        assert InputValidator.validate_frequency(1.0)[0]
        assert InputValidator.validate_duration(10.0)[0]
        assert InputValidator.validate_batch_size(32)[0]
        
        # They should not interfere with each other
        InputValidator.validate_frequency(-1.0)
        assert InputValidator.validate_duration(10.0)[0]
    
    def test_config_validator_reusability(self):
        """Test validator reusability"""
        # Validator should be reusable across multiple validations
        validator = ConfigValidator()
        
        # Multiple validation calls should work
        result1 = InputValidator.validate_frequency(1.0)
        result2 = InputValidator.validate_frequency(2.0)
        
        assert result1[0] and result2[0]


class TestPortability:
    """Test portability compliance"""
    
    def test_path_handling(self):
        """Test cross-platform path handling"""
        # Test various path formats
        paths = [
            "/absolute/path",
            "relative/path",
            "./current/path",
            "../parent/path"
        ]
        
        for path_str in paths:
            # Should not raise exception
            path = Path(path_str)
            assert isinstance(path, Path)
    
    def test_config_file_portability(self):
        """Test configuration portability"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'config.yaml'
            
            config = {
                'data': {'frequencies': [1.0, 3.0]},
                'model': {
                    'input_size': 5,
                    'hidden_size': 128,
                    'num_layers': 2,
                    'output_size': 1
                },
                'training': {'batch_size': 32, 'epochs': 10, 'learning_rate': 0.001}
            }
            
            # Save and load should work on any platform
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            is_valid, _, loaded = ConfigValidator.validate_config_file(config_path)
            assert is_valid
            assert loaded['data']['frequencies'] == [1.0, 3.0]


class TestQualityMetrics:
    """Test quality metrics collection"""
    
    def test_metrics_collector_initialization(self):
        """Test metrics collector setup"""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = QualityMetricsCollector(Path(tmpdir))
            assert collector.output_dir.exists()
    
    def test_performance_metrics_collection(self):
        """Test performance metrics recording"""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = QualityMetricsCollector(Path(tmpdir))
            
            metrics = collector.collect_performance_metrics(
                'test_operation',
                execution_time=1.5,
                memory_used=100.0
            )
            
            assert metrics['operation'] == 'test_operation'
            assert metrics['time_behaviour']['execution_time'] == 1.5
    
    def test_reliability_metrics_collection(self):
        """Test reliability metrics recording"""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = QualityMetricsCollector(Path(tmpdir))
            
            collector.record_error('TestError', 'ERROR')
            metrics = collector.collect_reliability_metrics()
            
            assert metrics['maturity']['error_count'] == 1
    
    def test_compliance_report_generation(self):
        """Test compliance report generation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = QualityMetricsCollector(Path(tmpdir))
            
            # Collect some metrics
            collector.collect_performance_metrics('test', 1.0, 50.0)
            collector.collect_functional_metrics(18, 20, 95.0)
            
            # Generate report
            report_path = collector.generate_compliance_report()
            
            assert report_path.exists()
            
            # Verify report content
            with open(report_path, 'r') as f:
                report = json.load(f)
            
            assert 'report_metadata' in report
            assert report['report_metadata']['standard'] == 'ISO/IEC 25010:2011'
            assert 'quality_characteristics' in report
            assert 'overall_scores' in report
    
    def test_performance_timer_context_manager(self):
        """Test performance timer context manager"""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = QualityMetricsCollector(Path(tmpdir))
            
            with PerformanceTimer(collector, 'test_operation'):
                # Simulate some work
                sum(range(1000))
            
            assert 'test_operation' in collector.metrics['performance']


# Compliance validation functions

def test_overall_compliance_score():
    """Test overall compliance score calculation"""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = QualityMetricsCollector(Path(tmpdir))
        
        # Collect various metrics
        collector.collect_performance_metrics('test', 1.0)
        collector.collect_reliability_metrics()
        collector.collect_security_metrics(100, 5)
        collector.collect_usability_metrics(95, 5)
        collector.collect_maintainability_metrics(85.0, 8.0, 90.0)
        collector.collect_functional_metrics(18, 20, 95.0)
        
        scores = collector._calculate_overall_scores()
        
        assert 'overall' in scores
        assert 0 <= scores['overall'] <= 100


def test_compliance_validation():
    """Test compliance status assessment"""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = QualityMetricsCollector(Path(tmpdir))
        
        # Collect metrics
        collector.collect_functional_metrics(20, 20, 100.0)
        
        status = collector._assess_compliance()
        
        assert 'functional_suitability' in collector.metrics


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

