#!/usr/bin/env python3
"""
ISO/IEC 25010 Compliance CLI Tool
Command-line interface for quality assurance operations
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.quality.metrics_collector import QualityMetricsCollector
from src.quality.validator import InputValidator, ConfigValidator, ValidationError
from src.quality.security import SecurityManager
from src.quality.monitoring import PerformanceMonitor, ReliabilityMonitor


def validate_config_command(args):
    """Validate configuration file"""
    try:
        print(f"Validating configuration: {args.config}")
        config = ConfigValidator.validate_and_load(args.config)
        
        print("✅ Configuration is valid!")
        print("\nConfiguration details:")
        print(f"  Frequencies: {config['data']['frequencies']}")
        print(f"  Sampling rate: {config['data']['sampling_rate']} Hz")
        print(f"  Duration: {config['data']['duration']}s")
        print(f"  Model hidden size: {config['model']['hidden_size']}")
        print(f"  Training epochs: {config['training']['epochs']}")
        
        return 0
    except ValidationError as e:
        print(f"❌ Configuration validation failed:")
        print(f"   {e}")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def generate_report_command(args):
    """Generate compliance report"""
    try:
        print(f"Generating ISO/IEC 25010 compliance report...")
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        collector = QualityMetricsCollector(output_dir)
        
        # Collect sample metrics
        print("  Collecting quality metrics...")
        collector.collect_functional_metrics(
            implemented_features=20,
            required_features=20,
            test_pass_rate=95.0
        )
        collector.collect_maintainability_metrics(
            test_coverage=85.0,
            code_complexity=8.0,
            documentation_coverage=90.0
        )
        collector.collect_reliability_metrics()
        
        # Generate report
        report_path = collector.generate_compliance_report(format=args.format)
        
        print(f"\n✅ Compliance report generated!")
        print(f"   Report: {report_path}")
        print(f"   Summary: {output_dir / 'compliance_summary.txt'}")
        
        # Print scores
        scores = collector._calculate_overall_scores()
        if scores:
            print(f"\n📊 Quality Scores:")
            for characteristic, score in sorted(scores.items()):
                status = "✅" if score >= 75 else "⚠️"
                print(f"   {status} {characteristic:25s}: {score:6.2f}%")
        
        return 0
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        import traceback
        traceback.print_exc()
        return 1


def check_security_command(args):
    """Perform security checks"""
    try:
        print("Performing security checks...")
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        security = SecurityManager(output_dir / 'security_audit.log')
        
        # Check files if provided
        if args.files:
            print("\nComputing file checksums:")
            for file_path in args.files:
                path = Path(file_path)
                if not path.exists():
                    print(f"  ⚠️  File not found: {file_path}")
                    continue
                
                checksum = security.compute_file_checksum(path)
                print(f"  ✅ {path.name}")
                print(f"     SHA-256: {checksum}")
        
        # Test encryption if requested
        if args.test_encryption:
            print("\nTesting encryption:")
            test_data = "test_sensitive_data"
            encrypted = security.encrypt_data(test_data)
            decrypted = security.decrypt_data(encrypted)
            
            if decrypted == test_data:
                print("  ✅ Encryption/Decryption: WORKING")
            else:
                print("  ❌ Encryption/Decryption: FAILED")
        
        # Generate audit report
        if args.generate_audit:
            audit_path = output_dir / 'security_audit_report.json'
            security.generate_audit_report(audit_path)
            print(f"\n✅ Security audit report: {audit_path}")
        
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def monitor_system_command(args):
    """Monitor system resources"""
    try:
        print("Monitoring system resources...")
        
        monitor = PerformanceMonitor()
        metrics = monitor.get_system_metrics()
        
        print(f"\n📊 System Metrics:")
        print(f"\nCPU:")
        print(f"  Cores: {metrics['cpu']['count']}")
        print(f"  Usage: {metrics['cpu']['percent']:.1f}%")
        
        print(f"\nMemory:")
        print(f"  Total: {metrics['memory']['total_mb']:.0f} MB")
        print(f"  Used: {metrics['memory']['used_mb']:.0f} MB")
        print(f"  Available: {metrics['memory']['available_mb']:.0f} MB")
        print(f"  Usage: {metrics['memory']['percent']:.1f}%")
        
        print(f"\nDisk:")
        print(f"  Total: {metrics['disk']['total_gb']:.1f} GB")
        print(f"  Used: {metrics['disk']['used_gb']:.1f} GB")
        print(f"  Free: {metrics['disk']['free_gb']:.1f} GB")
        print(f"  Usage: {metrics['disk']['percent']:.1f}%")
        
        if metrics['gpu']['available']:
            print(f"\nGPU:")
            print(f"  Device: {metrics['gpu']['device_name']}")
            print(f"  Total Memory: {metrics['gpu']['memory_total_mb']:.0f} MB")
            print(f"  Allocated: {metrics['gpu']['memory_allocated_mb']:.2f} MB")
        else:
            print(f"\nGPU: Not available")
        
        # Check limits
        print(f"\n🔍 Resource Limit Checks:")
        limits = monitor.check_resource_limits(
            max_cpu_percent=args.max_cpu,
            max_memory_percent=args.max_memory,
            max_disk_percent=args.max_disk
        )
        
        print(f"  CPU {'✅' if limits['cpu_ok'] else '❌'}: {metrics['cpu']['percent']:.1f}% (limit: {args.max_cpu}%)")
        print(f"  Memory {'✅' if limits['memory_ok'] else '❌'}: {metrics['memory']['percent']:.1f}% (limit: {args.max_memory}%)")
        print(f"  Disk {'✅' if limits['disk_ok'] else '❌'}: {metrics['disk']['percent']:.1f}% (limit: {args.max_disk}%)")
        
        if limits['all_ok']:
            print(f"\n✅ All resource limits OK")
            return 0
        else:
            print(f"\n⚠️  Some resource limits exceeded")
            return 1
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def run_tests_command(args):
    """Run compliance tests"""
    try:
        import pytest
        
        print("Running ISO/IEC 25010 compliance tests...")
        
        test_args = ['tests/test_quality_compliance.py', '-v']
        
        if args.test_class:
            test_args[0] += f'::{args.test_class}'
        
        if args.coverage:
            test_args.extend(['--cov=src.quality', '--cov-report=html'])
        
        exit_code = pytest.main(test_args)
        
        if exit_code == 0:
            print("\n✅ All compliance tests passed!")
        else:
            print("\n❌ Some tests failed")
        
        return exit_code
        
    except ImportError:
        print("❌ pytest not installed. Install with: pip install pytest pytest-cov")
        return 1
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1


def validate_input_command(args):
    """Validate input parameters"""
    try:
        print("Validating inputs...")
        
        all_valid = True
        
        # Validate frequency if provided
        if args.frequency is not None:
            is_valid, error = InputValidator.validate_frequency(args.frequency)
            if is_valid:
                print(f"  ✅ Frequency {args.frequency} Hz: Valid")
            else:
                print(f"  ❌ Frequency {args.frequency} Hz: {error}")
                all_valid = False
        
        # Validate frequencies list if provided
        if args.frequencies:
            frequencies = [float(f) for f in args.frequencies.split(',')]
            is_valid, error = InputValidator.validate_frequencies(frequencies)
            if is_valid:
                print(f"  ✅ Frequencies {frequencies}: Valid")
            else:
                print(f"  ❌ Frequencies {frequencies}: {error}")
                all_valid = False
            
            # Check Nyquist if sampling rate provided
            if args.sampling_rate:
                is_valid, error = InputValidator.validate_nyquist(frequencies, args.sampling_rate)
                if is_valid:
                    print(f"  ✅ Nyquist criterion: Satisfied")
                else:
                    print(f"  ❌ Nyquist criterion: {error}")
                    all_valid = False
        
        # Validate sampling rate if provided
        if args.sampling_rate is not None:
            is_valid, error = InputValidator.validate_sampling_rate(args.sampling_rate)
            if is_valid:
                print(f"  ✅ Sampling rate {args.sampling_rate} Hz: Valid")
            else:
                print(f"  ❌ Sampling rate {args.sampling_rate} Hz: {error}")
                all_valid = False
        
        # Validate batch size if provided
        if args.batch_size is not None:
            is_valid, error = InputValidator.validate_batch_size(args.batch_size)
            if is_valid:
                print(f"  ✅ Batch size {args.batch_size}: Valid")
            else:
                print(f"  ❌ Batch size {args.batch_size}: {error}")
                all_valid = False
        
        if all_valid:
            print("\n✅ All inputs valid!")
            return 0
        else:
            print("\n❌ Some inputs invalid")
            return 1
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='ISO/IEC 25010 Compliance CLI Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate configuration
  python compliance_cli.py validate-config config/config.yaml
  
  # Generate compliance report
  python compliance_cli.py generate-report -o compliance_reports
  
  # Check security
  python compliance_cli.py check-security --files main.py --test-encryption
  
  # Monitor system
  python compliance_cli.py monitor-system
  
  # Run tests
  python compliance_cli.py run-tests --coverage
  
  # Validate inputs
  python compliance_cli.py validate-input --frequencies "1.0,3.0,5.0,7.0" --sampling-rate 1000
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Validate config command
    validate_parser = subparsers.add_parser('validate-config', help='Validate configuration file')
    validate_parser.add_argument('config', type=str, help='Path to configuration file')
    validate_parser.set_defaults(func=validate_config_command)
    
    # Generate report command
    report_parser = subparsers.add_parser('generate-report', help='Generate compliance report')
    report_parser.add_argument('-o', '--output-dir', default='compliance_reports', help='Output directory')
    report_parser.add_argument('-f', '--format', choices=['json', 'yaml'], default='json', help='Report format')
    report_parser.set_defaults(func=generate_report_command)
    
    # Security check command
    security_parser = subparsers.add_parser('check-security', help='Perform security checks')
    security_parser.add_argument('--files', nargs='+', help='Files to check')
    security_parser.add_argument('--test-encryption', action='store_true', help='Test encryption')
    security_parser.add_argument('--generate-audit', action='store_true', help='Generate audit report')
    security_parser.add_argument('-o', '--output-dir', default='compliance_reports', help='Output directory')
    security_parser.set_defaults(func=check_security_command)
    
    # Monitor system command
    monitor_parser = subparsers.add_parser('monitor-system', help='Monitor system resources')
    monitor_parser.add_argument('--max-cpu', type=float, default=90.0, help='Maximum CPU percentage')
    monitor_parser.add_argument('--max-memory', type=float, default=85.0, help='Maximum memory percentage')
    monitor_parser.add_argument('--max-disk', type=float, default=90.0, help='Maximum disk percentage')
    monitor_parser.set_defaults(func=monitor_system_command)
    
    # Run tests command
    test_parser = subparsers.add_parser('run-tests', help='Run compliance tests')
    test_parser.add_argument('--test-class', help='Specific test class to run')
    test_parser.add_argument('--coverage', action='store_true', help='Generate coverage report')
    test_parser.set_defaults(func=run_tests_command)
    
    # Validate input command
    input_parser = subparsers.add_parser('validate-input', help='Validate input parameters')
    input_parser.add_argument('--frequency', type=float, help='Single frequency to validate')
    input_parser.add_argument('--frequencies', type=str, help='Comma-separated frequencies')
    input_parser.add_argument('--sampling-rate', type=int, help='Sampling rate in Hz')
    input_parser.add_argument('--batch-size', type=int, help='Batch size')
    input_parser.set_defaults(func=validate_input_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())

