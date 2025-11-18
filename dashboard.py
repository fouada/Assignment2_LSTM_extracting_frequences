#!/usr/bin/env python3
"""
Dashboard Launcher for LSTM Frequency Extraction
Professional interactive visualization interface

Author: Professional ML Engineering Team
Date: 2025

Usage:
    python dashboard.py                              # Launch with latest experiment
    python dashboard.py --experiment experiments/... # Launch with specific experiment
    python dashboard.py --port 8080                  # Run on custom port
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import Optional

from src.visualization.interactive_dashboard import create_dashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_latest_experiment(experiments_dir: Path = Path('experiments')) -> Optional[Path]:
    """
    Find the most recent experiment directory.
    
    Args:
        experiments_dir: Base experiments directory
        
    Returns:
        Path to latest experiment or None
    """
    if not experiments_dir.exists():
        logger.warning(f"Experiments directory not found: {experiments_dir}")
        return None
    
    # Get all experiment directories
    exp_dirs = [d for d in experiments_dir.iterdir() 
                if d.is_dir() and d.name.startswith('lstm_frequency_extraction_')]
    
    if not exp_dirs:
        logger.warning("No experiment directories found")
        return None
    
    # Sort by modification time and return the latest
    latest = max(exp_dirs, key=lambda p: p.stat().st_mtime)
    logger.info(f"Found latest experiment: {latest.name}")
    return latest


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Launch LSTM Frequency Extraction Interactive Dashboard',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch with latest experiment
  python dashboard.py
  
  # Launch with specific experiment
  python dashboard.py --experiment experiments/lstm_frequency_extraction_20251118_002838
  
  # Run on custom port
  python dashboard.py --port 8080
  
  # Enable debug mode
  python dashboard.py --debug
        """
    )
    
    parser.add_argument(
        '--experiment', '-e',
        type=str,
        help='Path to specific experiment directory'
    )
    
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=8050,
        help='Port to run dashboard on (default: 8050)'
    )
    
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug mode'
    )
    
    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='Do not automatically open browser'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    logger.info("="*80)
    logger.info("LSTM Frequency Extraction - Interactive Dashboard")
    logger.info("="*80)
    
    # Determine experiment directory
    if args.experiment:
        exp_dir = Path(args.experiment)
        if not exp_dir.exists():
            logger.error(f"Experiment directory not found: {exp_dir}")
            sys.exit(1)
        logger.info(f"Using specified experiment: {exp_dir}")
    else:
        exp_dir = find_latest_experiment()
        if exp_dir is None:
            logger.warning("No experiments found. Dashboard will launch without data.")
            logger.info("Run 'python main.py' first to generate experiment data.")
            exp_dir = None
    
    # Create and configure dashboard
    logger.info(f"Initializing dashboard on port {args.port}...")
    dashboard = create_dashboard(experiment_dir=exp_dir, port=args.port)
    
    # Print access information
    logger.info("\n" + "="*80)
    logger.info("Dashboard is ready!")
    logger.info("="*80)
    logger.info(f"\n🌐 Access the dashboard at:")
    logger.info(f"   http://localhost:{args.port}")
    logger.info(f"   http://127.0.0.1:{args.port}")
    logger.info("\n📊 Features:")
    logger.info("   • Interactive frequency extraction visualization")
    logger.info("   • Real-time training progress monitoring")
    logger.info("   • Comprehensive error analysis")
    logger.info("   • Performance metrics comparison")
    logger.info("   • Model architecture summary")
    logger.info("\n⌨️  Press Ctrl+C to stop the server")
    logger.info("="*80 + "\n")
    
    # Open browser automatically unless disabled
    if not args.no_browser:
        import webbrowser
        import threading
        def open_browser():
            import time
            time.sleep(1.5)  # Wait for server to start
            webbrowser.open(f'http://localhost:{args.port}')
        threading.Thread(target=open_browser, daemon=True).start()
    
    # Run dashboard
    try:
        dashboard.run(debug=args.debug)
    except KeyboardInterrupt:
        logger.info("\n\nShutting down dashboard...")
        logger.info("Dashboard closed successfully!")
    except Exception as e:
        logger.error(f"Error running dashboard: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

