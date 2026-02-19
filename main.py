"""
Main entry point for the Network Intrusion Detection System.
Provides CLI interface for training, running API, and launching dashboard.
"""

import argparse
import sys
import os
import subprocess
from pathlib import Path

def setup_environment():
    """Setup necessary directories."""
    Path('logs').mkdir(exist_ok=True)
    Path('models').mkdir(exist_ok=True)
    Path('data').mkdir(exist_ok=True)
    print("✓ Environment setup completed")


def run_training(args):
    """Run model training."""
    print("🔧 Starting model training...")
    
    cmd = [
        sys.executable, '-m', 'src.models.train',
        '--model', args.model,
        '--data', args.data,
    ]
    
    if args.save:
        cmd.append('--save')
    
    if args.n_estimators:
        cmd.extend(['--n-estimators', str(args.n_estimators)])
    
    if args.max_depth:
        cmd.extend(['--max-depth', str(args.max_depth)])
    
    if args.epochs:
        cmd.extend(['--epochs', str(args.epochs)])
    
    subprocess.run(cmd)


def run_api(args):
    """Run the API server."""
    print("🚀 Starting API server...")
    print(f"   Listening on http://{args.host}:{args.port}")
    print("   Press Ctrl+C to stop")
    
    os.environ['API_HOST'] = args.host
    os.environ['API_PORT'] = str(args.port)
    os.environ['API_DEBUG'] = str(args.debug)
    
    subprocess.run([sys.executable, '-m', 'src.api.app'])


def run_dashboard(args):
    """Run the visualization dashboard."""
    print("📊 Starting dashboard...")
    print(f"   Access at http://{args.host}:{args.port}")
    print("   Press Ctrl+C to stop")
    
    subprocess.run([sys.executable, '-m', 'src.visualization.dashboard'])


def generate_data(args):
    """Generate sample data."""
    print("📊 Generating sample network traffic data...")
    
    from src.data_Pipelines.generate_sample_data import generate_sample_data
    generate_sample_data(n_samples=args.samples, output_file=args.output)


def run_tests(args):
    """Run test suite."""
    print("🧪 Running tests...")
    
    cmd = ['pytest', 'tests/', '-v']
    
    if args.coverage:
        cmd.extend(['--cov=src', '--cov-report=html'])
    
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(
        description='Network Intrusion Detection System CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Generate sample data
  python main.py generate-data

  # Train Random Forest model
  python main.py train --model random_forest --data data/network_traffic.csv --save

  # Run API server
  python main.py api --host 0.0.0.0 --port 5000

  # Launch dashboard
  python main.py dashboard --host 0.0.0.0 --port 8050

  # Run tests
  python main.py test --coverage
        '''
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Generate Data Command
    gen_parser = subparsers.add_parser('generate-data', help='Generate sample network traffic data')
    gen_parser.add_argument('--samples', type=int, default=10000, help='Number of samples to generate')
    gen_parser.add_argument('--output', type=str, default='data/network_traffic.csv', help='Output file path')
    
    # Training Command
    train_parser = subparsers.add_parser('train', help='Train a machine learning model')
    train_parser.add_argument('--model', type=str, default='random_forest',
                             choices=['random_forest', 'gradient_boosting', 'logistic_regression', 
                                     'neural_network', 'ensemble'],
                             help='Model to train')
    train_parser.add_argument('--data', type=str, default='data/network_traffic.csv', help='Training data path')
    train_parser.add_argument('--save', action='store_true', help='Save trained model')
    train_parser.add_argument('--n-estimators', type=int, help='Number of estimators')
    train_parser.add_argument('--max-depth', type=int, help='Max depth for trees')
    train_parser.add_argument('--epochs', type=int, help='Epochs for neural network')
    
    # API Command
    api_parser = subparsers.add_parser('api', help='Run the Flask API server')
    api_parser.add_argument('--host', type=str, default='0.0.0.0', help='API host')
    api_parser.add_argument('--port', type=int, default=5000, help='API port')
    api_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    # Dashboard Command
    dash_parser = subparsers.add_parser('dashboard', help='Launch the Dash dashboard')
    dash_parser.add_argument('--host', type=str, default='0.0.0.0', help='Dashboard host')
    dash_parser.add_argument('--port', type=int, default=8050, help='Dashboard port')
    
    # Test Command
    test_parser = subparsers.add_parser('test', help='Run test suite')
    test_parser.add_argument('--coverage', action='store_true', help='Generate coverage report')
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Execute command
    if args.command == 'generate-data':
        generate_data(args)
    elif args.command == 'train':
        run_training(args)
    elif args.command == 'api':
        run_api(args)
    elif args.command == 'dashboard':
        run_dashboard(args)
    elif args.command == 'test':
        run_tests(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
