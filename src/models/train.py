"""
Training script for the network intrusion detection models.
Use this script to train models on your dataset.
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Ensure we're using the correct path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from data_Pipelines.data_pipeline import DataPipeline
from models.models import (
    RandomForestModel, GradientBoostingModel, LogisticRegressionModel,
    EnsembleModel, DeepLearningModel
)
from utils.helpers import Logger, ModelMetrics

# Setup logging
logger = Logger.setup_logger('training', 'logs/training.log')


def train_random_forest(X_train, X_test, y_train, y_test, args):
    """Train Random Forest model."""
    logger.info("Training Random Forest Model...")
    
    model = RandomForestModel(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state
    )
    
    model.train(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    
    if args.save:
        filepath = model.save_model(args.model_dir)
        logger.info(f"Model saved to {filepath}")
    
    return model, metrics


def train_gradient_boosting(X_train, X_test, y_train, y_test, args):
    """Train Gradient Boosting model."""
    logger.info("Training Gradient Boosting Model...")
    
    model = GradientBoostingModel(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        random_state=args.random_state
    )
    
    model.train(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    
    if args.save:
        filepath = model.save_model(args.model_dir)
        logger.info(f"Model saved to {filepath}")
    
    return model, metrics


def train_logistic_regression(X_train, X_test, y_train, y_test, args):
    """Train Logistic Regression model."""
    logger.info("Training Logistic Regression Model...")
    
    model = LogisticRegressionModel(random_state=args.random_state)
    
    model.train(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    
    if args.save:
        filepath = model.save_model(args.model_dir)
        logger.info(f"Model saved to {filepath}")
    
    return model, metrics


def train_neural_network(X_train, X_test, y_train, y_test, args):
    """Train Neural Network model."""
    logger.info("Training Neural Network Model...")
    
    model = DeepLearningModel(
        input_dim=X_train.shape[1],
        hidden_layers=[128, 64, 32]
    )
    
    model.train(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size)
    
    if args.save:
        filepath = model.model.save(os.path.join(args.model_dir, 'neural_network.h5'))
        logger.info(f"Model saved to {filepath}")
    
    return model, {}


def train_ensemble(X_train, X_test, y_train, y_test, args):
    """Train Ensemble model."""
    logger.info("Training Ensemble Model...")
    
    models = [
        RandomForestModel(n_estimators=50, max_depth=15),
        GradientBoostingModel(n_estimators=50, learning_rate=0.1),
        LogisticRegressionModel()
    ]
    
    ensemble = EnsembleModel(models)
    ensemble.train(X_train, y_train)
    
    metrics = ensemble.evaluate(X_test, y_test)
    
    return ensemble, metrics


def main():
    parser = argparse.ArgumentParser(
        description='Train network intrusion detection models'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='data/network_traffic.csv',
        help='Path to training data'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='random_forest',
        choices=['random_forest', 'gradient_boosting', 'logistic_regression', 
                'neural_network', 'ensemble'],
        help='Model type to train'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set size'
    )
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=100,
        help='Number of estimators for ensemble methods'
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=20,
        help='Maximum depth for tree-based models'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.1,
        help='Learning rate for Gradient Boosting'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of epochs for neural network'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for neural network'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random state for reproducibility'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save the trained model'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models/',
        help='Directory to save models'
    )
    
    args = parser.parse_args()
    
    # Create model directory
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    logger.info(f"Loading data from {args.data}")
    if not os.path.exists(args.data):
        logger.error(f"Data file not found: {args.data}")
        logger.info("Please provide training data in CSV format with target column 'attack'")
        return
    
    pipeline = DataPipeline()
    X_train, X_test, y_train, y_test = pipeline.prepare_data(
        args.data, 
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    logger.info(f"Data prepared: train {X_train.shape}, test {X_test.shape}")
    
    # Train selected model
    if args.model == 'random_forest':
        model, metrics = train_random_forest(X_train, X_test, y_train, y_test, args)
    elif args.model == 'gradient_boosting':
        model, metrics = train_gradient_boosting(X_train, X_test, y_train, y_test, args)
    elif args.model == 'logistic_regression':
        model, metrics = train_logistic_regression(X_train, X_test, y_train, y_test, args)
    elif args.model == 'neural_network':
        model, metrics = train_neural_network(X_train, X_test, y_train, y_test, args)
    elif args.model == 'ensemble':
        model, metrics = train_ensemble(X_train, X_test, y_train, y_test, args)
    
    # Print results
    logger.info(f"Training completed for {args.model}")
    if metrics:
        logger.info(f"Metrics: {metrics}")
        print(f"\n{args.model.upper()} Model Metrics:")
        print(f"  Accuracy:  {metrics.get('accuracy', 'N/A'):.4f}")
        print(f"  Precision: {metrics.get('precision', 'N/A'):.4f}")
        print(f"  Recall:    {metrics.get('recall', 'N/A'):.4f}")
        print(f"  F1-Score:  {metrics.get('f1', 'N/A'):.4f}")


if __name__ == '__main__':
    main()
