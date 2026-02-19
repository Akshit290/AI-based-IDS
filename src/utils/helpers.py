"""Utility functions for the intrusion detection system."""

import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and manage configuration."""
    
    @staticmethod
    def load_config(config_path='config.json'):
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return ConfigLoader.get_default_config()
    
    @staticmethod
    def get_default_config():
        """Return default configuration."""
        return {
            'model': {
                'type': 'random_forest',
                'random_state': 42,
                'n_estimators': 100,
                'max_depth': 20
            },
            'data': {
                'test_size': 0.2,
                'random_state': 42,
                'feature_scaling': True
            },
            'api': {
                'host': '0.0.0.0',
                'port': 5000,
                'debug': False
            }
        }


class ModelMetrics:
    """Calculate and manage model evaluation metrics."""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, y_pred_proba=None):
        """Calculate comprehensive evaluation metrics."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix, classification_report, roc_auc_score
        )
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            except:
                logger.warning("Could not calculate ROC-AUC score")
        
        return metrics
    
    @staticmethod
    def print_classification_report(y_true, y_pred):
        """Print detailed classification report."""
        from sklearn.metrics import classification_report
        print(classification_report(y_true, y_pred))


class DataValidator:
    """Validate data quality and integrity."""
    
    @staticmethod
    def validate_features(X, expected_shape=None):
        """Validate feature matrix."""
        errors = []
        
        if len(X.shape) != 2:
            errors.append(f"Expected 2D array, got shape {X.shape}")
        
        if expected_shape and X.shape[1] != expected_shape[1]:
            errors.append(f"Expected {expected_shape[1]} features, got {X.shape[1]}")
        
        if np.isnan(X).any():
            errors.append("Features contain NaN values")
        
        if np.isinf(X).any():
            errors.append("Features contain infinite values")
        
        if errors:
            logger.error(f"Feature validation failed: {errors}")
            return False, errors
        
        return True, []
    
    @staticmethod
    def validate_predictions(predictions):
        """Validate prediction output."""
        if predictions is None:
            return False, ["Predictions are None"]
        
        if len(predictions) == 0:
            return False, ["No predictions generated"]
        
        if np.isnan(predictions).any():
            return False, ["Predictions contain NaN values"]
        
        return True, []


class Logger:
    """Configure logging for the application."""
    
    @staticmethod
    def setup_logger(name, log_file='logs/app.log'):
        """Setup logger with file and console handlers."""
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger


class PredictionFormatter:
    """Format predictions for API responses."""
    
    @staticmethod
    def format_prediction(prediction, confidence, packet_info=None):
        """Format a single prediction."""
        is_attack = bool(prediction == 1)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'prediction': 'INTRUSION' if is_attack else 'NORMAL',
            'is_attack': is_attack,
            'confidence': float(confidence),
            'alert_level': 'HIGH' if is_attack and confidence > 0.9 else 
                          'MEDIUM' if is_attack else 'LOW',
            'packet_info': packet_info or {}
        }
    
    @staticmethod
    def format_batch_predictions(predictions, confidences, packet_ids=None):
        """Format batch predictions."""
        results = []
        
        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            result = PredictionFormatter.format_prediction(
                pred, conf, 
                packet_info={'packet_id': packet_ids[i]} if packet_ids else None
            )
            results.append(result)
        
        return {
            'total_packets': len(predictions),
            'intrusions_detected': sum(1 for p in predictions if p == 1),
            'predictions': results,
            'detection_rate': sum(1 for p in predictions if p == 1) / len(predictions)
        }
