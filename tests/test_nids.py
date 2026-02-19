"""
Unit tests for the network intrusion detection system.
Tests cover data pipeline, models, and API endpoints.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_Pipelines.data_pipeline import DataPipeline, FeatureEngineer
from models.models import (
    RandomForestModel, GradientBoostingModel, LogisticRegressionModel,
    EnsembleModel
)
from utils.helpers import DataValidator, PredictionFormatter, ModelMetrics


class TestDataPipeline:
    """Test data pipeline functionality."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
            'protocol': ['TCP'] * 50 + ['UDP'] * 50,
            'attack': [0] * 80 + [1] * 20
        })
    
    @pytest.fixture
    def pipeline(self):
        """Create DataPipeline instance."""
        return DataPipeline()
    
    def test_handle_missing_values(self, pipeline, sample_df):
        """Test missing value handling."""
        df = sample_df.copy()
        df.loc[0, 'feature1'] = np.nan
        
        result = pipeline.handle_missing_values(df)
        assert not result.isnull().any().any()
    
    def test_encode_categorical(self, pipeline, sample_df):
        """Test categorical encoding."""
        df = sample_df.copy()
        result = pipeline.encode_categorical(df, fit=True)
        
        assert 'protocol' in result.columns
        assert result['protocol'].dtype in [np.int32, np.int64]
    
    def test_normalize_features(self, pipeline):
        """Test feature normalization."""
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        
        X_scaled = pipeline.normalize_features(X, fit=True)
        
        assert X_scaled.shape == X.shape
        assert np.abs(X_scaled.mean()) < 0.01  # Should be centered
    
    def test_feature_columns_tracking(self, pipeline, sample_df):
        """Test that feature columns are properly tracked."""
        df = sample_df.copy()
        df = pipeline.encode_categorical(df, fit=True)
        df = pipeline.create_features(df)
        
        assert len(pipeline.feature_columns) > 0


class TestModels:
    """Test machine learning models."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training and test data."""
        np.random.seed(42)
        n_samples = 200
        n_features = 10
        
        X_train = np.random.randn(n_samples, n_features)
        y_train = np.random.randint(0, 2, n_samples)
        
        X_test = np.random.randn(50, n_features)
        y_test = np.random.randint(0, 2, 50)
        
        return X_train, X_test, y_train, y_test
    
    def test_random_forest_training(self, sample_data):
        """Test Random Forest model training."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = RandomForestModel(n_estimators=10, max_depth=5)
        model.train(X_train, y_train)
        
        assert model.is_trained
        predictions = model.predict(X_test)
        assert predictions.shape[0] == X_test.shape[0]
    
    def test_gradient_boosting_training(self, sample_data):
        """Test Gradient Boosting model training."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = GradientBoostingModel(n_estimators=10, max_depth=3)
        model.train(X_train, y_train)
        
        assert model.is_trained
        predictions = model.predict(X_test)
        assert predictions.shape[0] == X_test.shape[0]
    
    def test_logistic_regression_training(self, sample_data):
        """Test Logistic Regression model training."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = LogisticRegressionModel()
        model.train(X_train, y_train)
        
        assert model.is_trained
        predictions = model.predict(X_test)
        assert predictions.shape[0] == X_test.shape[0]
    
    def test_model_probability_predictions(self, sample_data):
        """Test probability predictions."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = RandomForestModel(n_estimators=10)
        model.train(X_train, y_train)
        
        proba = model.predict_proba(X_test)
        assert proba.shape == (X_test.shape[0], 2)
        assert np.all((proba >= 0) & (proba <= 1))
    
    def test_model_evaluation(self, sample_data):
        """Test model evaluation metrics."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = RandomForestModel(n_estimators=10)
        model.train(X_train, y_train)
        
        metrics = model.evaluate(X_test, y_test)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_ensemble_model(self, sample_data):
        """Test ensemble model."""
        X_train, X_test, y_train, y_test = sample_data
        
        models = [
            RandomForestModel(n_estimators=10),
            GradientBoostingModel(n_estimators=10)
        ]
        
        ensemble = EnsembleModel(models)
        ensemble.train(X_train, y_train)
        
        predictions = ensemble.predict(X_test)
        assert predictions.shape[0] == X_test.shape[0]


class TestDataValidator:
    """Test data validation utilities."""
    
    def test_validate_valid_features(self):
        """Test validation of valid features."""
        X = np.array([[1, 2, 3], [4, 5, 6]])
        is_valid, errors = DataValidator.validate_features(X)
        
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_nan_features(self):
        """Test validation detects NaN values."""
        X = np.array([[1, 2, np.nan], [4, 5, 6]])
        is_valid, errors = DataValidator.validate_features(X)
        
        assert not is_valid
        assert len(errors) > 0
    
    def test_validate_infinite_features(self):
        """Test validation detects infinite values."""
        X = np.array([[1, 2, np.inf], [4, 5, 6]])
        is_valid, errors = DataValidator.validate_features(X)
        
        assert not is_valid
        assert len(errors) > 0
    
    def test_validate_shape_mismatch(self):
        """Test validation detects shape mismatches."""
        X = np.array([[1, 2, 3], [4, 5, 6]])
        is_valid, errors = DataValidator.validate_features(X, expected_shape=(None, 5))
        
        assert not is_valid


class TestPredictionFormatter:
    """Test prediction formatting utilities."""
    
    def test_format_single_prediction(self):
        """Test formatting a single prediction."""
        result = PredictionFormatter.format_prediction(
            prediction=1,
            confidence=0.95,
            packet_info={'packet_id': 'pkt_001'}
        )
        
        assert result['is_attack'] == True
        assert result['prediction'] == 'INTRUSION'
        assert result['confidence'] == 0.95
        assert 'timestamp' in result
    
    def test_format_normal_prediction(self):
        """Test formatting a normal (non-attack) prediction."""
        result = PredictionFormatter.format_prediction(
            prediction=0,
            confidence=0.98
        )
        
        assert result['is_attack'] == False
        assert result['prediction'] == 'NORMAL'
    
    def test_format_batch_predictions(self):
        """Test formatting batch predictions."""
        predictions = np.array([0, 1, 0, 1, 1])
        confidences = np.array([0.9, 0.95, 0.85, 0.92, 0.88])
        packet_ids = [f'pkt_{i}' for i in range(5)]
        
        result = PredictionFormatter.format_batch_predictions(
            predictions, confidences, packet_ids
        )
        
        assert result['total_packets'] == 5
        assert result['intrusions_detected'] == 3
        assert len(result['predictions']) == 5


class TestFeatureEngineer:
    """Test feature engineering utilities."""
    
    def test_create_ratio_features(self):
        """Test ratio feature creation."""
        df = pd.DataFrame({
            'src_bytes': [100, 200, 300],
            'dst_bytes': [50, 100, 150]
        })
        
        result = FeatureEngineer.create_ratio_features(df)
        
        assert 'bytes_ratio' in result.columns
        assert result['bytes_ratio'].iloc[0] == pytest.approx(2.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
