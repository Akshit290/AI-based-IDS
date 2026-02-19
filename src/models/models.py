"""
Machine learning models for network intrusion detection.
Includes Random Forest, Gradient Boosting, and Neural Network models.
"""

import numpy as np
import logging
import joblib
from datetime import datetime
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_auc_score

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

logger = logging.getLogger(__name__)


class BaseModel:
    """Base class for all models."""
    
    def __init__(self, model_name):
        self.model = None
        self.model_name = model_name
        self.metrics = {}
        self.is_trained = False
    
    def save_model(self, path='models/'):
        """Save trained model to disk."""
        Path(path).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = Path(path) / f'{self.model_name}_{timestamp}.pkl'
        
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
        return filepath
    
    def load_model(self, filepath):
        """Load model from disk."""
        self.model = joblib.load(filepath)
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        if not self.is_trained:
            logger.error("Model is not trained yet")
            return None
        
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        self.metrics = metrics
        logger.info(f"Model Evaluation Metrics:\n{self._format_metrics(metrics)}")
        return metrics
    
    def _format_metrics(self, metrics):
        """Format metrics for display."""
        return f"""
        Accuracy:  {metrics['accuracy']:.4f}
        Precision: {metrics['precision']:.4f}
        Recall:    {metrics['recall']:.4f}
        F1-Score:  {metrics['f1']:.4f}
        """


class RandomForestModel(BaseModel):
    """Random Forest Classifier for intrusion detection."""
    
    def __init__(self, n_estimators=100, max_depth=20, random_state=42):
        super().__init__('random_forest')
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )
    
    def train(self, X_train, y_train):
        """Train the Random Forest model."""
        logger.info("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info("Random Forest training completed")
    
    def predict(self, X):
        """Make predictions."""
        if not self.is_trained:
            logger.error("Model not trained yet")
            return None
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        if not self.is_trained:
            logger.error("Model not trained yet")
            return None
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, feature_names=None):
        """Get feature importance scores."""
        if not self.is_trained:
            return None
        
        importances = self.model.feature_importances_
        
        if feature_names:
            return dict(zip(feature_names, importances))
        return importances


class GradientBoostingModel(BaseModel):
    """Gradient Boosting Classifier for intrusion detection."""
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42):
        super().__init__('gradient_boosting')
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            verbose=1
        )
    
    def train(self, X_train, y_train):
        """Train the Gradient Boosting model."""
        logger.info("Training Gradient Boosting model...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info("Gradient Boosting training completed")
    
    def predict(self, X):
        """Make predictions."""
        if not self.is_trained:
            logger.error("Model not trained yet")
            return None
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        if not self.is_trained:
            logger.error("Model not trained yet")
            return None
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, feature_names=None):
        """Get feature importance scores."""
        if not self.is_trained:
            return None
        
        importances = self.model.feature_importances_
        
        if feature_names:
            return dict(zip(feature_names, importances))
        return importances


class LogisticRegressionModel(BaseModel):
    """Logistic Regression for baseline comparison."""
    
    def __init__(self, random_state=42, max_iter=1000):
        super().__init__('logistic_regression')
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=max_iter,
            n_jobs=-1
        )
    
    def train(self, X_train, y_train):
        """Train the Logistic Regression model."""
        logger.info("Training Logistic Regression model...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info("Logistic Regression training completed")
    
    def predict(self, X):
        """Make predictions."""
        if not self.is_trained:
            logger.error("Model not trained yet")
            return None
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        if not self.is_trained:
            logger.error("Model not trained yet")
            return None
        return self.model.predict_proba(X)


class DeepLearningModel(BaseModel):
    """Neural Network model for intrusion detection."""
    
    def __init__(self, input_dim, hidden_layers=[128, 64, 32]):
        super().__init__('neural_network')
        
        if not KERAS_AVAILABLE:
            logger.error("TensorFlow/Keras not available")
            raise ImportError("Please install tensorflow: pip install tensorflow")
        
        self.model = self._build_model(input_dim, hidden_layers)
    
    def _build_model(self, input_dim, hidden_layers):
        """Build neural network architecture."""
        model = Sequential()
        model.add(Dense(hidden_layers[0], input_dim=input_dim, activation='relu'))
        model.add(Dropout(0.3))
        
        for units in hidden_layers[1:]:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(0.3))
        
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, epochs=20, batch_size=32, validation_split=0.2):
        """Train the neural network model."""
        logger.info("Training Neural Network model...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        self.is_trained = True
        logger.info("Neural Network training completed")
        return history
    
    def predict(self, X):
        """Make predictions."""
        if not self.is_trained:
            logger.error("Model not trained yet")
            return None
        predictions = self.model.predict(X, verbose=0)
        return (predictions > 0.5).astype(int).flatten()
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        if not self.is_trained:
            logger.error("Model not trained yet")
            return None
        
        proba = self.model.predict(X, verbose=0)
        return np.column_stack([1 - proba, proba])


class EnsembleModel:
    """Ensemble of multiple models for improved detection."""
    
    def __init__(self, models):
        self.models = models
        self.is_trained = False
    
    def train(self, X_train, y_train):
        """Train all models in the ensemble."""
        for model in self.models:
            model.train(X_train, y_train)
        self.is_trained = True
        logger.info("Ensemble training completed")
    
    def predict(self, X, voting='hard'):
        """Make predictions using ensemble voting."""
        if not self.is_trained:
            logger.error("Ensemble not trained yet")
            return None
        
        predictions = np.array([model.predict(X) for model in self.models])
        
        if voting == 'hard':
            # Majority voting
            final_pred = (predictions.sum(axis=0) > len(self.models) / 2).astype(int)
        else:
            # Soft voting - average probabilities
            proba = np.array([model.predict_proba(X)[:, 1] for model in self.models])
            final_pred = (proba.mean(axis=0) > 0.5).astype(int)
        
        return final_pred
    
    def predict_proba(self, X):
        """Get ensemble prediction probabilities."""
        if not self.is_trained:
            logger.error("Ensemble not trained yet")
            return None
        
        proba = np.array([model.predict_proba(X)[:, 1] for model in self.models])
        avg_proba = proba.mean(axis=0)
        
        return np.column_stack([1 - avg_proba, avg_proba])
    
    def evaluate(self, X_test, y_test):
        """Evaluate ensemble performance."""
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        }
        
        logger.info(f"Ensemble Metrics: {metrics}")
        return metrics
