"""
Data pipeline for network intrusion detection system.
Handles data loading, preprocessing, and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPipeline:
    """Handles data loading, cleaning, and preprocessing."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.target_column = 'attack'
        
    def load_data(self, filepath):
        """Load network intrusion dataset."""
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded data with shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset."""
        missing = df.isnull().sum()
        if missing.sum() > 0:
            logger.warning(f"Missing values found:\n{missing[missing > 0]}")
            # Fill numeric columns with median
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            # Fill categorical columns with mode
            cat_cols = df.select_dtypes(include=['object']).columns
            for col in cat_cols:
                df[col] = df[col].fillna(df[col].mode()[0])
        return df
    
    def encode_categorical(self, df, fit=True):
        """Encode categorical variables."""
        cat_columns = df.select_dtypes(include=['object']).columns
        
        for col in cat_columns:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                logger.info(f"Encoded {col} with {len(self.label_encoders[col].classes_)} classes")
            else:
                if col in self.label_encoders:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df
    
    def create_features(self, df):
        """Create and engineer features."""
        # Basic feature engineering
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target column if present
        if self.target_column in numeric_cols:
            numeric_cols.remove(self.target_column)
        
        self.feature_columns = numeric_cols
        return df
    
    def normalize_features(self, X, fit=True):
        """Normalize numerical features."""
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            logger.info("Scaler fitted and features normalized")
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def prepare_data(self, filepath, test_size=0.2, random_state=42):
        """Complete pipeline: load, clean, encode, normalize, and split data."""
        # Load data
        df = self.load_data(filepath)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Encode categorical variables
        df = self.encode_categorical(df, fit=True)
        
        # Create features
        df = self.create_features(df)
        
        # Separate features and target
        if self.target_column not in df.columns:
            logger.warning(f"Target column '{self.target_column}' not found")
            X = df[self.feature_columns]
            y = None
        else:
            X = df[self.feature_columns]
            y = df[self.target_column]
        
        # Normalize features
        X_scaled = self.normalize_features(X, fit=True)
        
        # Split data
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=random_state
            )
            logger.info(f"Data split: train {X_train.shape}, test {X_test.shape}")
            return X_train, X_test, y_train, y_test
        else:
            logger.warning("No target variable found, returning features only")
            return X_scaled, None, None, None
    
    def transform_new_data(self, new_data_df):
        """Transform new incoming data using fitted scaler and encoders."""
        # Encode categorical variables using fitted encoders
        df = new_data_df.copy()
        df = self.encode_categorical(df, fit=False)
        
        # Select only the features used during training
        X = df[self.feature_columns]
        
        # Normalize using fitted scaler
        X_scaled = self.normalize_features(X, fit=False)
        
        return X_scaled


class FeatureEngineer:
    """Advanced feature engineering for network traffic."""
    
    @staticmethod
    def aggregate_traffic(df, window='1H'):
        """Aggregate traffic statistics by time window."""
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            return df.resample(window).agg({
                'bytes': 'sum',
                'packets': 'sum',
                'duration': 'mean'
            })
        return df
    
    @staticmethod
    def create_statistical_features(df):
        """Create statistical features from raw metrics."""
        features = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            features[f'{col}_mean'] = df[col].mean()
            features[f'{col}_std'] = df[col].std()
            features[f'{col}_min'] = df[col].min()
            features[f'{col}_max'] = df[col].max()
        
        return pd.DataFrame([features])
    
    @staticmethod
    def create_ratio_features(df):
        """Create ratio features for anomaly detection."""
        if 'src_bytes' in df.columns and 'dst_bytes' in df.columns:
            df['bytes_ratio'] = df['src_bytes'] / (df['dst_bytes'] + 1)
        
        if 'src_packets' in df.columns and 'dst_packets' in df.columns:
            df['packet_ratio'] = df['src_packets'] / (df['dst_packets'] + 1)
        
        return df
