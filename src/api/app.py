"""
Real-time Flask API for network intrusion detection.
Provides endpoints for model prediction, health checks, and monitoring.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.models import RandomForestModel, GradientBoostingModel
from data_Pipelines.data_pipeline import DataPipeline
from utils.helpers import PredictionFormatter, Logger, DataValidator

# Setup logging
logger = Logger.setup_logger('nids_api', 'logs/api.log')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables for models and pipeline
model = None
data_pipeline = None
model_stats = {
    'total_predictions': 0,
    'intrusions_detected': 0,
    'api_uptime': datetime.now().isoformat()
}


@app.before_request
def initialize():
    """Initialize models on first request."""
    global model, data_pipeline
    
    if model is None:
        logger.info("Initializing models...")
        try:
            model = RandomForestModel(n_estimators=100, max_depth=20)
            data_pipeline = DataPipeline()
            logger.info("Models initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    }), 200


@app.route('/api/v1/predict', methods=['POST'])
def predict_single():
    """
    Predict intrusion for a single packet.
    
    Expected JSON format:
    {
        "features": [1.0, 2.0, 3.0, ...],
        "packet_id": "optional_id"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({'error': 'Missing features in request'}), 400
        
        features = np.array(data['features']).reshape(1, -1)
        packet_id = data.get('packet_id', 'unknown')
        
        # Validate features
        is_valid, errors = DataValidator.validate_features(features)
        if not is_valid:
            return jsonify({'error': f'Invalid features: {errors}'}), 400
        
        # Make prediction
        prediction = model.predict(features)[0]
        confidence = model.predict_proba(features)[0]
        confidence_score = max(confidence)
        
        # Format response
        result = PredictionFormatter.format_prediction(
            prediction, 
            confidence_score,
            {'packet_id': packet_id}
        )
        
        # Update statistics
        model_stats['total_predictions'] += 1
        if prediction == 1:
            model_stats['intrusions_detected'] += 1
        
        logger.info(f"Prediction made: {result}")
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/predict-batch', methods=['POST'])
def predict_batch():
    """
    Predict intrusions for multiple packets.
    
    Expected JSON format:
    {
        "packets": [
            {"features": [1.0, 2.0, ...], "packet_id": "id1"},
            {"features": [2.0, 3.0, ...], "packet_id": "id2"}
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'packets' not in data:
            return jsonify({'error': 'Missing packets in request'}), 400
        
        packets = data['packets']
        
        if not packets or len(packets) == 0:
            return jsonify({'error': 'Empty packets list'}), 400
        
        # Extract features and IDs
        features_list = [p.get('features', []) for p in packets]
        packet_ids = [p.get('packet_id', f'pkt_{i}') for i, p in enumerate(packets)]
        
        X = np.array(features_list)
        
        # Validate features
        is_valid, errors = DataValidator.validate_features(X)
        if not is_valid:
            return jsonify({'error': f'Invalid features: {errors}'}), 400
        
        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        confidences = np.max(probabilities, axis=1)
        
        # Format response
        result = PredictionFormatter.format_batch_predictions(
            predictions, confidences, packet_ids
        )
        
        # Update statistics
        model_stats['total_predictions'] += len(predictions)
        model_stats['intrusions_detected'] += sum(1 for p in predictions if p == 1)
        
        logger.info(f"Batch prediction made: {result['total_packets']} packets, {result['intrusions_detected']} intrusions")
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/stats', methods=['GET'])
def get_stats():
    """Get model statistics and performance metrics."""
    return jsonify({
        'total_predictions': model_stats['total_predictions'],
        'intrusions_detected': model_stats['intrusions_detected'],
        'detection_rate': (model_stats['intrusions_detected'] / max(model_stats['total_predictions'], 1)) * 100,
        'api_uptime': model_stats['api_uptime'],
        'current_time': datetime.now().isoformat()
    }), 200


@app.route('/api/v1/model-info', methods=['GET'])
def get_model_info():
    """Get information about the loaded model."""
    return jsonify({
        'model_type': model.model_name if model else 'Not loaded',
        'is_trained': model.is_trained if model else False,
        'metrics': model.metrics if model else {},
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/api/v1/alerts', methods=['GET'])
def get_alerts():
    """Get recent alerts and detected intrusions."""
    # This would typically query a database of alerts
    return jsonify({
        'total_alerts': model_stats['intrusions_detected'],
        'recent_alerts': [],
        'alert_threshold': 0.8,
        'timestamp': datetime.now().isoformat()
    }), 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors."""
    logger.error(f"Server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    logger.info("Starting Network Intrusion Detection System API...")
    app.run(
        host=os.getenv('API_HOST', '0.0.0.0'),
        port=int(os.getenv('API_PORT', 5000)),
        debug=os.getenv('API_DEBUG', 'False') == 'True'
    )
