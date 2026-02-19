"""
Real-time network intrusion detection engine.
Monitors live network traffic and generates alerts.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque
import threading

logger = logging.getLogger(__name__)


class RealtimeDetector:
    """Real-time network traffic anomaly detector."""
    
    def __init__(self, model=None, buffer_size=100):
        self.model = model
        self.buffer_size = buffer_size
        self.packet_buffer = deque(maxlen=buffer_size)
        self.alert_threshold = 0.7
        self.is_monitoring = False
        self.detection_stats = {
            'packets_processed': 0,
            'intrusions_detected': 0,
            'false_positives': 0
        }
    
    def set_model(self, model):
        """Set the detection model."""
        self.model = model
        logger.info(f"Model set: {model.model_name}")
    
    def process_packet(self, packet_features, packet_id=None):
        """
        Process a single packet and detect intrusions.
        
        Args:
            packet_features: numpy array of feature values
            packet_id: optional packet identifier
        
        Returns:
            dict with detection results
        """
        if self.model is None:
            logger.error("Model not set")
            return None
        
        # Ensure correct shape
        if len(packet_features.shape) == 1:
            packet_features = packet_features.reshape(1, -1)
        
        # Make prediction
        prediction = self.model.predict(packet_features)[0]
        confidence = self.model.predict_proba(packet_features)[0]
        confidence_score = max(confidence)
        
        # Add to buffer
        self.packet_buffer.append({
            'timestamp': datetime.now(),
            'packet_id': packet_id or f'pkt_{self.detection_stats["packets_processed"]}',
            'prediction': prediction,
            'confidence': confidence_score,
            'features': packet_features[0]
        })
        
        # Update statistics
        self.detection_stats['packets_processed'] += 1
        if prediction == 1:
            self.detection_stats['intrusions_detected'] += 1
        
        # Generate alert if needed
        result = {
            'packet_id': packet_id,
            'prediction': 'INTRUSION' if prediction == 1 else 'NORMAL',
            'confidence': float(confidence_score),
            'is_intrusion': bool(prediction == 1),
            'timestamp': datetime.now().isoformat(),
            'alert': self._generate_alert(prediction, confidence_score)
        }
        
        logger.info(f"Packet {packet_id}: {result['prediction']} (confidence: {confidence_score:.2f})")
        return result
    
    def process_batch(self, packets_data):
        """
        Process multiple packets.
        
        Args:
            packets_data: list of dicts with 'features' and optional 'packet_id'
        
        Returns:
            list of detection results
        """
        results = []
        
        for packet in packets_data:
            features = packet.get('features')
            packet_id = packet.get('packet_id')
            
            result = self.process_packet(np.array(features), packet_id)
            results.append(result)
        
        return results
    
    def _generate_alert(self, prediction, confidence):
        """Generate alert if intrusion detected."""
        if prediction == 1 and confidence > self.alert_threshold:
            return {
                'level': 'HIGH' if confidence > 0.9 else 'MEDIUM',
                'message': f'Intrusion detected with {confidence:.2%} confidence',
                'action': 'BLOCK' if confidence > 0.95 else 'MONITOR'
            }
        return None
    
    def get_statistics(self):
        """Get detection statistics."""
        total = self.detection_stats['packets_processed']
        intrusions = self.detection_stats['intrusions_detected']
        
        return {
            'packets_processed': total,
            'intrusions_detected': intrusions,
            'detection_rate': (intrusions / total * 100) if total > 0 else 0,
            'false_positives': self.detection_stats['false_positives']
        }
    
    def get_buffer_summary(self):
        """Get summary of buffered packets."""
        if not self.packet_buffer:
            return None
        
        df = pd.DataFrame(list(self.packet_buffer))
        
        return {
            'total_packets': len(df),
            'intrusions': (df['prediction'] == 1).sum(),
            'avg_confidence': df['confidence'].mean(),
            'time_range': {
                'start': df['timestamp'].min().isoformat(),
                'end': df['timestamp'].max().isoformat()
            }
        }
    
    def reset_statistics(self):
        """Reset detection statistics."""
        self.detection_stats = {
            'packets_processed': 0,
            'intrusions_detected': 0,
            'false_positives': 0
        }
        logger.info("Statistics reset")


class AnomalyDetector:
    """Detect anomalies in network traffic patterns."""
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.traffic_history = deque(maxlen=window_size)
        self.baseline_stats = {
            'mean_packet_size': 0,
            'std_packet_size': 0,
            'mean_packets_per_second': 0
        }
    
    def update_baseline(self, traffic_data):
        """
        Update baseline statistics from normal traffic.
        
        Args:
            traffic_data: pandas DataFrame with traffic statistics
        """
        self.baseline_stats = {
            'mean_packet_size': traffic_data['packet_size'].mean(),
            'std_packet_size': traffic_data['packet_size'].std(),
            'mean_packets_per_second': traffic_data['packets_per_second'].mean()
        }
        logger.info(f"Baseline updated: {self.baseline_stats}")
    
    def detect_anomalies(self, current_traffic):
        """
        Detect statistical anomalies in current traffic.
        
        Args:
            current_traffic: dict with current traffic metrics
        
        Returns:
            dict with anomaly scores
        """
        anomalies = {}
        
        # Check packet size anomaly
        if 'packet_size' in current_traffic:
            z_score = abs(
                (current_traffic['packet_size'] - self.baseline_stats['mean_packet_size']) /
                (self.baseline_stats['std_packet_size'] + 1e-6)
            )
            anomalies['packet_size_anomaly'] = z_score > 3
        
        # Check traffic rate anomaly
        if 'packets_per_second' in current_traffic:
            expected = self.baseline_stats['mean_packets_per_second']
            actual = current_traffic['packets_per_second']
            anomalies['traffic_rate_anomaly'] = actual > expected * 2
        
        return anomalies


class ThresholdDetector:
    """Simple threshold-based intrusion detection."""
    
    def __init__(self):
        self.thresholds = {
            'bytes_per_second': 1000000,  # 1 MB/s
            'packets_per_second': 10000,
            'connection_attempts': 100,
            'protocol_ratio': 0.9  # More than 90% one protocol
        }
    
    def detect(self, traffic_metrics):
        """
        Detect intrusions based on thresholds.
        
        Args:
            traffic_metrics: dict with traffic metrics
        
        Returns:
            boolean indicating if anomaly detected
        """
        anomalies = []
        
        for metric, threshold in self.thresholds.items():
            if metric in traffic_metrics:
                if traffic_metrics[metric] > threshold:
                    anomalies.append(metric)
        
        return len(anomalies) > 0, anomalies
    
    def set_threshold(self, metric, value):
        """Update a specific threshold."""
        self.thresholds[metric] = value
        logger.info(f"Threshold updated: {metric} = {value}")
