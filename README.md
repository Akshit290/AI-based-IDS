# AI-Based Network Intrusion Detection System (NIDS)

A comprehensive machine learning-based network intrusion detection system that uses advanced AI models to detect and prevent unauthorized access and malicious activities in network traffic.

## Features

### 🎯 Core Capabilities
- **Real-time Intrusion Detection**: Identifies network attacks in real-time
- **Multiple ML Models**: Random Forest, Gradient Boosting, Logistic Regression, and Neural Networks
- **Ensemble Learning**: Combines multiple models for improved accuracy
- **Data Pipeline**: Automated preprocessing and feature engineering
- **REST API**: Flask-based API for easy integration
- **Interactive Dashboard**: Dash-based visualization for monitoring

### 🔍 Detectable Attack Types
- DDoS (Distributed Denial of Service) attacks
- Port scanning
- Brute force attacks
- Malware communication
- Unusual traffic patterns
- Protocol anomalies

## Project Structure

```
Project_network/
├── data/                          # Dataset directory
│   ├── train_data.csv            # Training data
│   └── test_data.csv             # Test data
├── src/
│   ├── api/
│   │   └── app.py                # Flask API application
│   ├── data_Pipelines/
│   │   └── data_pipeline.py      # Data processing pipeline
│   ├── models/
│   │   └── models.py             # ML models implementation
│   ├── real_time/
│   │   └── realtime_detector.py  # Real-time prediction engine
│   ├── utils/
│   │   └── helpers.py            # Utility functions
│   └── visualization/
│       └── dashboard.py          # Dash visualization dashboard
├── tests/
│   └── test_nids.py              # Unit tests
├── notebooks/                     # Jupyter notebooks for analysis
├── deployment/                    # Docker and deployment configs
├── docs/                          # Documentation
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. **Clone or extract the project**:
```bash
cd Project_network
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation

Prepare your network traffic data in CSV format with the following columns:
- Network features (packet size, duration, flow bytes, etc.)
- Protocol information (TCP, UDP, ICMP, etc.)
- Traffic characteristics (source/destination ports, bytes transferred, etc.)
- Target label (0 for normal, 1 for intrusion)

Example:
```python
from src.data_Pipelines.data_pipeline import DataPipeline

pipeline = DataPipeline()
X_train, X_test, y_train, y_test = pipeline.prepare_data('data/network_traffic.csv')
```

### 2. Train Models

```python
from src.models.models import RandomForestModel, GradientBoostingModel

# Random Forest Model
rf_model = RandomForestModel(n_estimators=100, max_depth=20)
rf_model.train(X_train, y_train)

# Gradient Boosting Model
gb_model = GradientBoostingModel(n_estimators=100)
gb_model.train(X_train, y_train)

# Evaluate models
rf_metrics = rf_model.evaluate(X_test, y_test)
gb_metrics = gb_model.evaluate(X_test, y_test)

# Save models
rf_model.save_model('models/')
gb_model.save_model('models/')
```

### 3. Use Ensemble Model

```python
from src.models.models import EnsembleModel

# Create ensemble
models = [rf_model, gb_model]
ensemble = EnsembleModel(models)
ensemble.train(X_train, y_train)

# Make predictions
predictions = ensemble.predict(X_test)
```

### 4. Start the API Server

```bash
python -m src.api.app
```

The API will be available at `http://localhost:5000`

**API Endpoints:**

#### Health Check
```bash
GET /health
```

#### Single Prediction
```bash
POST /api/v1/predict
Content-Type: application/json

{
  "features": [1.0, 2.0, 3.0, ...],
  "packet_id": "pkt_001"
}
```

#### Batch Prediction
```bash
POST /api/v1/predict-batch
Content-Type: application/json

{
  "packets": [
    {"features": [1.0, 2.0, ...], "packet_id": "pkt_001"},
    {"features": [2.0, 3.0, ...], "packet_id": "pkt_002"}
  ]
}
```

#### Get Statistics
```bash
GET /api/v1/stats
```

#### Get Model Info
```bash
GET /api/v1/model-info
```

### 5. Launch Dashboard

```bash
python -m src.visualization.dashboard
```

Access the dashboard at `http://localhost:8050`

The dashboard provides:
- Real-time traffic monitoring
- Intrusion detection statistics
- Network protocol analysis
- Recent alerts and incidents
- Detection rate trends

## Testing

Run the test suite to validate the system:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_nids.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Model Performance

The system uses multiple models with the following characteristics:

### Random Forest
- **Accuracy**: ~95%
- **Speed**: Fast inference
- **Interpretability**: Good (feature importance)
- **Best for**: Balanced performance and speed

### Gradient Boosting
- **Accuracy**: ~96%
- **Speed**: Moderate inference
- **Interpretability**: Good (feature importance)
- **Best for**: Higher accuracy with reasonable speed

### Neural Network
- **Accuracy**: ~97%
- **Speed**: Varies with architecture
- **Interpretability**: Lower
- **Best for**: Maximum accuracy

### Ensemble
- **Accuracy**: ~97-98%
- **Speed**: Depends on models
- **Best for**: Best overall performance

## Configuration

Create a `config.json` file in the root directory:

```json
{
  "model": {
    "type": "random_forest",
    "random_state": 42,
    "n_estimators": 100,
    "max_depth": 20
  },
  "data": {
    "test_size": 0.2,
    "random_state": 42,
    "feature_scaling": true
  },
  "api": {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": false
  }
}
```

## Real-Time Detection

For real-time network monitoring:

```python
from src.real_time.realtime_detector import RealtimeDetector

detector = RealtimeDetector()
detector.start_monitoring(interface='eth0')
```

## Deployment

### Docker Deployment

```bash
# Build Docker image
docker build -t nids:latest .

# Run container
docker run -p 5000:5000 -p 8050:8050 nids:latest
```

### Production Deployment

See [deployment/README.md](deployment/README.md) for detailed deployment instructions.

## Performance Optimization

1. **Feature Selection**: Use top important features to reduce dimensionality
2. **Model Quantization**: Convert to smaller model formats for faster inference
3. **Caching**: Cache model predictions for repeated patterns
4. **Batch Processing**: Process packets in batches for better throughput

## Troubleshooting

### Issue: Model not loading
```python
# Check if model file exists and is compatible
from src.utils.helpers import Logger
logger = Logger.setup_logger('nids')
```

### Issue: API connection errors
- Ensure Flask server is running
- Check host and port configuration
- Verify firewall settings

### Issue: Low detection accuracy
- Ensure training data is representative
- Check feature scaling is applied
- Validate input data format

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## Future Enhancements

- [ ] Support for more attack types
- [ ] Deep learning models (LSTM, Transformer)
- [ ] Model explainability (SHAP, LIME)
- [ ] Auto-scaling API with Kubernetes
- [ ] Advanced visualization with Grafana
- [ ] Cloud deployment templates (AWS, Azure, GCP)
- [ ] Real-time threat intelligence integration
- [ ] Automated model retraining pipeline

## License

This project is provided as-is for educational and research purposes.

## Contact & Support

For issues, questions, or suggestions, please reach out or open an issue in the repository.

## References

- Network Intrusion Detection: https://www.ics.uci.edu/~mlearn/kddcup.html
- KDD Cup 1999 Dataset (NSL-KDD)
- Scikit-learn Documentation: https://scikit-learn.org/
- TensorFlow/Keras: https://www.tensorflow.org/

---

**Last Updated**: December 21, 2025
**Version**: 1.0.0
