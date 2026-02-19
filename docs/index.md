# Documentation

## Table of Contents

1. **System Architecture** - [ARCHITECTURE.md](../ARCHITECTURE.md)
2. **Quick Start Guide** - [QUICKSTART.md](../QUICKSTART.md)
3. **API Documentation** - [api.md](api.md)
4. **Training Guide** - [training.md](training.md)
5. **Deployment Guide** - [deployment/README.md](../deployment/README.md)

## Overview

This is an AI-based Network Intrusion Detection System (NIDS) built with Python and machine learning.

### Key Features
- Multiple ML models (Random Forest, Gradient Boosting, Neural Network)
- Real-time detection capabilities
- REST API for easy integration
- Interactive Dash dashboard
- Comprehensive testing suite

### Quick Links
- **Source Code**: [src/](../src/)
- **Tests**: [tests/](../tests/)
- **Sample Data**: [data/](../data/)
- **Models**: [models/](../models/) (created after training)

## Getting Started

1. Read [QUICKSTART.md](../QUICKSTART.md) for 5-minute setup
2. Follow [training.md](training.md) for model training
3. Review [api.md](api.md) for API integration
4. Check [ARCHITECTURE.md](../ARCHITECTURE.md) for system details

## Main Components

### Data Pipeline
- Handles loading, cleaning, and preprocessing network traffic data
- Encodes categorical variables
- Normalizes numerical features
- Performs train-test split

**Module**: `src/data_Pipelines/data_pipeline.py`

### Machine Learning Models
Five different models with different trade-offs:
1. **Random Forest**: Fast and accurate
2. **Gradient Boosting**: Better accuracy
3. **Logistic Regression**: Lightweight baseline
4. **Neural Network**: Deep learning approach
5. **Ensemble**: Best performance (combines models)

**Module**: `src/models/models.py`

### REST API
Flask-based API server for:
- Single packet prediction
- Batch packet prediction
- Statistics retrieval
- Model information

**Module**: `src/api/app.py`

### Real-time Detection
Monitors live network traffic and generates alerts:
- Packet processing
- Anomaly detection
- Threshold-based detection
- Alert generation

**Module**: `src/real_time/realtime_detector.py`

### Dashboard
Interactive visualization with:
- Real-time monitoring
- KPI metrics
- Traffic analysis
- Alert management

**Module**: `src/visualization/dashboard.py`

## Common Tasks

### Train a Model
```bash
python main.py train --model random_forest --data data/network_traffic.csv --save
```

### Run API Server
```bash
python main.py api --host 0.0.0.0 --port 5000
```

### Launch Dashboard
```bash
python main.py dashboard --host 0.0.0.0 --port 8050
```

### Generate Sample Data
```bash
python main.py generate-data --samples 10000
```

### Run Tests
```bash
python main.py test --coverage
```

## Support

For issues or questions:
1. Check [QUICKSTART.md](../QUICKSTART.md) troubleshooting section
2. Review [ARCHITECTURE.md](../ARCHITECTURE.md) for design details
3. Examine unit tests in [tests/](../tests/)
4. Check logs in [logs/](../logs/)

---

**Version**: 1.0.0  
**Last Updated**: December 21, 2025
