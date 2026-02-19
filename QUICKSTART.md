# Quick Start Guide - Network Intrusion Detection System

## Overview

This guide will help you get the AI-based Network Intrusion Detection System (NIDS) up and running in 5 minutes.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- ~500MB free disk space

## Step 1: Setup Environment

### Windows Setup
```powershell
# Navigate to project directory
cd c:\Users\bisha\Downloads\Project_network

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Linux/Mac Setup
```bash
# Navigate to project directory
cd Project_network

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Generate Sample Data (Optional)

If you don't have training data, generate sample data:

```bash
python src/data_Pipelines/generate_sample_data.py
```

This creates `data/network_traffic.csv` with 10,000 sample network traffic records.

## Step 3: Train a Model

Choose one of the models to train:

```bash
# Random Forest (Recommended for beginners)
python src/models/train.py --model random_forest --data data/network_traffic.csv --save

# Gradient Boosting (Better accuracy)
python src/models/train.py --model gradient_boosting --data data/network_traffic.csv --save

# Ensemble (Best accuracy)
python src/models/train.py --model ensemble --data data/network_traffic.csv --save

# Neural Network (Deep learning)
python src/models/train.py --model neural_network --data data/network_traffic.csv --save
```

**Expected output:**
```
Training completed for random_forest
RANDOM_FOREST Model Metrics:
  Accuracy:  0.9523
  Precision: 0.9456
  Recall:    0.9234
  F1-Score:  0.9343
```

## Step 4: Start the API Server

```bash
python -m src.api.app
```

The API will start on `http://localhost:5000`

**Example: Make a Prediction**

```bash
# Using curl
curl -X POST http://localhost:5000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [100, 1, 50, 500, 600, 1, 0, 0, 0, 0],
    "packet_id": "pkt_001"
  }'
```

**Response:**
```json
{
  "timestamp": "2025-12-21T14:32:15.123456",
  "prediction": "NORMAL",
  "is_attack": false,
  "confidence": 0.95,
  "alert_level": "LOW"
}
```

## Step 5: Launch Dashboard

In a new terminal (with venv activated):

```bash
python -m src.visualization.dashboard
```

Access the dashboard at `http://localhost:8050`

The dashboard shows:
- Real-time traffic statistics
- Detection metrics
- Protocol distribution
- Recent alerts

## Common Tasks

### Check API Health
```bash
curl http://localhost:5000/health
```

### Get Detection Statistics
```bash
curl http://localhost:5000/api/v1/stats
```

### Batch Prediction
```bash
curl -X POST http://localhost:5000/api/v1/predict-batch \
  -H "Content-Type: application/json" \
  -d '{
    "packets": [
      {"features": [100, 1, 50, 500, 600, 1, 0, 0, 0, 0], "packet_id": "pkt_001"},
      {"features": [200, 2, 100, 1000, 1200, 2, 0, 0, 0, 0], "packet_id": "pkt_002"}
    ]
  }'
```

### Run Tests
```bash
pytest tests/ -v
```

## Project Structure Summary

```
src/
├── api/              # Flask REST API
├── data_Pipelines/   # Data preprocessing and feature engineering
├── models/           # ML models (RF, GB, LR, DL, Ensemble)
├── real_time/        # Real-time detection engine
├── utils/            # Helper functions and utilities
└── visualization/    # Dash dashboard

tests/
└── test_nids.py      # Unit tests

data/
├── network_traffic.csv  # Sample training data

models/              # Trained model files (created after training)
logs/                # Application logs (created automatically)
```

## Features Explained

### Input Features (for predictions)

When sending predictions to the API, provide these 10 features:

1. **duration**: Connection duration in seconds
2. **protocol_type_encoded**: TCP/UDP/ICMP (encoded as number)
3. **service_encoded**: Service type (encoded)
4. **src_bytes**: Bytes sent from source
5. **dst_bytes**: Bytes sent to destination
6. **flag_encoded**: Connection status (encoded)
7. **land**: Same source/destination (0 or 1)
8. **wrong_fragment**: Count of wrong fragments
9. **urgent**: Count of urgent packets
10. **hot**: Count of hot indicators

### Output Interpretation

```json
{
  "prediction": "INTRUSION|NORMAL",        // Detection result
  "confidence": 0.95,                      // Confidence score (0-1)
  "alert_level": "HIGH|MEDIUM|LOW",        // Severity level
  "is_attack": true|false                  // Boolean flag
}
```

## Troubleshooting

### Issue: "ModuleNotFoundError"
```bash
# Ensure virtual environment is activated
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: "Port already in use"
```bash
# Change port in command
python -m src.api.app --port 5001
```

### Issue: "No data file found"
```bash
# Generate sample data first
python src/data_Pipelines/generate_sample_data.py
```

### Issue: "Model not found"
```bash
# Train a model first
python src/models/train.py --model random_forest --data data/network_traffic.csv --save
```

## Next Steps

1. **Use Your Own Data**: Replace `data/network_traffic.csv` with your dataset
2. **Fine-tune Models**: Adjust hyperparameters in train.py
3. **Integrate with Network**: Connect to actual network interfaces
4. **Deploy to Production**: Use Docker or cloud platforms
5. **Monitor Continuously**: Set up alerts and logging

## Performance Tips

- **Faster Training**: Use smaller dataset initially
- **Better Accuracy**: Use ensemble model (combines multiple models)
- **Faster Inference**: Use Random Forest model
- **GPU Support**: Install TensorFlow-GPU for faster neural network training

## Support & Resources

- **Dataset**: NSL-KDD, CICIDS2017
- **Libraries**: scikit-learn, TensorFlow, Flask, Dash
- **Deployment**: Docker, Kubernetes, AWS, Azure, GCP

## Summary

You now have a fully functional AI-based network intrusion detection system with:
- ✅ Data pipeline for preprocessing
- ✅ Multiple ML models
- ✅ REST API for predictions
- ✅ Real-time monitoring
- ✅ Interactive dashboard
- ✅ Comprehensive testing

Start protecting your network today! 🚀

---
**Version**: 1.0.0  
**Last Updated**: December 21, 2025
