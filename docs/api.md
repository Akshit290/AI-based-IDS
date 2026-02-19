# REST API Documentation

## Base URL
```
http://localhost:5000
```

## Authentication
Currently no authentication required. For production, add API key authentication.

---

## Endpoints

### Health Check
Check if the API server is running.

**Request:**
```
GET /health
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "timestamp": "2025-12-21T14:32:15.123456",
  "version": "1.0.0"
}
```

---

### Single Prediction
Predict if a single network packet is malicious.

**Request:**
```
POST /api/v1/predict
Content-Type: application/json
```

**Body:**
```json
{
  "features": [100.0, 1, 50, 500, 600, 1, 0, 0, 0, 0],
  "packet_id": "pkt_001"
}
```

**Response (200 OK):**
```json
{
  "timestamp": "2025-12-21T14:32:15.123456",
  "prediction": "NORMAL",
  "is_attack": false,
  "confidence": 0.95,
  "alert_level": "LOW",
  "packet_info": {
    "packet_id": "pkt_001"
  }
}
```

**Error Response (400 Bad Request):**
```json
{
  "error": "Missing features in request"
}
```

---

### Batch Prediction
Predict multiple packets at once for better performance.

**Request:**
```
POST /api/v1/predict-batch
Content-Type: application/json
```

**Body:**
```json
{
  "packets": [
    {
      "features": [100.0, 1, 50, 500, 600, 1, 0, 0, 0, 0],
      "packet_id": "pkt_001"
    },
    {
      "features": [200.0, 2, 100, 1000, 1200, 2, 0, 0, 0, 0],
      "packet_id": "pkt_002"
    }
  ]
}
```

**Response (200 OK):**
```json
{
  "total_packets": 2,
  "intrusions_detected": 0,
  "detection_rate": 0.0,
  "predictions": [
    {
      "timestamp": "2025-12-21T14:32:15.123456",
      "prediction": "NORMAL",
      "is_attack": false,
      "confidence": 0.95,
      "alert_level": "LOW",
      "packet_info": {
        "packet_id": "pkt_001"
      }
    },
    {
      "timestamp": "2025-12-21T14:32:15.234567",
      "prediction": "NORMAL",
      "is_attack": false,
      "confidence": 0.88,
      "alert_level": "LOW",
      "packet_info": {
        "packet_id": "pkt_002"
      }
    }
  ]
}
```

---

### Get Statistics
Retrieve detection statistics.

**Request:**
```
GET /api/v1/stats
```

**Response (200 OK):**
```json
{
  "total_predictions": 1500,
  "intrusions_detected": 45,
  "detection_rate": 3.0,
  "api_uptime": "2025-12-21T10:00:00.000000",
  "current_time": "2025-12-21T14:32:15.123456"
}
```

---

### Get Model Info
Get information about the loaded detection model.

**Request:**
```
GET /api/v1/model-info
```

**Response (200 OK):**
```json
{
  "model_type": "random_forest",
  "is_trained": true,
  "metrics": {
    "accuracy": 0.9523,
    "precision": 0.9456,
    "recall": 0.9234,
    "f1": 0.9343,
    "confusion_matrix": [
      [2000, 100],
      [50, 250]
    ]
  },
  "timestamp": "2025-12-21T14:32:15.123456"
}
```

---

### Get Alerts
Retrieve recent detected intrusions/alerts.

**Request:**
```
GET /api/v1/alerts
```

**Response (200 OK):**
```json
{
  "total_alerts": 45,
  "recent_alerts": [
    {
      "timestamp": "2025-12-21T14:30:00",
      "source_ip": "192.168.1.105",
      "destination_ip": "10.0.0.50",
      "attack_type": "Port Scan",
      "severity": "HIGH",
      "action": "BLOCKED"
    }
  ],
  "alert_threshold": 0.8,
  "timestamp": "2025-12-21T14:32:15.123456"
}
```

---

## Feature Format

### Input Features (10 values)
1. **duration** (float): Connection duration in seconds
   - Range: 0-1000+
   - Example: 100.5

2. **protocol_type** (int): Encoded protocol
   - 0: TCP, 1: UDP, 2: ICMP
   - Example: 1

3. **service** (int): Encoded service type
   - 0: HTTP, 1: FTP, 2: DNS, etc.
   - Example: 50

4. **src_bytes** (float): Source to destination bytes
   - Range: 0-1000000+
   - Example: 500

5. **dst_bytes** (float): Destination to source bytes
   - Range: 0-1000000+
   - Example: 600

6. **flag** (int): Connection status flags
   - Encoded value
   - Example: 1

7. **land** (int): Same source/destination indicator
   - 0: Different, 1: Same
   - Example: 0

8. **wrong_fragment** (int): Wrong fragments count
   - Range: 0-20
   - Example: 0

9. **urgent** (int): Urgent packets count
   - Range: 0-10
   - Example: 0

10. **hot** (int): Hot indicators count
    - Range: 0-20
    - Example: 0

### Response Fields

- **prediction**: "INTRUSION" or "NORMAL"
- **is_attack**: Boolean (true if intrusion)
- **confidence**: Float (0.0-1.0)
  - 0.9+: Very confident
  - 0.7-0.9: Confident
  - Below 0.7: Less confident

- **alert_level**: "HIGH", "MEDIUM", "LOW"
  - HIGH: Intrusion with high confidence (>0.9)
  - MEDIUM: Intrusion with moderate confidence
  - LOW: Normal traffic

- **timestamp**: ISO format timestamp of prediction

---

## Error Responses

### 400 Bad Request
Missing or invalid parameters.
```json
{
  "error": "Missing features in request"
}
```

### 404 Not Found
Endpoint does not exist.
```json
{
  "error": "Endpoint not found"
}
```

### 500 Internal Server Error
Server error during prediction.
```json
{
  "error": "Internal server error"
}
```

---

## Rate Limiting

Currently no rate limiting is implemented. For production, add:
- Request rate limiting (e.g., 1000 req/minute)
- Batch size limits (e.g., max 10000 packets)
- Timeout configuration

---

## Examples

### Python Client
```python
import requests
import json

# Single prediction
url = "http://localhost:5000/api/v1/predict"
data = {
    "features": [100, 1, 50, 500, 600, 1, 0, 0, 0, 0],
    "packet_id": "pkt_001"
}
response = requests.post(url, json=data)
print(response.json())

# Batch prediction
url = "http://localhost:5000/api/v1/predict-batch"
data = {
    "packets": [
        {"features": [100, 1, 50, 500, 600, 1, 0, 0, 0, 0], "packet_id": "pkt_001"},
        {"features": [200, 2, 100, 1000, 1200, 2, 0, 0, 0, 0], "packet_id": "pkt_002"}
    ]
}
response = requests.post(url, json=data)
print(response.json())
```

### cURL Examples
```bash
# Health check
curl http://localhost:5000/health

# Single prediction
curl -X POST http://localhost:5000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [100, 1, 50, 500, 600, 1, 0, 0, 0, 0], "packet_id": "pkt_001"}'

# Get statistics
curl http://localhost:5000/api/v1/stats
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-21 | Initial release |

---

**Last Updated**: December 21, 2025
