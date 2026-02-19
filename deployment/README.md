# Deployment Guides

## Docker Deployment

### Build and Run
```bash
# Build image
docker build -t nids:latest .

# Run container
docker run -p 5000:5000 -p 8050:8050 nids:latest
```

### Docker Compose
```bash
# Start all services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f
```

## Kubernetes Deployment

### Create deployment
```bash
kubectl apply -f deployment/kubernetes.yaml
```

### Expose service
```bash
kubectl expose deployment nids-api --type=LoadBalancer --port=5000
```

## AWS Deployment

### EC2 Instance
```bash
# SSH into instance
ssh -i key.pem ec2-user@instance-ip

# Clone repository
git clone <repo-url>
cd Project_network

# Install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run API
python main.py api
```

### Lambda Deployment
- See `lambda_handler.py` for serverless setup
- Use API Gateway for HTTP endpoints

## Azure Deployment

### App Service
1. Create Azure App Service
2. Deploy via Git or Docker
3. Configure environment variables
4. Enable continuous deployment

### Container Instances
```bash
az container create \
  --resource-group <group> \
  --name nids-api \
  --image nids:latest \
  --port 5000
```

## GCP Deployment

### Cloud Run
```bash
gcloud run deploy nids-api \
  --source . \
  --platform managed \
  --region us-central1
```

### Compute Engine
```bash
# Create VM
gcloud compute instances create nids-vm

# SSH and deploy
gcloud compute ssh nids-vm

# Run application
python main.py api
```

## Production Checklist

- [ ] Configure HTTPS/SSL
- [ ] Setup authentication
- [ ] Configure database
- [ ] Setup logging and monitoring
- [ ] Configure auto-scaling
- [ ] Setup CI/CD pipeline
- [ ] Configure backup/recovery
- [ ] Setup alerts and notifications
- [ ] Load testing completed
- [ ] Security audit passed

## Monitoring

### Application Monitoring
- API response time
- Error rates
- Model accuracy drift
- Data quality metrics

### System Monitoring
- CPU/Memory usage
- Disk space
- Network bandwidth
- Model inference latency

### Recommended Tools
- Prometheus + Grafana
- ELK Stack (Elasticsearch, Logstash, Kibana)
- DataDog
- New Relic

## Troubleshooting

### API not responding
```bash
# Check service status
docker ps
docker logs nids-api

# Restart service
docker restart nids-api
```

### High latency
- Scale horizontally (more containers)
- Optimize model inference
- Cache predictions
- Use batch processing

### Memory issues
- Reduce batch size
- Enable model quantization
- Use lighter models
