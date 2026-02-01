# Deployment Guide

## 🚀 Quick Deploy to Streamlit Cloud (Recommended)

### Prerequisites
- GitHub account
- Streamlit Cloud account (free at share.streamlit.io)

### Step-by-Step Deployment

#### 1. Push Code to GitHub

```bash
# Initialize git repository
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: RL Dynamic Pricing System"

# Add remote repository (replace with your repo URL)
git remote add origin https://github.com/YOUR_USERNAME/rl-pricing-system.git

# Push to GitHub
git push -u origin main
```

#### 2. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Select your repository: `YOUR_USERNAME/rl-pricing-system`
4. Set **Main file path**: `app.py`
5. Click **"Deploy!"**

Your app will be live at: `https://YOUR_USERNAME-rl-pricing-system.streamlit.app`

#### 3. Configure Secrets (Optional)

If you have API keys or secrets:

1. In Streamlit Cloud, go to your app settings
2. Click **"Secrets"**
3. Add secrets in TOML format:

```toml
[api]
key = "your-api-key"
```

Access in code:
```python
import streamlit as st
api_key = st.secrets["api"]["key"]
```

---

## 🐳 Docker Deployment

### Local Docker Deployment

```bash
# Build the image
docker build -t rl-pricing-system .

# Run the container
docker run -p 8501:8501 rl-pricing-system

# Or use docker-compose
docker-compose up -d
```

Access at: http://localhost:8501

### Docker Hub Deployment

```bash
# Login to Docker Hub
docker login

# Tag the image
docker tag rl-pricing-system YOUR_USERNAME/rl-pricing-system:latest

# Push to Docker Hub
docker push YOUR_USERNAME/rl-pricing-system:latest

# Deploy anywhere
docker pull YOUR_USERNAME/rl-pricing-system:latest
docker run -p 8501:8501 YOUR_USERNAME/rl-pricing-system:latest
```

---

## ☁️ AWS Deployment

### AWS EC2

```bash
# SSH into your EC2 instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Install Docker
sudo apt update
sudo apt install docker.io -y
sudo systemctl start docker

# Pull and run
sudo docker pull YOUR_USERNAME/rl-pricing-system
sudo docker run -d -p 8501:8501 YOUR_USERNAME/rl-pricing-system

# Configure security group to allow port 8501
```

Access at: http://your-ec2-ip:8501

### AWS ECS (Fargate)

1. Push Docker image to Amazon ECR
2. Create ECS cluster
3. Create task definition with your image
4. Create service
5. Configure load balancer

---

## 🌐 Google Cloud Platform

### Google Cloud Run

```bash
# Install gcloud CLI
# https://cloud.google.com/sdk/docs/install

# Initialize
gcloud init

# Deploy
gcloud run deploy rl-pricing \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8501

# Get URL
gcloud run services describe rl-pricing --region us-central1 --format 'value(status.url)'
```

---

## 🔧 Heroku Deployment

### Setup Files

Create `setup.sh`:
```bash
mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

Create `Procfile`:
```
web: sh setup.sh && streamlit run app.py
```

### Deploy

```bash
# Login to Heroku
heroku login

# Create app
heroku create your-app-name

# Push to Heroku
git push heroku main

# Open app
heroku open
```

---

## 🔒 Production Best Practices

### 1. Environment Variables

Store sensitive config in environment variables:

```python
import os

API_KEY = os.getenv('API_KEY')
DATABASE_URL = os.getenv('DATABASE_URL')
```

### 2. Logging

Add proper logging:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
```

### 3. Error Handling

Wrap critical sections:

```python
try:
    # Critical operation
    result = model.predict()
except Exception as e:
    logger.error(f"Prediction failed: {e}")
    st.error("An error occurred. Please try again.")
```

### 4. Caching

Use Streamlit caching for expensive operations:

```python
@st.cache_data
def load_model():
    return trained_model

@st.cache_resource
def get_database_connection():
    return db.connect()
```

### 5. Rate Limiting

Implement rate limiting for API endpoints:

```python
from functools import wraps
import time

def rate_limit(max_calls=10, time_window=60):
    calls = []
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            calls[:] = [c for c in calls if c > now - time_window]
            
            if len(calls) >= max_calls:
                raise Exception("Rate limit exceeded")
            
            calls.append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator
```

### 6. Monitoring

Add health check endpoint and monitoring:

```python
import psutil

def system_health():
    return {
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_percent': psutil.disk_usage('/').percent
    }
```

---

## 📊 Performance Optimization

### 1. Model Serialization

Save trained models for faster loading:

```python
# Save
torch.save(model.state_dict(), 'model.pth')

# Load
model.load_state_dict(torch.load('model.pth'))
```

### 2. Data Caching

Cache frequently accessed data:

```python
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_historical_data():
    return pd.read_csv('data.csv')
```

### 3. Lazy Loading

Load resources only when needed:

```python
if 'model' not in st.session_state:
    st.session_state.model = load_model()
```

---

## 🔍 Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Find process using port 8501
   lsof -i :8501
   # Kill process
   kill -9 PID
   ```

2. **Memory issues**
   - Reduce batch size
   - Use smaller models
   - Enable garbage collection

3. **Slow loading**
   - Enable caching
   - Optimize data loading
   - Use CDN for static assets

### Debug Mode

Run with debug logging:

```bash
streamlit run app.py --logger.level=debug
```

---

## 📝 Post-Deployment Checklist

- [ ] Test all features in production
- [ ] Set up monitoring and alerts
- [ ] Configure backup strategy
- [ ] Document API endpoints
- [ ] Set up CI/CD pipeline
- [ ] Enable HTTPS
- [ ] Configure domain name
- [ ] Set up analytics
- [ ] Create user documentation
- [ ] Plan for scaling

---

## 🆘 Support

For deployment issues:
- Check Streamlit docs: https://docs.streamlit.io
- Community forum: https://discuss.streamlit.io
- GitHub issues: Create issue in repository

---

**Happy Deploying! 🚀**
