# SmartVine AI — Production Multi-Service Container
# Precision Viticulture Vision & Ripeness Platform
# Deploys FastAPI backend (port 8000) and Streamlit dashboard (port 7860)

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy codebase
COPY . .

# Expose Streamlit (7860) and FastAPI (8000)
EXPOSE 7860 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Launch both FastAPI REST API and Streamlit Dashboard concurrently
CMD sh -c "uvicorn app.api:app --host 0.0.0.0 --port 8000 & streamlit run app/dashboard.py --server.port 7860 --server.address 0.0.0.0"
