FROM --platform=linux/amd64  python:3.11-slim

WORKDIR /app

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc
RUN pip install --no-cache-dir -r requirements.txt
# RUN apt-get install python3-psycopg2

COPY source/ .

EXPOSE 8501

# Health check with timing
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "chatbot_frontend.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]