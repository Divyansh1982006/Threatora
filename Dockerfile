# NetForecast Dockerfile (NTRO PS 26153)
# Containerized offline environment for Flask web dashboard & CLI

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install minimal system dependencies for PCAP parsing & building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpcap-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source code and artifacts
COPY . .

# Expose Flask Web Dashboard port
EXPOSE 5000

# Default command: launch Gunicorn WSGI production server with dynamic port support
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 2 --timeout 120 server.app:app"]
