# Use slim Python base image
FROM python:3.12-slim

# Set environment variables to prevent .pyc files and set UTF-8 encoding
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8

WORKDIR /app

COPY requirements.txt /app
# Install git, install dependencies from requirements.txt, then remove git to save space
RUN apt-get update && \
    apt-get install -y git && \
    pip install -r requirements.txt && \
    apt-get remove -y git && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy source code
COPY src/ .

# Default command (allows overriding with arguments)
ENTRYPOINT ["python", "infer_ifcbbins.py"]
