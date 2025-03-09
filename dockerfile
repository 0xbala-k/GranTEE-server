FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Set the working directory to /app
WORKDIR /app

# Install system dependencies including libc6-dev for C headers
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libc6-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create a non-root user and set permissions
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port 8000 for the API
EXPOSE $PORT

# Add health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/ || exit 1

# Run the application with uvicorn in JSON array form
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]