# Dockerfile (API)
FROM python:3.10-slim

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code and artifacts
COPY app/ ./app/
COPY artifacts/ ./artifacts/
COPY logs/ ./logs/

# Expose API port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
