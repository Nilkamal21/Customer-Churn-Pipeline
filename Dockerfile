# 1. Use an official lightweight Python runtime as a parent image
FROM python:3.10-slim

# 2. Set the working directory inside the container runtime environment
WORKDIR /app

# 3. Copy our requirements definition file into the capsule first
COPY requirements.txt .

# 4. Install the required system packages
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy application code and assets into the container
COPY src/ ./src/
COPY data/ ./data/
COPY artifacts/ ./artifacts/
COPY app.py .
COPY config.yaml .
COPY mlflow.db .

# 6. Expose the standard networking port our API will communicate over
EXPOSE 8000

# 7. Command to execute Uvicorn directly from within the container
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]