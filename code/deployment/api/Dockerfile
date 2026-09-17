FROM python:3.12-slim

WORKDIR /app

# Install Python dependencies
COPY code/deployment/api/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy API code
COPY code/deployment/api/app.py .

# Copy trained model
COPY models/wine_model.joblib ./model/wine_model.joblib

# Start FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]