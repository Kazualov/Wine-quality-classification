FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY code/deployment/app/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy Streamlit application
COPY code/deployment/app/app.py .

# Start Streamlit
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]