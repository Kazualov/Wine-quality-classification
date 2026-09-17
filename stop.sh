#!/bin/bash

set -e

# Go to project root
cd "$(dirname "$0")"

echo "Stopping Wine Quality MLOps..."

# Stop FastAPI and Streamlit containers
echo "Stopping Docker containers..."

cd code/deployment

docker compose down

cd ../..

# Stop Airflow
echo "Stopping Airflow..."

pkill -f "airflow standalone" || true

echo ""
echo "========================================="
echo "Wine Quality MLOps stopped."
echo "========================================="