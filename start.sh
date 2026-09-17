#!/bin/bash

set -e

# Go to project root
cd "$(dirname "$0")"

# Activate Airflow environment
source .airflow_venv/bin/activate

# Set Airflow home
export AIRFLOW_HOME="$(pwd)/services/airflow"

# Start Airflow in background
echo "Starting Airflow..."

mkdir -p "$AIRFLOW_HOME/logs"

nohup airflow standalone \
    > "$AIRFLOW_HOME/airflow.log" 2>&1 &

AIRFLOW_PID=$!

echo "Airflow PID: $AIRFLOW_PID"

# Wait for Airflow to start
echo "Waiting for Airflow..."

for i in {1..30}; do

    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        echo "Airflow is ready!"
        break
    fi

    sleep 2

done

# Trigger pipeline immediately
echo "Starting wine pipeline..."

airflow dags trigger wine_quality_pipeline

echo ""
echo "========================================="
echo "Wine Quality MLOps is running!"
echo "========================================="
echo ""
echo "Airflow:   http://localhost:8080"
echo "Frontend:  http://localhost:8501"
echo "FastAPI:   http://localhost:8000"
echo ""
echo "Pipeline will continue running every 5 minutes."
echo ""