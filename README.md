# Wine Quality MLOps

## 1. Install dependencies

Create the main virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

---

## 2. Configure DagsHub

If you want to run training without MLFlow logging set MLFLOW_ENABLED to False

```python
MLFLOW_ENABLED = False
```
If so, you can skip this step, if not - set it to True and follow the instructions

Create `.env` in the project root:

```env
MLFLOW_TRACKING_URI=your_dagshub_mlflow_uri
MLFLOW_TRACKING_USERNAME=your_dagshub_username
MLFLOW_TRACKING_PASSWORD=your_dagshub_token
```

---

## 3. Setup Airflow

Airflow uses a separate virtual environment.

Create it:

```bash
python3.12 -m venv .airflow_venv
```

Activate it:

```bash
source .airflow_venv/bin/activate
```

Install Airflow:

```bash
AIRFLOW_VERSION=3.3.1
PYTHON_VERSION=3.12
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

pip install "apache-airflow==${AIRFLOW_VERSION}" \
  --constraint "${CONSTRAINT_URL}"
```

Set the Airflow home directory:

```bash
export AIRFLOW_HOME="$(pwd)/services/airflow"
```

Initialize and start Airflow:

```bash
airflow standalone
```

Open:

```text
http://localhost:8080
```

The DAG is:

```text
wine_quality_pipeline
```

The pipeline runs every 5 minutes:

```text
Data Engineering
       ↓
Model Engineering
       ↓
Deployment
```

The Airflow virtual environment is not committed to Git.

---

## 4. Manual pipeline

### Data Engineering

```bash
python code/datasets/preprocess.py
```

Creates:

```text
data/processed/train.csv
data/processed/test.csv
```

### Model Engineering

Open:

```text
code/models/train.py
```

---

## 5. Deployment

From the project root:

```bash
cd code/deployment
docker compose up --build
```

Open:

```text
http://localhost:8501
```

FastAPI:

```text
http://localhost:8000
```

FastAPI documentation:

```text
http://localhost:8000/docs
```

Stop the containers:

```bash
docker compose down
```

---

## 6. Start everything

After Airflow has been installed, use:

```bash
./start.sh
```

`start.sh`:

- starts Airflow;
- triggers the pipeline immediately;
- runs Data Engineering;
- runs Model Engineering;
- starts FastAPI and Streamlit;
- keeps the Airflow schedule running every 5 minutes.

Open:

```text
Airflow:  http://localhost:8080
Frontend: http://localhost:8501
FastAPI:  http://localhost:8000
```

---

## 7. Stop everything

```bash
./stop.sh
```

This stops:

- FastAPI;
- Streamlit;
- Airflow.

---

## 8. Git

Do not commit:

```text
.venv/
.airflow_venv/
.env
__pycache__/
services/airflow/logs/
services/airflow/*.db
```

Commit:

```text
start.sh
stop.sh
services/airflow/dags/wine_pipeline.py
code/
data/
models/
requirements.txt
README.md
```

---

## Full launch

```bash
source .venv/bin/activate
./start.sh
```

Then open:

```text
http://localhost:8501
```

The complete pipeline is orchestrated by Airflow and runs every 5 minutes.
