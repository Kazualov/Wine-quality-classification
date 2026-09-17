# Wine Quality MLOps

## 1. Install dependencies

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Configure DagsHub
If you wish to skip logging - use MLFLOW_ENABLED = False in code/models/train.py
If so - don't configure anything, skip this step

Create `.env` in the project root:

```env
MLFLOW_TRACKING_URI=your_dagshub_mlflow_uri
MLFLOW_TRACKING_USERNAME=your_dagshub_username
MLFLOW_TRACKING_PASSWORD=your_dagshub_token
```

Make sure `.env` is in `.gitignore`.

## 3. Run Data Engineering

From the project root:

```bash
python code/datasets/preprocess.py
```

This creates:

```text
data/processed/train.csv
data/processed/test.csv
```

## 4. Train the model

Open:

```text
code/models/train.py
```

Run:

```bash
python code/models/train.py
```

The trained model will be saved to:

```text
models/wine_model.joblib
```

## 5. Start the application

From the project root:

```bash
cd code/deployment
docker compose up --build
```

Open the web application:

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

## 6. Stop the application

```bash
docker compose down
```

## Full run

```bash
python code/datasets/preprocess.py
python code/models/train.py
cd code/deployment
docker compose up --build
```
