import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from dotenv import load_dotenv
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
load_dotenv()

#Settings
#True  -> log model, parameters and metrics to MLflow/DagsHub
#False -> train model and save it locally only
MLFLOW_ENABLED = False


#Imports for MLflow

#Import MLflow only when it is enabled
if MLFLOW_ENABLED:
    import dagshub
    import mlflow
    import mlflow.sklearn


#Paths

root = Path(__file__).resolve().parents[2]

train_path = root / "data/processed/train.csv"
test_path = root / "data/processed/test.csv"

model_dir = root / "models"
model_path = model_dir / "wine_model.joblib"


#1. Load data

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("Train data shape:", train_df.shape)
print("Test data shape:", test_df.shape)


#2. Separate features and target

#"quality" is the target that we want to predict
X_train = train_df.drop(columns=["quality"])
y_train = train_df["quality"]

X_test = test_df.drop(columns=["quality"])
y_test = test_df["quality"]


#3. Create the model

model = ExtraTreesRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=1,
    min_samples_split=2,
    max_features=0.7,
    random_state=42,
    n_jobs=-1
)


#4. Train the model

print("\nTraining Extra Trees...")

model.fit(X_train, y_train)

print("Training completed!")


#5. Make predictions

y_pred = model.predict(X_test)


#6. Calculate test metrics

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)


#7. Print test metrics

print("\n")
print("=" * 60)
print("TEST METRICS")
print("=" * 60)

print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")


#8. Save the trained model locally

#Create models directory if it does not exist
model_dir.mkdir(parents=True, exist_ok=True)

#Save the trained model
joblib.dump(model, model_path)

print("\nModel saved to:", model_path)


#9. Log experiment to MLflow / DagsHub

if MLFLOW_ENABLED:

    print("\nLogging experiment to DagsHub MLflow...")

    # Set DagsHub MLflow tracking server
    mlflow.set_tracking_uri(
        os.environ["MLFLOW_TRACKING_URI"]
    )

    # Use DagsHub credentials
    os.environ["MLFLOW_TRACKING_USERNAME"] = (
        os.environ["MLFLOW_TRACKING_USERNAME"]
    )

    os.environ["MLFLOW_TRACKING_PASSWORD"] = (
        os.environ["MLFLOW_TRACKING_PASSWORD"]
    )

    mlflow.set_experiment("wine-quality")

    with mlflow.start_run():

        mlflow.log_param("model", "ExtraTreesRegressor")
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("max_depth", 20)
        mlflow.log_param("min_samples_leaf", 1)
        mlflow.log_param("min_samples_split", 2)
        mlflow.log_param("max_features", 0.7)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(
            model,
            name="wine_model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

    print("MLflow logging completed!")

else:

    print("\nMLflow logging is disabled.")
    print("Model was saved locally only.")