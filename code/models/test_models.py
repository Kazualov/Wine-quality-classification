from pathlib import Path

import pandas as pd

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


root = Path(__file__).resolve().parents[2]

train_path = root / "data/processed/train.csv"
test_path = root / "data/processed/test.csv"


#1. Load data

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("Train data shape:", train_df.shape)
print("Test data shape:", test_df.shape)


#2. Separate features and target

# "quality" is the value we want to predict
X_train = train_df.drop(columns=["quality"])
y_train = train_df["quality"]

X_test = test_df.drop(columns=["quality"])
y_test = test_df["quality"]


#3. Create models

models = {
    "Linear Regression": LinearRegression(),

    "Ridge": Ridge(
        alpha=1.0
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=200,
        random_state=42
    )
}


#4. Train and evaluate models

results = []

for name, model in models.items():

    print("\n" + "=" * 50)
    print("Model:", name)
    print("=" * 50)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)

    # Save results
    results.append({
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })


#5. Compare models

results_df = pd.DataFrame(results)

# Sort by R2 from highest to lowest
results_df = results_df.sort_values(
    by="R2",
    ascending=False
)

print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(results_df.to_string(index=False))