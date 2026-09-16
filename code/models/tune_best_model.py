from pathlib import Path

import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV


root = Path(__file__).resolve().parents[2]

train_path = root / "data/processed/train.csv"


#1. Load training data

train_df = pd.read_csv(train_path)

print("Train data shape:", train_df.shape)


#2. Separate features and target

X = train_df.drop(columns=["quality"])
y = train_df["quality"]


#3. Create Extra Trees model

model = ExtraTreesRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)


#4. Define hyperparameters

param_grid = {
    "max_depth": [15, 20, 25, 30, None],

    "min_samples_leaf": [1, 2],

    "min_samples_split": [2, 4],

    "max_features": [1.0, "sqrt", 0.7],
}


#5. Run Grid Search with 5-fold Cross Validation

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,

    # R2 is our main metric
    scoring="r2",

    # Use all CPU cores
    n_jobs=-1,

    # Show progress
    verbose=1,

    # Save training scores as well
    return_train_score=True,
)


print("\nStarting Grid Search...")

grid_search.fit(X, y)


#6. Print the best configuration

print("\n")
print("=" * 70)
print("BEST EXTRA TREES CONFIGURATION")
print("=" * 70)

print("Best parameters:")
print(grid_search.best_params_)

print("\nBest CV R2:")
print(grid_search.best_score_)


#7. Show the best configurations

results = pd.DataFrame(grid_search.cv_results_)

results = results[
    [
        "param_max_depth",
        "param_min_samples_leaf",
        "param_min_samples_split",
        "param_max_features",
        "mean_train_score",
        "mean_test_score",
        "std_test_score",
    ]
]

results = results.sort_values(
    by="mean_test_score",
    ascending=False
)

print("\n")
print("=" * 70)
print("TOP 10 CONFIGURATIONS")
print("=" * 70)

print(results.head(10).to_string(index=False))