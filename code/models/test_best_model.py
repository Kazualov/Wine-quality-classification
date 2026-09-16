from pathlib import Path

import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_validate


root = Path(__file__).resolve().parents[2]

train_path = root / "data/processed/train.csv"


#1. Load data

train_df = pd.read_csv(train_path)

print("Train data shape:", train_df.shape)


#2. Separate features and target

X = train_df.drop(columns=["quality"])
y = train_df["quality"]


#3. Create the final Extra Trees model

model = ExtraTreesRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=1,
    min_samples_split=2,
    max_features=0.7,
    random_state=42,
    n_jobs=-1
)


#4. Create 5-fold cross validation

# The dataset is split into 5 parts.
# The model is trained on 4 parts and validated on 1 part.
# This process is repeated 5 times.
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


#5. Evaluate the model

scores = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring="r2",
    return_train_score=True
)


#6. Get train and validation scores

train_scores = scores["train_score"]
validation_scores = scores["test_score"]


#7. Print results for every fold

print("\n")
print("=" * 60)
print("CROSS VALIDATION RESULTS")
print("=" * 60)

for i in range(5):
    print(
        f"Fold {i + 1}: "
        f"Train R2 = {train_scores[i]:.4f}, "
        f"Validation R2 = {validation_scores[i]:.4f}"
    )


#8. Calculate average scores

mean_train_r2 = train_scores.mean()
mean_validation_r2 = validation_scores.mean()

train_std = train_scores.std()
validation_std = validation_scores.std()


#9. Calculate overfitting gap

overfit_gap = mean_train_r2 - mean_validation_r2


#10. Print final results

print("\n")
print("=" * 60)
print("FINAL EXTRA TREES CHECK")
print("=" * 60)

print(f"Mean Train R2:       {mean_train_r2:.4f}")
print(f"Mean Validation R2:  {mean_validation_r2:.4f}")

print(f"Train R2 Std:        {train_std:.4f}")
print(f"Validation R2 Std:   {validation_std:.4f}")

print(f"Overfitting Gap:     {overfit_gap:.4f}")