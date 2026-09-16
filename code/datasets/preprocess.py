from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

root = Path(__file__).resolve().parents[2]

raw_path = root / "data/raw/winequality-white.csv"
processed_dir = root / "data/processed"

train_path = root / "data/processed/train.csv"
test_path = root / "data/processed/test.csv"


#1. Load data

#The wine dataset uses ";" as the separator
df = pd.read_csv(raw_path, sep=";")
print("Original data shape:", df.shape)


#2. Remove missing values

#Remove rows that contain missing values
df = df.dropna()
print("After removing missing values:", df.shape)
#No data removed so kinda useless

#3. Remove outliers

#I used the IQR method
#A datapoint is outlier is:
#below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR

#Get all feature columns.
#"quality" is the target, so we do NOT use it for outlier removal.
feature_columns = df.columns.drop("quality")

#Start with all rows considered valid
valid_rows = pd.Series(True, index=df.index)

for column in feature_columns:
    #First quartile (25%)
    q1 = df[column].quantile(0.25)

    #Third quartile (75%)
    q3 = df[column].quantile(0.75)

    #Interquartile range
    iqr = q3 - q1

    #Allowed value range
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    #Keep only rows inside the allowed range
    valid_rows &= (
            (df[column] >= lower_bound)
            & (df[column] <= upper_bound)
    )

#Remove rows containing outliers
df = df[valid_rows].reset_index(drop=True)

print("After removing outliers:", df.shape)


#4. Split into train and test

#Create the directory if it does not exist
train_path.parent.mkdir(parents=True, exist_ok=True)

#Separate features and target
X = df.drop(columns=["quality"])
y = df["quality"]

#80% -> training data
#20% -> testing data
#
#stratify=y keeps approximately the same distribution
#of wine quality values in both datasets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


#Put features and target back together
train_df = X_train.copy()
train_df["quality"] = y_train

test_df = X_test.copy()
test_df["quality"] = y_test


#5. Save processed data

#Save training dataset
train_df.to_csv(train_path, index=False)

#Save testing dataset
test_df.to_csv(test_path, index=False)

print("Train data saved to:", train_path)
print("Test data saved to:", test_path)
print("Data processing completed!")