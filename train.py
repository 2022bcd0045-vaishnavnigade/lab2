import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

os.makedirs("output/model", exist_ok=True)
os.makedirs("output/evaluation", exist_ok=True)

# Load data
data = pd.read_csv("dataset/winequality-red.csv", sep=";")

X = data.drop("quality", axis=1)
y = data["quality"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest
model = RandomForestRegressor(
    n_estimators=50,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R2 Score: {r2}")

joblib.dump(model, "output/model/model.pkl")

results = {
    "experiment": "EXP-03",
    "model": "Random Forest",
    "n_estimators": 50,
    "max_depth": 10,
    "features": "All",
    "preprocessing": "None",
    "MSE": mse,
    "R2_Score": r2
}

with open("output/evaluation/results.json", "w") as f:
    json.dump(results, f, indent=4)
