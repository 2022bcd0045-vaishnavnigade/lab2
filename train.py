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

# Feature selection using correlation
corr = data.corr()["quality"].abs()
selected_features = corr[corr > 0.1].index.drop("quality")

X = data[selected_features]
y = data["quality"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
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
    "experiment": "EXP-04",
    "model": "Random Forest",
    "n_estimators": 100,
    "max_depth": 15,
    "features": "Selected (correlation-based)",
    "preprocessing": "None",
    "MSE": mse,
    "R2_Score": r2
}

with open("output/evaluation/results.json", "w") as f:
    json.dump(results, f, indent=4)
