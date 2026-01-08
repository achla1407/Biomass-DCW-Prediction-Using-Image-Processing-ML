import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt 

print("Loading dataset...")

df = pd.read_csv("final_dataset_clean.csv")

# DROP NON-IMAGE LEAKAGE
if "day" in df.columns:
    df = df.drop(columns=["day"])

if "image_name" in df.columns:
    df = df.drop(columns=["image_name"])

X = df.drop(columns=["dcw_g_per_l"])
y = df["dcw_g_per_l"]

print("Features used:")
print(X.columns.tolist())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Random Forest...")

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

print("Evaluating model...")

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("------------------")
print(f"MAE  : {mae:.3f} g/L")
print(f"RMSE : {rmse:.3f} g/L")
print(f"R²   : {r2:.3f}")

cv_scores = cross_val_score(
    model, X, y, cv=5, scoring="neg_mean_absolute_error"
)

print("\nCross-Validation MAE:", -cv_scores)
print("Mean CV MAE:", -cv_scores.mean())

importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\nTop Features Driving DCW:")
print(feature_importance_df)
plt.figure(figsize=(8, 5))
plt.barh(
    feature_importance_df["feature"],
    feature_importance_df["importance"]
)
plt.xlabel("Importance")
plt.title("Feature Importance for DCW Prediction")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

joblib.dump(model, "dcw_random_forest_model.pkl")
print("\nModel saved as dcw_random_forest_model.pkl")
