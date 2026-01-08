import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("final_dataset_clean.csv")

print("\nDataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())


# =========================
# 2. BASIC CLEANING
# =========================
# Drop non-feature columns if still present
df = df.drop(columns=["image_name"], errors="ignore")

# Remove rows with missing target
df = df.dropna(subset=["dcw_g_per_l"])

# Round numeric values (stability)
num_cols = df.select_dtypes(include="number").columns
df[num_cols] = df[num_cols].round(3)


# =========================
# 3. FEATURE / TARGET SPLIT
# =========================
TARGET = "dcw_g_per_l"

X = df.drop(columns=[TARGET])
y = df[TARGET]

print("\nNumber of features:", X.shape[1])


# =========================
# 4. TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)


# =========================
# 5. MODEL DEFINITION
# =========================
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)


# =========================
# 6. TRAIN MODEL
# =========================
model.fit(X_train, y_train)


# =========================
# 7. EVALUATION
# =========================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("-----------------")
print(f"MAE  : {mae:.3f} g/L")
print(f"RMSE : {rmse:.3f} g/L")
print(f"R²   : {r2:.3f}")


# =========================
# 8. CROSS VALIDATION (SMALL DATA SAFE)
# =========================
cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="neg_mean_absolute_error"
)

print("\nCross-Validation MAE:", np.abs(cv_scores).round(3))
print("Mean CV MAE:", np.abs(cv_scores.mean()).round(3))


# =========================
# 9. FEATURE IMPORTANCE
# =========================
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\nTop Features Driving DCW:")
print(feature_importance_df)

# Plot
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


# =========================
# 10. SAVE MODEL
# =========================
import joblib
joblib.dump(model, "dcw_random_forest_model.pkl")

print("\n✅ Model saved as dcw_random_forest_model.pkl")
