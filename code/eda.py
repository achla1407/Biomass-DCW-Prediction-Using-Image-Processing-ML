import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# LOAD DATA
# ---------------------------
df = pd.read_csv("final_dataset.csv")
print("Dataset shape:", df.shape)

# ---------------------------
# ROUND NUMERIC FEATURES
# ---------------------------
numeric_cols = df.select_dtypes(include="number").columns
df[numeric_cols] = df[numeric_cols].round(3)

# ---------------------------
# TARGET DISTRIBUTION
# ---------------------------
df = df.drop(columns=["image_name","blob_area_fraction"])

plt.figure(figsize=(5,4))
plt.hist(df["dcw_g_per_l"], bins=10, edgecolor="black")
plt.xlabel("DCW (g/L)")
plt.ylabel("Frequency")
plt.title("DCW Distribution")
plt.show()

# ---------------------------
# CORRELATION WITH TARGET
# ---------------------------
corr = df.corr(numeric_only=True)["dcw_g_per_l"].sort_values(ascending=False)
print("\nCorrelation with DCW:\n")
print(corr)

# ---------------------------
# CORRELATION HEATMAP (MANUAL)
# ---------------------------
plt.figure(figsize=(10,6))
plt.imshow(df.corr(numeric_only=True), cmap="coolwarm")
plt.colorbar()
plt.xticks(range(len(df.columns)), df.columns, rotation=90)
plt.yticks(range(len(df.columns)), df.columns)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()

# ---------------------------
# SAVE CLEAN DATASET
df.to_csv("final_dataset_clean1.csv", index=False)

print("✅ Clean dataset saved")
print(df.head())

