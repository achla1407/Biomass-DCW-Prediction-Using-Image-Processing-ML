import os
import pandas as pd
from extract_features import extract_features

# -----------------------------------------
# PATHS
# -----------------------------------------

IMAGE_ROOT = "images"
DCW_CSV = "dcw_values.csv"
OUTPUT_CSV = "final_dataset.csv"

# -----------------------------------------
# LOAD DCW VALUES
# -----------------------------------------

dcw_df = pd.read_csv(DCW_CSV)

# Create fast lookup dictionary
# key = (concentration, day)
dcw_lookup = {
    (str(row["concentration"]), int(row["day"])): row["dcw_g_per_l"]
    for _, row in dcw_df.iterrows()
}

# -----------------------------------------
# BUILD DATASET
# -----------------------------------------

records = []

for concentration in os.listdir(IMAGE_ROOT):
    conc_path = os.path.join(IMAGE_ROOT, concentration)

    if not os.path.isdir(conc_path):
        continue

    for day_folder in os.listdir(conc_path):
        if not day_folder.startswith("day"):
            continue

        day = int(day_folder.replace("day", ""))
        day_path = os.path.join(conc_path, day_folder)

        # Fetch DCW value
        key = (concentration, day)
        if key not in dcw_lookup:
            print(f"⚠️ DCW missing for {key}, skipping")
            continue

        dcw_value = dcw_lookup[key]

        for img_name in os.listdir(day_path):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(day_path, img_name)

            try:
                features = extract_features(img_path)

                record = {
                    "concentration": float(concentration),
                    "day": day,
                    "image_name": img_name,
                    "dcw_g_per_l": dcw_value
                }

                record.update(features)
                records.append(record)

            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")

# -----------------------------------------
# SAVE FINAL DATASET
# -----------------------------------------

dataset_df = pd.DataFrame(records)
dataset_df.to_csv(OUTPUT_CSV, index=False)

print("✅ Dataset built successfully!")
print(f"📄 Saved as: {OUTPUT_CSV}")
print(f"🔢 Total samples: {len(dataset_df)}")
