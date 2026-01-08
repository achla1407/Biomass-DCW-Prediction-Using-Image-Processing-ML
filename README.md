

# Dry Cell Weight (DCW) Prediction using Image Processing + Machine Learning

##  Project Overview

This project predicts **Dry Cell Weight (DCW, g/L)** of *Streptomyces hygroscopicus* fermentation broth **from images**, eliminating the need for centrifugation and drying.

The pipeline:

1. Capture broth images across fermentation days
2. Measure experimental DCW for those samples
3. Extract meaningful visual features using OpenCV
4. Train a regression model
5. Deploy a Streamlit app to predict DCW from new images

---

##  Motive

Biomass estimation traditionally requires:

* Centrifugation
* Filtration / drying
* Weighing
* Time-consuming and equipment-heavy steps

This prototype replaces that with:
 Mobile photo +  ML model
→ **Instant DCW prediction**


## Project Structure

```
Biomass_DCW/
│
├── images/                     # Collected fermentation images
│   ├── 1.2/                    # Substrate concentration (g/L)
│   │   ├── day0/
│   │   ├── day3/
│   │   ├── day6/
│   ├── 2.2/
│   ├── 3.2/
│
├── dcw_values.csv              # Lab-measured DCW values
├── extract_features.py         # Image processing + feature engineering
├── build_dataset.py            # Converts images → feature rows
├── final_dataset.csv           # Full features + DCW target
├── train_model.py              # Train/test split + RF training
├── dcw_random_forest_model.pkl # Saved trained model
└── app.py                      # Streamlit app for predictions
```

##  Data Collection

* Images were captured from broth culture flasks at:

  ```
  Day 0, Day 3, Day 6
  ```
* Three substrate concentrations tested:

  ```
  1.2%, 2.2%, 3.2%
  ```
* DCW measured manually using:

  * Sampling
  * Centrifugation
  * Drying
  * Gravimetric estimation

The image and DCW data are linked sample-by-sample to build the training set.

---

##  Feature Engineering (extract_features.py)

Each image is processed to compute:

* Lightness mean & variation
* Pellet mask coverage
* Pellet count and size distribution
* Biomass sediment height
* Texture sharpness (Laplacian variance)
* Dark region fractions
* Concentration and time metadata

Features are exported to a **final_dataset.csv**.

---

## Model Training (train_model.py)

* Algorithm: **Random Forest Regression**
* Inputs: Image-derived numerical features
* Target: DCW in g/L
* Model saved as `dcw_random_forest_model.pkl`

Performance Example:

```
MAE   ~ 0.23 g/L (with DAY)
MAE   ~ 1.10 g/L (image-only)
R²    ~ 0.95 → 0.49
```

---

##  Streamlit App (app.py)

Run:

```
streamlit run app.py
```

Workflow:

```
Upload image →
Features extracted →
Model predicts DCW →
DCW is displayed (g/L)
```

App also:

* Shows extracted features per image
* Lets user select **day & concentration**
* Supports repeated predictions

---

## Advantages

* Fast & non-invasive
* No wet-lab steps required after training
* Can be deployed to any laptop
* Works with standard mobile/lab images
* Scalable to future experiments

---

##  Future Improvements

* Add more training images (all days)
* Control lighting/background
* Try Gradient Boosting / XGBoost
* Build per-day prediction models
* Consider CNNs with larger dataset

---

##  Conclusion

This **prototype** demonstrates that **visual biomass indicators** strongly correlate with DCW, proving that **digital fermentation monitoring** can replace traditional lab measurement for many cases.
Also I am working on the accuracy and also the dataset in more such batches , this is just the prototype of the idea / project

---

##  Credits

**Project & Experiments:** Achala Pandey 
**Guidance :** Dr. Rupika Sinha 

---


