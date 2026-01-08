import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tempfile
import os

# Import your feature extraction function
from extract_features import extract_features

# Page config

st.set_page_config(
    page_title="DCW Prediction App",
    page_icon="🧫",
    layout="centered"
)


# Load trained model

@st.cache_resource
def load_model():
    return joblib.load("dcw_random_forest_model.pkl")

model = load_model()


# UI Header

st.markdown(
    """
    <h1 style='text-align: center;'>🧫 Dry Cell Weight (DCW) Predictor</h1>
    <p style='text-align: center; color: gray;'>
    Image-based biomass estimation for <b>S. hygroscopicus</b>
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# Sidebar inputs

st.sidebar.header("🧪 Experimental Parameters")

day = st.sidebar.slider(
    "Incubation Day",
    min_value=0,
    max_value=6,
    value=3,
    step=1,
    help="Day of incubation (used during model training)"
)

concentration = st.sidebar.selectbox(
    "Substrate Concentration",
    options=[1.2, 2.2, 3.2],
    help="Initial substrate concentration"
)


# Image uploader

st.subheader("📷 Upload Culture Image")

uploaded_file = st.file_uploader(
    "Upload flask image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)


# Prediction logic

if uploaded_file is not None:
    # Show image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Save temp image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
      
        # Feature extraction
       
        features = extract_features(tmp_path)

        feature_df = pd.DataFrame([features])

        # Add metadata features
        feature_df["day"] = day
        feature_df["concentration"] = concentration

        # Ensure column order matches training
        model_features = model.feature_names_in_
        feature_df = feature_df[model_features]

       
        # Prediction

        prediction = model.predict(feature_df)[0]

        st.divider()

        st.markdown(
            f"""
            <h2 style='text-align: center;'>🔬 Predicted DCW</h2>
            <h1 style='text-align: center; color: #2E86C1;'>
            {prediction:.2f} g/L
            </h1>
            """,
            unsafe_allow_html=True
        )

        st.success("Prediction completed successfully ✔")

        # show extracted features
        with st.expander("🔍 View extracted features"):
            st.dataframe(feature_df.T, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")

    finally:
        os.remove(tmp_path)

else:
    st.info("⬆ Upload an image to start DCW prediction")

st.divider()

st.caption(
    "⚠ This is a research prototype. Predictions are approximate and depend on imaging conditions."
)

