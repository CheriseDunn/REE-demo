# file: app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------
# Page configuration
# ---------------------
st.set_page_config(page_title="REE Provenance Predictor")

st.title("🧠 Authentica.ai")
st.subheader("Upload REE isotope data to predict sample origin")

# ---------------------
# File upload
# ---------------------
uploaded_file = st.file_uploader("📤 Upload your CSV file", type="csv")

# Sample CSV for demo/training
sample_url = "https://raw.githubusercontent.com/cherisedunn/REE-demo/main/Demo_REE_Expanded_Global.csv"

# ---------------------
# Once a file is uploaded
# ---------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("✅ File uploaded successfully!")
    st.write(df.head())

    # Check if 'Region' column exists (for training)
    if 'Region' in df.columns:
        X = df.drop('Region', axis=1)
        y = df['Region']

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        st.success(f"✅ Model trained. Test accuracy: {accuracy*100:.2f}%")

        # Predict on full dataset
        predictions = model.predict(X)
        confidence = np.max(model.predict_proba(X), axis=1) * 100

        df['Predicted Region'] = predictions
        df['Confidence (%)'] = confidence

        st.subheader("🔍 Top 5 Predictions")
        for i in range(min(5, len(df))):
            st.write(f"Sample {i+1}: **{confidence[i]:.1f}%** confidence it’s from **{predictions[i]}**")

        # Confidence bar chart
        st.subheader("📊 Prediction Confidence by Sample")
        fig, ax = plt.subplots()
        sns.barplot(x=confidence, y=df.index, hue=predictions, dodge=False, ax=ax)
        ax.set_xlabel("Confidence (%)")
        ax.set_ylabel("Sample Index")
        st.pyplot(fig)

        # Download button
        st.download_button("💾 Download Results as CSV", df.to_csv(index=False), file_name="REE_predictions.csv")

    else:
        st.warning("⚠️ No 'Region' column found in your file. Using demo training data instead.")

        # Load sample training data
        demo_data = pd.read_csv(sample_url)
        X_train = demo_data.drop('Region', axis=1)
        y_train = demo_data['Region']

        X_pred = df.copy()

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict uploaded samples
        predictions = model.predict(X_pred)
        confidence = np.max(model.predict_proba(X_pred), axis=1) * 100

        df['Predicted Region'] = predictions
        df['Confidence (%)'] = confidence

        st.subheader("🔍 Top 5 Predictions")
        for i in range(min(5, len(df))):
            st.write(f"Sample {i+1}: **{confidence[i]:.1f}%** confidence it’s from **{predictions[i]}**")

        st.subheader("📊 Prediction Confidence by Sample")
        fig, ax = plt.subplots()
        sns.barplot(x=confidence, y=df.index, hue=predictions, dodge=False, ax=ax)
        ax.set_xlabel("Confidence (%)")
        ax.set_ylabel("Sample Index")
        st.pyplot(fig)

        st.download_button("💾 Download Predictions", df.to_csv(index=False), file_name="REE_predictions.csv")

else:
    st.info("Please upload a CSV file to get started.")
    st.markdown("[📥 Download a sample file to test](https://raw.githubusercontent.com/cherisedunn/REE-demo/main/Demo_REE_Expanded_Global.csv)")
