# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- PAGE SETUP ---
st.set_page_config(page_title="REE Provenance Predictor", layout="centered")
st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1>🧠 Authentica.ai</h1>
        <h4 style='color: #6c757d;'>Adding a Layer of Trust</h4>
        <hr style='margin-top: 20px; margin-bottom: 30px;'>
    </div>
""", unsafe_allow_html=True)

# --- UPLOAD CSV ---
uploaded_file = st.file_uploader("📤 Upload your REE isotopic CSV file", type="csv")

# --- LOAD SAMPLE CSV ---
sample_url = "https://raw.githubusercontent.com/cherisedunn/REE-demo/main/Demo_REE_Expanded_Global.csv"
st.markdown("[🧾 Download Sample CSV](https://raw.githubusercontent.com/cherisedunn/REE-demo/main/Demo_REE_Expanded_Global.csv)")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    if 'Region' in df.columns:
        # --- Supervised Learning ---
        X = df.drop('Region', axis=1)
        y = df['Region']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        st.success(f"✅ Model trained. Accuracy on test set: {accuracy * 100:.2f}%")

        predictions = model.predict(X)
        proba = model.predict_proba(X)
        confidence = np.max(proba, axis=1) * 100

        df['Predicted_Region'] = predictions
        df['Prediction_Confidence (%)'] = confidence

        st.subheader("🔍 Prediction Summary")
        for i in range(min(5, len(df))):
            st.write(f"Sample {i+1}: **{confidence[i]:.1f}%** confidence this sample is from **{predictions[i]}**")

        st.subheader("📊 Prediction Confidence by Sample")
        fig, ax = plt.subplots()
        sns.barplot(x=confidence, y=df.index, hue=predictions, dodge=False, ax=ax)
        ax.set_xlabel("Confidence (%)")
        ax.set_ylabel("Sample Index")
        st.pyplot(fig)

        st.download_button("💾 Download Results as CSV", df.to_csv(index=False), file_name="REE_Predictions.csv")
    
    else:
        st.warning("⚠️ No 'Region' column found. Only prediction mode is enabled using demo training data.")

        # --- Load dummy training data from GitHub ---
        train_data = pd.read_csv(sample_url)
        X_train = train_data.drop('Region', axis=1)
        y_train = train_data['Region']

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(df)
        proba = model.predict_proba(df)
        confidence = np.max(proba, axis=1) * 100

        df['Predicted_Region'] = predictions
        df['Prediction_Confidence (%)'] = confidence

        st.subheader("🔍 Prediction Summary")
        for i in range(min(5, len(df))):
            st.write(f"Sample {i+1}: **{confidence[i]:.1f}%** confidence this sample is from **{predictions[i]}**")

        st.subheader("📊 Prediction Confidence by Sample")
        fig, ax = plt.subplots()
        sns.barplot(x=confidence, y=df.index, hue=predictions, dodge=False, ax=ax)
        ax.set_xlabel("Confidence (%)")
        ax.set_ylabel("Sample Index")
        st.pyplot(fig)

        st.download_button("💾 Download Results as CSV", df.to_csv(index=False), file_name="REE_Predictions.csv")
