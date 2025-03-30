import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- PAGE CONFIG ---
st.set_page_config(page_title="REE Regional Classifier", layout="centered")
st.image("Authentica_Logo.png", use_container_width=True)
st.title("🧪 Rare Earth Element (REE) Origin Predictor")
st.markdown("Upload your isotopic data to predict the likely region of origin using a trained Random Forest model.")


# --- SIDEBAR ---
with st.sidebar:
    st.header("ℹ️ About Authentica.ai")
    st.write("AI-powered REE traceability using geochemical fingerprinting.")
    st.markdown("[🌐 Visit our website](https://reedomo.streamlit.app)")
    st.markdown("[📬 Contact us](mailto:hello@authentica.ai)")
    st.markdown("[🧾 Download sample CSV](https://raw.githubusercontent.com/cherisedunn/REE-demo/main/Demo_REE_Expanded_Global.csv)")

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("📤 Upload your REE CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data Preview")
    st.write(df.head())

    if 'Region' in df.columns:
        # --- Train/Test Split ---
        X = df.drop('Region', axis=1)
        y = df['Region']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # --- Train Model ---
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # --- Evaluate Model ---
        accuracy = model.score(X_test, y_test)
        st.success(f"✅ Model trained. Accuracy on test set: {accuracy * 100:.2f}%")

        # --- Predict on Full Data ---
        predictions = model.predict(X)
        proba = model.predict_proba(X)
        confidence_scores = np.max(proba, axis=1) * 100

        df['Predicted_Region'] = predictions
        df['Prediction_Confidence (%)'] = confidence_scores
        st.subheader("🧪 Prediction Results")
        st.dataframe(df)

        # --- Plot Prediction Confidence ---
        st.subheader("📊 Prediction Confidence by Sample")
        fig_conf, ax_conf = plt.subplots()
        sns.barplot(y=df.index, x=df['Prediction_Confidence (%)'], hue=df['Predicted_Region'], dodge=False, ax=ax_conf)
        ax_conf.set_xlabel("Confidence (%)")
        ax_conf.set_ylabel("Sample Index")
        st.pyplot(fig_conf)

        # --- Download Results ---
        st.download_button("💾 Download Predictions as CSV", df.to_csv(index=False), file_name="Predicted_REE_Origins.csv")

    else:
        # --- Inference-Only Mode ---
        st.warning("⚠️ No 'Region' column found. Running in prediction-only mode with dummy training data.")

        # Train dummy model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        dummy_data = pd.read_csv("Demo_REE_Expanded_Global.csv")
        X_train = dummy_data.drop('Region', axis=1)
        y_train = dummy_data['Region']
        model.fit(X_train, y_train)

        predictions = model.predict(df)
        proba = model.predict_proba(df)
        confidence_scores = np.max(proba, axis=1) * 100

        df['Predicted_Region'] = predictions
        df['Prediction_Confidence (%)'] = confidence_scores
        st.success("🎉 Predictions complete!")
        st.dataframe(df)

        # --- Confidence Plot ---
        st.subheader("📊 Prediction Confidence by Sample")
        fig_conf, ax_conf = plt.subplots()
        sns.barplot(y=df.index, x=df['Prediction_Confidence (%)'], hue=df['Predicted_Region'], dodge=False, ax=ax_conf)
        ax_conf.set_xlabel("Confidence (%)")
        ax_conf.set_ylabel("Sample Index")
        st.pyplot(fig_conf)

        # --- Download Button ---
        st.download_button("💾 Download Predictions as CSV", df.to_csv(index=False), file_name="Predicted_REE_Origins.csv")
