
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
st.set_page_config(page_title="Authentica.ai REE Predictor", layout="centered")
st.image("https://raw.githubusercontent.com/cherisedunn/REE-demo/main/Authentica_Clean_Logo.png", use_column_width=True)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
st.title("🔬 Rare Earth Element (REE) Origin Predictor")
st.markdown("Upload your isotopic data to predict the likely region of origin using a trained Random Forest model.")

uploaded_file = st.file_uploader("Upload your REE CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data Preview")
    st.write(df.head())

    if 'Region' in df.columns:
        # Split data for training
        X = df.drop('Region', axis=1)
        y = df['Region']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        accuracy = model.score(X_test, y_test)
        st.success(f"✅ Model trained! Accuracy on test set: {accuracy*100:.2f}%")

        # Visualize
        st.subheader("📊 Isotope Feature Visualization")
        fig1, ax1 = plt.subplots()
        sns.scatterplot(data=df, x="Sr87_Sr86", y="Pb206_Pb204", hue="Region", ax=ax1)
        st.pyplot(fig1)

        fig2 = sns.pairplot(df, hue="Region")
        st.pyplot(fig2)

        # Make predictions
        st.subheader("🔎 Predict on Full Dataset")
        predictions = model.predict(X)
        df['Predicted_Region'] = predictions
        st.dataframe(df)
    else:
        st.warning("⚠️ No 'Region' column found. Assuming prediction-only mode.")

        # Inference-only mode
        model = RandomForestClassifier(n_estimators=100, random_state=42)

        # Load dummy training data
        dummy_data = pd.read_csv("Demo_REE_Expanded_Sample.csv")
        X_train = dummy_data.drop('Region', axis=1)
        y_train = dummy_data['Region']
        model.fit(X_train, y_train)

        predictions = model.predict(df)
        df['Predicted_Region'] = predictions
        st.success("🎉 Predictions complete!")
        st.dataframe(df)

        st.download_button("Download Results as CSV", df.to_csv(index=False), file_name="Predicted_REE_Origins.csv")
