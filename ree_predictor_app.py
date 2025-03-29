
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
st.set_page_config(page_title="Authentica.ai REE Predictor", layout="centered")
st.image("https://raw.githubusercontent.com/cherisedunn/REE-demo/main/Authentica_Clean_Logo.png", use_column_width=True)
st.title("🔬 Rare Earth Element (REE) Origin Predictor")
st.markdown("Upload your isotopic data to predict the likely region of origin using a trained Random Forest model.")

uploaded_file = st.file_uploader("Upload your REE CSV file", type="csv")
st.write("Sample count per region:")
st.write(y.value_counts())

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data Preview")
    st.write(df.head())

if 'Region' in df.columns:
    # Split data for training
    X = df.drop('Region', axis=1)
    y = df['Region']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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
st.subheader("🔮 Predict on Full Dataset")
predictions = model.predict(X)

# Get prediction probabilities (confidence)
proba = model.predict_proba(X)
confidence_scores = np.max(proba, axis=1) * 100  # Max probability for each prediction

# Add predictions and confidence to DataFrame
df['Predicted_Region'] = predictions
df['Prediction_Confidence (%)'] = confidence_scores

# Show dataframe with predictions
st.dataframe(df)

# Plot confidence scores
st.subheader("📊 Prediction Confidence by Sample")
fig_conf, ax_conf = plt.subplots()
sns.barplot(y=df.index, x=df['Prediction_Confidence (%)'], hue=df['Predicted_Region'], dodge=False, ax=ax_conf)
ax_conf.set_xlabel("Confidence (%)")
ax_conf.set_ylabel("Sample Index")
st.pyplot(fig_conf)
# Show dataframe with predictions
st.dataframe(df)

# Plot confidence scores
st.subheader("📊 Prediction Confidence by Sample")
fig_conf, ax_conf = plt.subplots()
sns.barplot(y=df.index, x=df['Prediction_Confidence (%)'], hue=df['Predicted_Region'], dodge=False, ax=ax_conf)
ax_conf.set_xlabel("Confidence (%)")
ax_conf.set_ylabel("Sample Index")
st.pyplot(fig_conf)

    fig_conf, ax_conf = plt.subplots()
    sns.barplot(y=df.index, x=df['Prediction_Confidence (%)'], hue=df['Predicted_Region'], dodge=False, ax=ax_conf)
    ax_conf.set_xlabel("Confidence (%)")
    ax_conf.set_ylabel("Sample Index")
    st.pyplot(fig_conf)
        
    proba = model.predict_proba(X)
    confidence_scores = np.max(proba, axis=1) * 100  # Max probability for each prediction
    df['Prediction_Confidence (%)'] = confidence_scores
                
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
