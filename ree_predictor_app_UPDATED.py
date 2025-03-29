
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

st.set_page_config(page_title="Authentica.ai REE Predictor", layout="centered")
st.image("https://raw.githubusercontent.com/cherisedunn/REE-demo/main/Authentica_Clean_Logo.png", use_column_width=True)
st.title("Rare Earth Element (REE) Origin Predictor")
st.write("Upload your isotopic data to predict the likely region of origin using a trained Random Forest model.")

uploaded_file = st.file_uploader("Upload your REE CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Preview of Uploaded Data")
    st.write(df.head())

    if 'Region' in df.columns:
        st.subheader("📊 Visualize Isotopic Ratios")
        sns.pairplot(df, hue='Region')
        st.pyplot()

        X = df.drop("Region", axis=1)
        y = df["Region"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader("📈 Classification Report")
        st.text(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        st.pyplot()

        # 🎯 Confidence % plot
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)
            classes = model.classes_
            prob_df = pd.DataFrame(probs, columns=classes, index=[f"Sample {i+1}" for i in range(len(probs))])

            st.subheader("🔬 Model Confidence per Sample")
            st.write("Each bar shows how confident the model is about the sample's origin.")

            fig, ax = plt.subplots(figsize=(10, 6))
            bottoms = np.zeros(len(prob_df))
            for region in classes:
                ax.bar(prob_df.index, prob_df[region], label=region, bottom=bottoms)
                bottoms += prob_df[region]

            ax.set_ylabel("Confidence (%)")
            ax.set_title("Predicted Region Confidence by Sample")
            ax.legend(title="Predicted Region")
            plt.xticks(rotation=45)
            st.pyplot(fig)

    else:
        st.subheader("🧠 Predicting Region of Unknown Samples")
        if 'Sr87_Sr86' in df.columns:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            sample_data = pd.read_csv("https://raw.githubusercontent.com/cherisedunn/REE-demo/main/REE_Sample_With_Region.csv")
            X_train = sample_data.drop("Region", axis=1)
            y_train = sample_data["Region"]
            model.fit(X_train, y_train)

            predictions = model.predict(df)
            df["Predicted_Region"] = predictions
            st.write(df)

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(df)
                classes = model.classes_
                prob_df = pd.DataFrame(probs, columns=classes, index=[f"Sample {i+1}" for i in range(len(probs))])

                st.subheader("🔬 Model Confidence per Sample")
                st.write("Each bar shows how confident the model is about the sample's origin.")

                fig, ax = plt.subplots(figsize=(10, 6))
                bottoms = np.zeros(len(prob_df))
                for region in classes:
                    ax.bar(prob_df.index, prob_df[region], label=region, bottom=bottoms)
                    bottoms += prob_df[region]

                ax.set_ylabel("Confidence (%)")
                ax.set_title("Predicted Region Confidence by Sample")
                ax.legend(title="Predicted Region")
                plt.xticks(rotation=45)
                st.pyplot(fig)
