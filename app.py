# ============================================================
# STREAMLIT APP: KNN Robot Surface Classifier (diperbaiki)
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

st.title("Robot Surface Classification (KNN)")

# ==========================
# 1️⃣ Upload CSV
# ==========================
uploaded_file = st.file_uploader(
    "Upload CSV file (65 fitur getaran, optional kolom 'label')", 
    type="csv"
)

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Preview data:")
    st.dataframe(data.head())

    # ==========================
    # 2️⃣ Pisahkan fitur & label
    # ==========================
    if "label" in data.columns:
        X_raw = data.drop(columns=["label"]).values
        y_true = data["label"].values
    else:
        X_raw = data.values
        y_true = None

    # ==========================
    # 3️⃣ Scaling
    # ==========================
    scaler = joblib.load("scaler_standard.pkl")
    X_scaled = scaler.transform(X_raw)

    # ==========================
    # 4️⃣ PCA
    # ==========================
    pca = joblib.load("pca_model.pkl")
    X_pca = pca.transform(X_scaled)

    # ==========================
    # 5️⃣ Load model KNN
    # ==========================
    knn_model = joblib.load("KNN_final_model.pkl")
    y_pred = knn_model.predict(X_pca)
    st.write("Predicted Surface Labels:")
    st.write(y_pred)

    # ==========================
    # 6️⃣ Visualisasi confusion matrix jika ada label
    # ==========================
    if y_true is not None:
        cm = confusion_matrix(y_true, y_pred)
        st.write("Confusion Matrix:")
        st.dataframe(cm)

        # Heatmap
        fig, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=np.unique(y_true),
            yticklabels=np.unique(y_true)
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")
        plt.title("Confusion Matrix Heatmap")
        st.pyplot(fig)

        # Classification report
        cr = classification_report(y_true, y_pred)
        st.text("Classification Report:\n" + cr)

st.info("Pipeline: StandardScaler → PCA → KNN (cosine, weights='distance')")
