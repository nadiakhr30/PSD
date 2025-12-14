# ============================================================
# STREAMLIT APP: KNN Robot Surface Classifier
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

# ============================
# 1️⃣ Upload data
# ============================
uploaded_file = st.file_uploader("Upload CSV file (65 columns = vibration points)", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Preview data:")
    st.dataframe(data.head())

    # ============================
    # 2️⃣ Preprocessing
    # ============================
    X_raw = data.values  # anggap semua kolom adalah fitur
    scaler = joblib.load("scaler_standard.pkl") if "scaler_standard.pkl" in st.session_state else StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    
    # ============================
    # 4️⃣ PCA transform
    # ============================
    pca = joblib.load("pca_model.pkl") if "pca_model.pkl" in st.session_state else PCA(n_components=20, random_state=42)
    X_pca = pca.transform(X_scaled)
    
    # ============================
    # 5️⃣ Load KNN model
    # ============================
    knn_model = joblib.load("KNN_final_model.pkl") if "KNN_final_model.pkl" in st.session_state else \
        KNeighborsClassifier(n_neighbors=3, metric='cosine', weights='distance')
    
    # ============================
    # 6️⃣ Predict
    # ============================
    y_pred = knn_model.predict(X_pca)
    st.write("Predicted Surface Labels:")
    st.write(y_pred)

    # ============================
    # 7️⃣ Visualisasi: Confusion Matrix (jika ada label aktual)
    # ============================
    if "label" in data.columns:  # optional jika ada kolom label
        y_true = data["label"].values
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

        # Classification Report
        cr = classification_report(y_true, y_pred)
        st.text("Classification Report:\n" + cr)

st.info("Pipeline: SMOTE → StandardScaler → PCA → KNN (cosine, weights='distance')")
