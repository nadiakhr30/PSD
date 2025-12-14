# ============================================================
# STREAMLIT APP: KNN Robot Surface Classifier (2 Label, Modern)
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ----------------------------
# 1️⃣ Page Config
# ----------------------------
st.set_page_config(
    page_title="Robot Surface Classifier",
    page_icon="🤖",
    layout="wide"
)

# ----------------------------
# 2️⃣ Title & Intro
# ----------------------------
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>Robot Surface Classification (KNN)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Deteksi permukaan robot: Licin vs Kasar</p>", unsafe_allow_html=True)
st.info("Pipeline: StandardScaler → PCA → KNN (cosine, weights='distance')")

st.markdown("---")
st.subheader("📁 Upload CSV File")
st.markdown("File CSV harus memiliki **65 fitur getaran**. Kolom `label` opsional untuk evaluasi.")

# ----------------------------
# 3️⃣ Upload CSV
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload CSV file here", 
    type="csv"
)

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Preview data:")
    st.dataframe(data.head())

    # ----------------------------
    # 4️⃣ Pisahkan fitur & label
    # ----------------------------
    if "label" in data.columns:
        X_raw = data.drop(columns=["label"]).values
        y_true = data["label"].values
    else:
        X_raw = data.values
        y_true = None

    # ----------------------------
    # 5️⃣ Scaling
    # ----------------------------
    scaler = joblib.load("scaler_standard.pkl")
    X_scaled = scaler.transform(X_raw)

    # ----------------------------
    # 6️⃣ PCA
    # ----------------------------
    pca = joblib.load("pca_model.pkl")
    X_pca = pca.transform(X_scaled)

    # ----------------------------
    # 7️⃣ Load KNN model
    # ----------------------------
    knn_model = joblib.load("KNN_final_model.pkl")
    y_pred = knn_model.predict(X_pca)

    # ----------------------------
    # 8️⃣ Mapping label ke nama permukaan (AMAN)
    # ----------------------------
    label_mapping = {0: "Permukaan Licin", 1: "Permukaan Kasar"}
    y_pred_labels = [label_mapping.get(int(i), "Unknown") for i in y_pred]

    if y_true is not None:
        y_true_labels = [label_mapping.get(int(i), "Unknown") for i in y_true]

    # ----------------------------
    # 9️⃣ Tampilkan prediksi di tabel interaktif
    # ----------------------------
    data["Predicted Surface"] = y_pred_labels
    st.subheader("📊 Tabel Prediksi Permukaan")
    st.dataframe(
        data.style.apply(
            lambda x: ['background-color: lightgreen' if v=="Permukaan Licin" 
                       else 'background-color: lightcoral' 
                       for v in x["Predicted Surface"]], axis=1
        )
    )

    # ----------------------------
    # 🔟 Visualisasi distribusi prediksi
    # ----------------------------
    st.subheader("📈 Distribusi Prediksi Permukaan")
    pred_count = pd.Series(y_pred_labels).value_counts()

    # Dua kolom: bar chart & pie chart
    col1, col2 = st.columns(2)

    with col1:
        st.bar_chart(pred_count)

    with col2:
        fig, ax = plt.subplots()
        ax.pie(pred_count, labels=pred_count.index, autopct='%1.1f%%', colors=['lightgreen','lightcoral'])
        ax.set_title("Distribusi Prediksi Permukaan")
        st.pyplot(fig)

    # ----------------------------
    # 1️⃣1️⃣ Confusion Matrix & Metrics (jika ada label asli)
    # ----------------------------
    if y_true is not None:
        st.subheader("✅ Evaluasi Model (dengan label asli)")
        cm = confusion_matrix(y_true_labels, y_pred_labels)
        st.write("Confusion Matrix:")
        st.dataframe(cm)

        # Heatmap
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_mapping.values(),
            yticklabels=label_mapping.values()
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")
        plt.title("Confusion Matrix Heatmap")
        st.pyplot(fig)

        # Accuracy
        accuracy = accuracy_score(y_true_labels, y_pred_labels)
        st.write(f"Accuracy: {accuracy*100:.2f}%")

        # Classification report
        cr = classification_report(y_true_labels, y_pred_labels)
        st.text("Classification Report:\n" + cr)

# ----------------------------
# ℹ️ Footer Info
# ----------------------------
st.markdown("---")
st.info("App ini menggunakan model KNN dengan preprocessing StandardScaler → PCA → KNN (cosine, weights='distance').")
