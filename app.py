# ============================================================
# STREAMLIT APP: KNN Robot Surface Classifier (2 Label, Modern & Rapih)
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
    layout="wide"
)

# ----------------------------
# 2️⃣ Custom CSS untuk font & icon
# ----------------------------
st.markdown("""
<style>
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');

h1 {
    font-family: 'Arial', sans-serif;
    font-size: 38px;
    text-align: center;
    color: #1F77B4;
}
h2 {
    font-family: 'Arial', sans-serif;
    font-size: 24px;
    text-align: center;
    color: #333333;
}
p {
    font-family: 'Arial', sans-serif;
    font-size: 16px;
    text-align: center;
    color: #555555;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# 3️⃣ Title & Intro dengan icon FontAwesome
# ----------------------------
st.markdown("<h1><i class='fa fa-robot'></i> Robot Surface Classification (KNN)</h1>", unsafe_allow_html=True)
st.markdown("<p>Deteksi permukaan robot: Licin vs Kasar</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'><span style='background-color:#E0F7FA; padding:5px 10px; border-radius:5px;'>Pipeline: StandardScaler → PCA → KNN (cosine, weights='distance')</span></p>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<h2>📁 Upload CSV File</h2>", unsafe_allow_html=True)
st.markdown("<p>File CSV harus memiliki <b>65 fitur getaran</b>. Kolom <b>label</b> opsional untuk evaluasi.</p>", unsafe_allow_html=True)

# ----------------------------
# 4️⃣ Upload CSV
# ----------------------------
uploaded_file = st.file_uploader("Upload CSV file here", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Preview data:")
    st.dataframe(data.head())

    # ----------------------------
    # Pisahkan fitur & label
    # ----------------------------
    if "label" in data.columns:
        X_raw = data.drop(columns=["label"]).values
        y_true = data["label"].values
    else:
        X_raw = data.values
        y_true = None

    # ----------------------------
    # Scaling & PCA
    # ----------------------------
    scaler = joblib.load("scaler_standard.pkl")
    X_scaled = scaler.transform(X_raw)

    pca = joblib.load("pca_model.pkl")
    X_pca = pca.transform(X_scaled)

    # ----------------------------
    # Load KNN model
    # ----------------------------
    knn_model = joblib.load("KNN_final_model.pkl")
    y_pred = knn_model.predict(X_pca)

    # ----------------------------
    # Mapping label ke nama permukaan
    # ----------------------------
    label_mapping = {0: "Permukaan Licin", 1: "Permukaan Kasar"}
    y_pred_int = [int(i) if int(i) in label_mapping else 0 for i in y_pred]
    y_pred_labels = [label_mapping[i] for i in y_pred_int]

    if y_true is not None:
        y_true_int = [int(i) if int(i) in label_mapping else 0 for i in y_true]
        y_true_labels = [label_mapping[i] for i in y_true_int]

    # ----------------------------
    # Tabel prediksi interaktif
    # ----------------------------
    data["Predicted Surface"] = y_pred_labels
    st.subheader("Tabel Prediksi Permukaan")
    def highlight_surface(val):
        if val == "Permukaan Licin":
            return 'background-color: lightgreen'
        elif val == "Permukaan Kasar":
            return 'background-color: lightcoral'
        else:
            return ''
    st.dataframe(data.style.applymap(lambda v: highlight_surface(v), subset=["Predicted Surface"]))

    # ----------------------------
    # Distribusi prediksi
    # ----------------------------
    st.subheader("Distribusi Prediksi Permukaan")
    pred_count = pd.Series(y_pred_labels).value_counts()
    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(pred_count)
    with col2:
        fig, ax = plt.subplots()
        ax.pie(pred_count, labels=pred_count.index, autopct='%1.1f%%', colors=['lightgreen','lightcoral'])
        ax.set_title("Distribusi Prediksi Permukaan")
        st.pyplot(fig)

    # ----------------------------
    # Confusion matrix & metrics
    # ----------------------------
    if y_true is not None:
        st.subheader("Evaluasi Model (dengan label asli)")
        cm = confusion_matrix(y_true_labels, y_pred_labels)
        st.write("Confusion Matrix:")
        st.dataframe(cm)

        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_mapping.values(),
                    yticklabels=label_mapping.values())
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")
        plt.title("Confusion Matrix Heatmap")
        st.pyplot(fig)

        accuracy = accuracy_score(y_true_labels, y_pred_labels)
        st.write(f"Accuracy: {accuracy*100:.2f}%")

        cr = classification_report(y_true_labels, y_pred_labels)
        st.text("Classification Report:\n" + cr)

# Footer info
st.markdown("---")
st.markdown("<p style='text-align:center;'>App menggunakan model <b>KNN</b> dengan preprocessing <b>StandardScaler → PCA → KNN</b>.</p>", unsafe_allow_html=True)
