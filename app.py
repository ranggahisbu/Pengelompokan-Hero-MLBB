import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="K-Means Clustering MLBB Hero", layout="wide")
st.title("🛡️ Pengelompokan Hero Mobile Legends: M5 Edition")

# --- Load CSV Langsung dari Repo ---
@st.cache_data
def load_data():
    df = pd.read_csv("M5_World_Championship.csv")
    return df

df = load_data()

# --- Tampilkan Data Awal ---
st.subheader("🔍 Data Awal")
st.dataframe(df)

# --- Fitur yang digunakan ---
features = ['T_PickPercentage', 'T_BansPercentage', 'T_WinRate', 'T_PicksBansPercentage']
data = df.copy()

# --- Preprocessing Data ---
for col in features:
    data[col] = data[col].str.replace('%', '').replace('-', np.nan).astype(float).fillna(0)

# --- Normalisasi ---
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[features])

# --- Pilihan Jumlah Klaster ---
st.sidebar.header("⚙️ Pengaturan Klaster")
k = st.sidebar.slider("Jumlah Klaster (k)", min_value=2, max_value=10, value=5)

# --- Clustering KMeans ---
kmeans = KMeans(n_clusters=k, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_data)
data['Cluster'] = cluster_labels

# --- Evaluasi ---
sil_score = silhouette_score(scaled_data, cluster_labels)
st.markdown(f"### 📈 Silhouette Score: `{sil_score:.4f}`")

# --- PCA & Visualisasi ---
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

pca_df = pd.DataFrame(pca_result, columns=["PCA1", "PCA2"])
pca_df["Cluster"] = cluster_labels
pca_df["Hero"] = data["Hero"]

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=pca_df, x="PCA1", y="PCA2", hue="Cluster", palette="tab10", s=70, ax=ax)

# Tambahkan label nama hero
for i in range(pca_df.shape[0]):
    ax.text(pca_df.PCA1[i]+0.01, pca_df.PCA2[i]+0.01, str(pca_df.Hero[i]), fontsize=8, alpha=0.6)

ax.set_title("Visualisasi Klaster Hero (PCA)")
st.pyplot(fig)

# --- Tabel Hasil Klastering ---
st.subheader("📋 Hasil Klastering")
st.dataframe(data[['Hero'] + features + ['Cluster']])
