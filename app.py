
import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
import numpy as np
from scipy.stats import shapiro, chi2

# ==========================================
# PENGATURAN HALAMAN
# ==========================================
st.set_page_config(
    page_title="Iris Dashboard",
    page_icon="📊",
    layout="wide"
)

# ==========================================
# JUDUL DASHBOARD
# ==========================================
st.title("My Iris Dataset Dashboard 🌺")
st.write(
    "Dashboard untuk eksplorasi data, visualisasi bivariat dan multivariat, "
    "uji normalitas, serta deteksi outlier."
)

# ==========================================
# MEMUAT DATA
# ==========================================
@st.cache_data
def load_data():
    df = sns.load_dataset("iris")
    return df


df = load_data()

attributes = df.select_dtypes(include=np.number).columns.tolist()
species_list = df["species"].unique().tolist()

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("Pengaturan Dashboard")

selected_attribute = st.sidebar.selectbox(
    "Pilih Variabel",
    attributes
)

selected_species = st.sidebar.multiselect(
    "Pilih Spesies",
    species_list,
    default=species_list
)

# ==========================================
# FILTER DATA
# ==========================================
filtered_df = df[df["species"].isin(selected_species)].copy()

if filtered_df.empty:
    st.warning("Pilih minimal satu spesies pada sidebar.")
    st.stop()

# ==========================================
# INFORMASI DATA
# ==========================================
st.header("1. Informasi Data")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Jumlah Observasi", len(filtered_df))

with col2:
    st.metric("Jumlah Variabel Numerik", len(attributes))

with col3:
    st.metric("Jumlah Spesies Terpilih", filtered_df["species"].nunique())

st.subheader("Data Iris")
st.dataframe(filtered_df, use_container_width=True)

st.subheader("Statistik Deskriptif")
st.dataframe(
    filtered_df[attributes].describe(),
    use_container_width=True
)

# ==========================================
# VISUALISASI DATA
# ==========================================
st.header("2. Eksplorasi dan Visualisasi Data")

# Scatter Matrix
st.subheader("2.1 Scatter Matrix")

fig_scatter = px.scatter_matrix(
    filtered_df,
    dimensions=attributes,
    color="species",
    title="Scatter Matrix Variabel Iris"
)

fig_scatter.update_layout(height=750)
st.plotly_chart(fig_scatter, use_container_width=True)

# Histogram
st.subheader("2.2 Histogram dan Box Plot")

fig_hist = px.histogram(
    filtered_df,
    x=selected_attribute,
    color="species",
    marginal="box",
    barmode="overlay",
    title=f"Distribusi {selected_attribute} Berdasarkan Spesies"
)

st.plotly_chart(fig_hist, use_container_width=True)

# Violin Plot
st.subheader("2.3 Violin Plot")

fig_violin = px.violin(
    filtered_df,
    x="species",
    y=selected_attribute,
    color="species",
    box=True,
    points="all",
    title=f"Sebaran {selected_attribute} Berdasarkan Spesies"
)

st.plotly_chart(fig_violin, use_container_width=True)

# ==========================================
# UJI NORMALITAS SHAPIRO-WILK
# ==========================================
st.header("3. Uji Normalitas")

st.write(
    "Uji Shapiro-Wilk dilakukan pada setiap variabel numerik "
    "untuk masing-masing spesies yang dipilih."
)

normality_results = []

for species in selected_species:
    group = filtered_df.loc[
        filtered_df["species"] == species,
        attributes
    ]

    for attribute in attributes:
        values = group[attribute].dropna()

        if 3 <= len(values) <= 5000:
            statistic, p_value = shapiro(values)

            if p_value > 0.05:
                conclusion = "Tidak menolak H0 (normal)"
            else:
                conclusion = "Menolak H0 (tidak normal)"

            normality_results.append({
                "Spesies": species,
                "Variabel": attribute,
                "Statistik Shapiro-Wilk": statistic,
                "P-value": p_value,
                "Kesimpulan": conclusion
            })

if normality_results:
    normality_df = pd.DataFrame(normality_results)
    st.dataframe(normality_df, use_container_width=True)
else:
    st.warning(
        "Data belum cukup untuk melakukan uji Shapiro-Wilk."
    )

st.caption(
    "Kriteria: jika p-value > 0,05, data tidak menunjukkan "
    "bukti yang cukup untuk menolak normalitas. "
    "Uji ini memeriksa normalitas tiap variabel, bukan normalitas multivariat."
)

# ==========================================
# DETEKSI OUTLIER MULTIVARIAT
# ==========================================
st.header("4. Deteksi Outlier Multivariat")

st.write(
    "Deteksi dilakukan menggunakan jarak Mahalanobis kuadrat (D²) "
    "berdasarkan empat variabel numerik. Perhitungan dilakukan "
    "secara terpisah untuk setiap spesies."
)

outlier_results = []

degrees_freedom = len(attributes)
cutoff = chi2.ppf(0.95, df=degrees_freedom)

for species in selected_species:
    group = filtered_df.loc[
        filtered_df["species"] == species,
        attributes
    ].dropna()

    if len(group) <= degrees_freedom:
        continue

    values = group.to_numpy()
    mean_values = np.mean(values, axis=0)
    covariance = np.cov(values, rowvar=False)
    inverse_covariance = np.linalg.pinv(covariance)

    differences = values - mean_values

    mahalanobis_squared = np.einsum(
        "ij,jk,ik->i",
        differences,
        inverse_covariance,
        differences
    )

    for position, (index, distance_squared) in enumerate(
        zip(group.index, mahalanobis_squared)
    ):
        if distance_squared > cutoff:
            row = filtered_df.loc[index]

            outlier_results.append({
                "Nomor Data": int(index) + 1,
                "Spesies": species,
                "Sepal Length": row["sepal_length"],
                "Sepal Width": row["sepal_width"],
                "Petal Length": row["petal_length"],
                "Petal Width": row["petal_width"],
                "Jarak Mahalanobis D²": distance_squared,
                "Batas Chi-Square": cutoff
            })

st.write(f"Batas Chi-Square (α = 0,05): {cutoff:.4f}")

if outlier_results:
    outlier_df = pd.DataFrame(outlier_results)

    st.warning(
        f"Ditemukan {len(outlier_df)} kandidat outlier "
        "berdasarkan batas yang digunakan."
    )

    st.dataframe(outlier_df, use_container_width=True)
else:
    st.success(
        "Tidak ditemukan kandidat outlier berdasarkan batas yang digunakan."
    )

st.caption(
    "Outlier ditandai jika D² lebih besar dari kuantil 95% distribusi "
    "Chi-Square dengan derajat bebas sebanyak variabel numerik. "
    "Hasil ini merupakan kandidat outlier dan perlu diperiksa lebih lanjut."
)

# ==========================================
# KESIMPULAN DATA
# ==========================================
st.header("5. Ringkasan")

st.write(
    f"Data yang sedang dianalisis terdiri dari {len(filtered_df)} observasi "
    f"dan {len(attributes)} variabel numerik. "
    f"Spesies yang dipilih: {', '.join(selected_species)}."
)

st.caption(
    "Praktikum Eksplorasi dan Visualisasi Data Multivariat"
)

