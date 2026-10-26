
%%writefile app.py

import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
import numpy as np
from scipy.stats import shapiro, chi2
from scipy.spatial.distance import mahalanobis

# KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Iris Dashboard",
    page_icon="📊",
    layout="wide"
)

# JUDUL DASHBOARD
st.title("My Iris Dataset Dashboard 🌺")
st.write(
    """
    Dashboard untuk mengeksplorasi data Iris,
    menguji normalitas setiap variabel numerik,
    dan mendeteksi outlier multivariat.
    """
)
st.divider()

# MEMUAT DATASET
@st.cache_data
def load_data():
    return sns.load_dataset("iris")

df = load_data()

attributes = df.columns[:-1].tolist()
species = df["species"].unique().tolist()

# INFORMASI DATASET
with st.expander("About the Iris Dataset"):
    st.write(
        """
        Dataset Iris terdiri dari 150 observasi,
        empat variabel numerik, dan tiga spesies:
        Setosa, Versicolor, dan Virginica.
        """
    )

# SIDEBAR
with st.sidebar:
    st.header("Filters")
    selected_attribute = st.selectbox(
        "Select Attribute:", attributes, index=0
    )
    selected_species = st.multiselect(
        "Select Species:",
        species,
        default=species
    )

filtered_df = df[df["species"].isin(selected_species)]

# STATISTIK DESKRIPTIF
st.header("Basic Statistics")
st.dataframe(
    filtered_df[attributes].describe(),
    use_container_width=True
)

# FUNGSI VISUALISASI
def create_pairplot(data, attributes):
    fig = px.scatter_matrix(
        data,
        dimensions=attributes,
        color="species",
        opacity=0.6,
        title="Pairwise Scatter Plot"
    )
    fig.update_traces(diagonal_visible=False)
    fig.update_layout(height=750)
    return fig

def create_histogram(data, attribute):
    fig = px.histogram(
        data,
        x=attribute,
        color="species",
        marginal="box",
        barmode="overlay",
        opacity=0.7,
        title=f"Histogram of {attribute}"
    )
    fig.update_layout(hovermode="x unified")
    return fig

def create_violin_plot(data, attribute, show_points=True):
    points = "all" if show_points else False
    return px.violin(
        data,
        x="species",
        y=attribute,
        color="species",
        box=True,
        points=points,
        title=f"Violin Plot of {attribute}"
    )

# VISUALISASI DATA
st.header("Data Visualization")
st.subheader("Pairwise Scatter Plot (All Attributes)")

if filtered_df.empty:
    st.warning("Pilih minimal satu spesies pada sidebar.")
else:
    st.plotly_chart(
        create_pairplot(filtered_df, attributes),
        use_container_width=True
    )

st.subheader("Single Attribute Distribution")

if not filtered_df.empty:
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            create_histogram(filtered_df, selected_attribute),
            use_container_width=True
        )

    with col2:
        show_points = st.checkbox(
            "Show individual points", value=True
        )
        st.plotly_chart(
            create_violin_plot(
                filtered_df, selected_attribute, show_points
            ),
            use_container_width=True
        )

# UJI NORMALITAS UNIVARIAT
st.divider()
st.header("Normality Test (Shapiro-Wilk)")
st.write(
    "Uji normalitas dilakukan pada setiap variabel numerik "
    "untuk data yang sudah difilter."
)

if len(filtered_df) >= 3:
    normality_results = []

    for col in attributes:
        stat, p_value = shapiro(filtered_df[col])
        normality_results.append({
            "Variabel": col,
            "Statistik W": stat,
            "P-value": p_value,
            "Kesimpulan": (
                "Tidak menolak normalitas"
                if p_value >= 0.05
                else "Indikasi tidak normal"
            )
        })

    st.dataframe(
        pd.DataFrame(normality_results),
        use_container_width=True
    )
    st.caption(
        "Interpretasi pada taraf 5%: p-value ≥ 0,05 "
        "berarti tidak cukup bukti untuk menyatakan data "
        "menyimpang dari normalitas."
    )
else:
    st.warning("Diperlukan minimal 3 observasi untuk uji Shapiro-Wilk.")

# DETEKSI OUTLIER MULTIVARIAT
st.divider()
st.header("Multivariate Outlier Detection")
st.write(
    "Outlier dideteksi menggunakan jarak Mahalanobis "
    "berdasarkan empat variabel numerik."
)

if len(filtered_df) > len(attributes) + 1:
    X = filtered_df[attributes].to_numpy(dtype=float)
    mean_vector = np.mean(X, axis=0)
    cov_matrix = np.cov(X, rowvar=False)

    try:
        inv_cov = np.linalg.pinv(cov_matrix)
        distances = np.array([
            mahalanobis(x, mean_vector, inv_cov) ** 2
            for x in X
        ])

        cutoff = chi2.ppf(0.975, df=len(attributes))
        outlier_mask = distances > cutoff

        outlier_df = filtered_df.copy()
        outlier_df["Mahalanobis_D2"] = distances
        outlier_df["Status"] = np.where(
            outlier_mask, "Outlier", "Bukan outlier"
        )

        col1, col2 = st.columns(2)
        col1.metric("Jumlah observasi dianalisis", len(X))
        col2.metric("Jumlah kandidat outlier", int(outlier_mask.sum()))

        st.write(
            f"Batas Mahalanobis D² pada taraf 5% "
            f"(kuantil 97,5% chi-kuadrat, df=4): {cutoff:.3f}"
        )
        st.dataframe(
            outlier_df[outlier_df["Status"] == "Outlier"],
            use_container_width=True
        )
        with st.expander("Lihat seluruh hasil deteksi"):
            st.dataframe(outlier_df, use_container_width=True)

        st.caption(
            "Hasil ini merupakan kandidat outlier. "
            "Interpretasikan dengan memperhatikan spesies dan "
            "karakteristik data; asumsi distribusi perlu diperiksa."
        )
    except (ValueError, np.linalg.LinAlgError):
        st.error("Jarak Mahalanobis tidak dapat dihitung pada data terpilih.")
else:
    st.warning(
        "Pilih spesies dengan jumlah observasi yang cukup "
        "untuk mendeteksi outlier."
    )

# TABEL DATA
with st.expander("View Dataset"):
    st.dataframe(filtered_df, use_container_width=True)

st.divider()
st.caption(
    "Praktikum Eksplorasi dan Visualisasi Data Multivariat"
)

    mime="text/csv"
)
