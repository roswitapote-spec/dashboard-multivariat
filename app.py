import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, chi2

st.set_page_config(
    page_title="Dashboard Multivariat",
    page_icon="📊",
    layout="wide"
)

st.title("Dashboard Analisis Data Multivariat")
st.write(
    "Eksplorasi data, visualisasi, uji normalitas, "
    "dan deteksi outlier."
)

file = st.file_uploader(
    "Unggah data hasil analisis R (CSV)",
    type=["csv"]
)

if file is not None:
    df = pd.read_csv(file)
elif __import__("os").path.exists("data_multivariat.csv"):
    df = pd.read_csv("data_multivariat.csv")
else:
    st.info("Unggah file CSV untuk memulai analisis.")
    st.stop()

num = df.select_dtypes(include=np.number)

st.subheader("1. Ringkasan Data")
c1, c2, c3 = st.columns(3)
c1.metric("Jumlah baris", df.shape[0])
c2.metric("Jumlah kolom", df.shape[1])
c3.metric("Data kosong", int(df.isna().sum().sum()))

st.dataframe(df, use_container_width=True)

st.subheader("2. Statistik Deskriptif")
st.dataframe(num.describe().T, use_container_width=True)

if num.shape[1] >= 2:
    st.subheader("3. Visualisasi Multivariat")

    x = st.selectbox("Variabel X", num.columns, index=0)
    y = st.selectbox(
        "Variabel Y",
        num.columns,
        index=1
    )

    plotdata = num[[x, y]].dropna()

    fig, ax = plt.subplots()
    ax.scatter(plotdata[x], plotdata[y])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"Hubungan {x} dan {y}")
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("4. Matriks Korelasi")
    corr = num.corr()

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.subheader("5. Uji Normalitas Tiap Variabel")
hasil = []

for kolom in num.columns:
    nilai = num[kolom].dropna()

    if 3 <= len(nilai) <= 5000 and nilai.nunique() > 1:
        stat, p = shapiro(nilai)
        hasil.append({
            "Variabel": kolom,
            "Statistik": stat,
            "P-value": p,
            "Kesimpulan": (
                "Tidak menolak normalitas"
                if p > 0.05
                else "Tidak berdistribusi normal"
            )
        })

if hasil:
    st.dataframe(pd.DataFrame(hasil), use_container_width=True)
else:
    st.write("Data tidak memenuhi syarat uji Shapiro-Wilk.")

st.subheader("6. Deteksi Outlier Multivariat")

bersih = num.replace(
    [np.inf, -np.inf], np.nan
).dropna()

if bersih.shape[1] >= 2 and len(bersih) > bersih.shape[1]:
    try:
        pusat = bersih.mean().to_numpy()
        kov = np.cov(bersih.to_numpy(), rowvar=False)
        inv_kov = np.linalg.pinv(kov)

        selisih = bersih.to_numpy() - pusat
        jarak = np.einsum(
            "ij,jk,ik->i", selisih, inv_kov, selisih
        )

        batas = chi2.ppf(0.95, df=bersih.shape[1])

        outlier = bersih.copy()
        outlier["Jarak_Mahalanobis"] = jarak
        outlier["Status"] = np.where(
            jarak > batas, "Outlier", "Bukan Outlier"
        )

        st.write(f"Batas jarak Mahalanobis: {batas:.3f}")
        st.dataframe(
            outlier[outlier["Status"] == "Outlier"],
            use_container_width=True
        )

        fig, ax = plt.subplots()
        ax.scatter(range(1, len(jarak) + 1), jarak)
        ax.axhline(batas, color="red", linestyle="--")
        ax.set_xlabel("Nomor Pengamatan")
        ax.set_ylabel("Jarak Mahalanobis")
        ax.set_title("Deteksi Outlier")
        st.pyplot(fig)
        plt.close(fig)

    except (ValueError, np.linalg.LinAlgError):
        st.warning("Outlier tidak dapat dihitung dari data ini.")
else:
    st.info("Diperlukan minimal dua variabel numerik.")

st.subheader("7. Unduh Data")
st.download_button(
    "Unduh hasil data CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="data_dashboard.csv",
    mime="text/csv"
)
