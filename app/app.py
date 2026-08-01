# -*- coding: utf-8 -*-
"""Demo teknis: prediksi risiko stunting dari kondisi struktural pra-kelahiran.

Model dilatih pada IFLS-5 (2014). Ini demo portofolio, bukan alat kesehatan.
Jalankan: streamlit run app.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import derivasi as dv
import transformers

# pickle dibuat saat kelas masih di __main__ (notebook); beri alias supaya
# joblib.load menemukannya. Streamlit menjalankan skrip bukan sebagai __main__.
sys.modules["__main__"].IQRCapper = transformers.IQRCapper
sys.modules["__main__"].IndeksKemakmuran = transformers.IndeksKemakmuran

st.set_page_config(page_title="Demo Prediksi Risiko Stunting", page_icon="📊")


MODEL_DIR = Path(__file__).parent / "model"


@st.cache_resource
def muat():
    model = joblib.load(MODEL_DIR / "model_stunting.pkl")
    with open(MODEL_DIR / "meta_model.json") as f:
        meta = json.load(f)
    return model, meta


@st.cache_resource
def buat_explainer(_model):
    import shap
    return shap.TreeExplainer(_model.named_steps["classifier"])


model, meta = muat()
AMBANG = meta["ambang"]

NAMA_TAMPIL = {
    "ibu_tinggi":          "Tinggi badan ibu",
    "ayah_tinggi":         "Tinggi badan ayah",
    "ibu_usia_konsepsi":   "Usia ibu saat konsepsi",
    "kemakmuran_skor":     "Kualitas hunian (lantai, dinding, jumlah kamar)",
    "ibu_pendidikan":      "Pendidikan ibu",
    "ayah_pendidikan":     "Pendidikan ayah",
    "urban":               "Tinggal di kota / desa",
    "ibu_bekerja":         "Ibu bekerja",
    "ayah_merokok":        "Ayah merokok",
    "limbah_dekat_rumah":  "Limbah di dekat rumah",
    "tumpukan_sampah":     "Tumpukan sampah di sekitar rumah",
    "air_tergenang":       "Air tergenang di sekitar rumah",
    "dekat_kandang":       "Rumah dekat kandang ternak",
    "ventilasi_cukup":     "Ventilasi rumah",
    "dapur_kamar_menyatu": "Dapur menyatu dengan kamar tidur",
    "air_minum_layak":     "Kelayakan sumber air minum",
    "air_mck_layak":       "Kelayakan air mandi / cuci",
    "air_mck_sama":        "Air MCK sama dengan air minum",
    "jamban_layak":        "Kelayakan jamban",
    "limbah_aman":         "Cara pembuangan limbah cair",
    "sampah_aman":         "Cara pembuangan sampah",
}

# ---------------------------------------------------------------------------
# Sidebar: keterbatasan (wajib ada)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Keterbatasan")
    st.markdown(
        """
- Alat skrining teknis untuk demo, bukan diagnosis. Tidak menggantikan
  pemeriksaan tenaga kesehatan.
- Hanya memakai kondisi rumah tangga dan orang tua sebelum anak lahir. Faktor
  setelah kelahiran (pola makan, penyakit, pengasuhan) tidak diperhitungkan
  dan itu sebabnya performanya terbatas.
- Dilatih pada data IFLS-5 tahun 2014 & mungkin tidak mewakili kondisi
  sekarang.
- Data cross-sectional: hasil tidak dapat menyatakan sebab-akibat, hanya
  asosiasi.
- Angka SHAP menjelaskan cara model berpikir, bukan besarnya pengaruh di
  dunia nyata.
"""
    )

st.title("Demo Prediksi Risiko Stunting")
st.caption(
    "Menguji model terlatih IFLS-5: risiko stunting anak 0 sampai 59 bulan "
    "diprediksi hanya dari kondisi struktural rumah tangga dan karakteristik "
    "orang tua sebelum kelahiran."
)

# ---------------------------------------------------------------------------
# Formulir, dikelompokkan per domain. Nilai bawaan = modus data latih.
# ---------------------------------------------------------------------------
def pilih(label, opsi: dict, bantuan=None):
    """Dropdown tanpa pilihan awal: mengembalikan None sampai pemakai memilih."""
    return st.selectbox(label, list(opsi), index=None,
                        placeholder="belum dipilih",
                        format_func=lambda k: opsi[k], help=bantuan)


def ya_tidak(label, bantuan=None):
    return pilih(label, {1: "Ya", 0: "Tidak"}, bantuan)


def angka(label, lo, hi, langkah=None, bantuan=None):
    """Kotak angka yang mulai kosong. Rentang mengikuti data latih."""
    return st.number_input(label, min_value=lo, max_value=hi, value=None,
                           step=langkah, placeholder=f"{lo:g} sampai {hi:g}",
                           help=bantuan)


st.info("Semua kolom wajib diisi. Model tidak menerima nilai kosong.")

st.subheader("1 · Orang tua")
k1, k2 = st.columns(2)
with k1:
    ibu_tinggi = angka("Tinggi badan ibu (cm)", 130.0, 185.0, 0.5)
    ibu_usia = angka("Usia ibu saat konsepsi (tahun)", 15, 50)
    ibu_pendidikan = pilih("Pendidikan terakhir ibu", dv.LABEL_PENDIDIKAN)
    ibu_bekerja = ya_tidak("Ibu bekerja")
with k2:
    ayah_tinggi = angka("Tinggi badan ayah (cm)", 140.0, 195.0, 0.5)
    ayah_pendidikan = pilih("Pendidikan terakhir ayah", dv.LABEL_PENDIDIKAN)
    ayah_merokok = ya_tidak("Ayah pernah punya kebiasaan merokok")

st.subheader("2 · Kondisi hunian")
k1, k2 = st.columns(2)
with k1:
    krk08 = pilih("Bahan utama lantai", dv.LABEL_LANTAI)
    jumlah_kamar = angka("Jumlah kamar/ruangan di rumah", 1, 15)
    ventilasi = ya_tidak("Ventilasi rumah memadai")
with k2:
    krk09 = pilih("Bahan utama dinding luar", dv.LABEL_DINDING)
    dapur_kamar = ya_tidak("Dapur menyatu dengan kamar tidur")

st.subheader("3 · Air & sanitasi")
k1, k2 = st.columns(2)
with k1:
    kr13 = pilih("Sumber utama air minum", dv.LABEL_AIR_MINUM)
    mck_sama = ya_tidak("Air untuk mandi/cuci berasal dari sumber yang sama "
                        "dengan air minum")
    # ditanyakan hanya kalau sumbernya beda, mengikuti pola lompat KR16
    kr17 = pilih("Sumber utama air mandi/cuci", dv.LABEL_AIR_MCK) \
        if mck_sama == 0 else None
with k2:
    kr20 = pilih("Tempat buang air besar sebagian besar anggota rumah",
                 dv.LABEL_JAMBAN)
    kr21 = pilih("Pembuangan air limbah rumah tangga", dv.LABEL_LIMBAH)
    kr22 = pilih("Cara membuang sampah", dv.LABEL_SAMPAH)

st.subheader("4 · Lingkungan sekitar")
k1, k2 = st.columns(2)
with k1:
    urban = pilih("Wilayah tempat tinggal", {1: "Kota", 0: "Desa"})
    limbah_dekat = ya_tidak("Ada limbah (manusia/hewan) di dekat rumah")
    tumpukan = ya_tidak("Ada tumpukan sampah di sekitar rumah")
with k2:
    tergenang = ya_tidak("Ada air tergenang di sekitar rumah")
    kandang = ya_tidak("Rumah berdekatan dengan kandang ternak")

# --- daftar isian wajib, untuk memeriksa kelengkapan sebelum memprediksi ---
WAJIB = {
    "Tinggi badan ibu": ibu_tinggi,
    "Usia ibu saat konsepsi": ibu_usia,
    "Pendidikan terakhir ibu": ibu_pendidikan,
    "Ibu bekerja": ibu_bekerja,
    "Tinggi badan ayah": ayah_tinggi,
    "Pendidikan terakhir ayah": ayah_pendidikan,
    "Ayah merokok": ayah_merokok,
    "Bahan utama lantai": krk08,
    "Bahan utama dinding luar": krk09,
    "Jumlah kamar": jumlah_kamar,
    "Ventilasi rumah": ventilasi,
    "Dapur menyatu dengan kamar": dapur_kamar,
    "Sumber utama air minum": kr13,
    "Air mandi/cuci sama dengan air minum": mck_sama,
    "Tempat buang air besar": kr20,
    "Pembuangan air limbah": kr21,
    "Cara membuang sampah": kr22,
    "Wilayah tempat tinggal": urban,
    "Limbah di dekat rumah": limbah_dekat,
    "Tumpukan sampah": tumpukan,
    "Air tergenang": tergenang,
    "Dekat kandang ternak": kandang,
}
if mck_sama == 0:                      # hanya wajib saat sumbernya berbeda
    WAJIB["Sumber utama air mandi/cuci"] = kr17

belum_diisi = [nama for nama, nilai in WAJIB.items() if nilai is None]

# ---------------------------------------------------------------------------
# Prediksi
# ---------------------------------------------------------------------------
tekan = st.button("Prediksi risiko", type="primary", disabled=bool(belum_diisi))
if belum_diisi:
    st.caption(f"Belum terisi ({len(belum_diisi)}): " + ", ".join(belum_diisi))

if tekan:
    jawaban_mentah = {
        "ibu_tinggi": ibu_tinggi, "ayah_tinggi": ayah_tinggi,
        "ibu_usia_konsepsi": float(ibu_usia), "jumlah_kamar": int(jumlah_kamar),
        "ibu_pendidikan": ibu_pendidikan, "ayah_pendidikan": ayah_pendidikan,
        "urban": urban, "ibu_bekerja": ibu_bekerja, "ayah_merokok": ayah_merokok,
        "limbah_dekat_rumah": limbah_dekat, "tumpukan_sampah": tumpukan,
        "air_tergenang": tergenang, "dekat_kandang": kandang,
        "ventilasi_cukup": ventilasi, "dapur_kamar_menyatu": dapur_kamar,
        "kr13": kr13, "air_mck_sama": mck_sama, "kr17": kr17,
        "kr20": kr20, "kr21": kr21, "kr22": kr22,
        "krk08": krk08, "krk09": krk09,
    }
    jawaban = dv.turunkan(jawaban_mentah)

    baris = pd.DataFrame([jawaban]).reindex(columns=meta["kolom_input"])
    assert not baris.isna().any().any(), "ada kolom yang belum terisi"

    proba = float(model.predict_proba(baris)[0, 1])
    berisiko = proba >= AMBANG

    st.divider()
    # Satu angka besar saja. Ambang bukan penilaian tinggi rendahnya risiko,
    # melainkan aturan tindakan, jadi ia turun ke baris status dan keterangan.
    st.metric("Prediksi risiko untuk profil ini", f"{proba*100:.1f}%")

    if berisiko:
        st.warning("**Ditandai untuk tindak lanjut**")
    else:
        st.info("**Tidak ditandai untuk tindak lanjut**")

    st.markdown(
        "Model ini mungkin tidak akurat dan hasilnya tidak layak dipercaya "
        "untuk menilai anak secara perorangan karena keterbatasan data "
        "training. Angka di atas hanya menggambarkan kecenderungan pada data "
        "2014, bukan kondisi anak yang sebenarnya."
    )
    st.caption(
        f"Ambang penandaan {AMBANG*100:.1f}%, dipilih dari data latih untuk "
        f"target recall {meta['target_recall']:.2f}. Ambang menentukan apakah "
        "sebuah profil ditandai, bukan tingkat risikonya. Alat skrining teknis "
        "untuk demo, bukan diagnosis, dan tidak menggantikan pemeriksaan "
        "tenaga kesehatan."
    )

    # -----------------------------------------------------------------------
    # Rincian SHAP
    # -----------------------------------------------------------------------
    explainer = buat_explainer(model)
    X_trans = np.asarray(model[:-1].transform(baris), dtype=float)

    sv = explainer.shap_values(X_trans)
    if isinstance(sv, list):
        nilai = sv[1][0]
    elif np.array(sv).ndim == 3:
        nilai = np.array(sv)[0, :, 1]
    else:
        nilai = np.array(sv)[0]

    base = explainer.expected_value
    base = float(base[1]) if hasattr(base, "__len__") else float(base)

    if abs(base + nilai.sum() - proba) > 0.001:
        st.error("Rincian SHAP tidak konsisten dengan prediksi, "
                 "kemungkinan urutan kolom bergeser. Rincian tidak ditampilkan.")
    else:
        st.subheader("Faktor yang mendorong prediksi")
        st.caption(
            "Rincian penjumlahan menurut model (SHAP), satuan poin persen. "
            "Titik awal adalah rata-rata prediksi model, karena model ini dilatih "
            "dengan pembobotan kelas seimbang sehingga titik awalnya sekitar "
            "50%, bukan prevalensi."
        )
        kolom_shap = meta["kolom_setelah_transform"]
        urut = np.argsort(-np.abs(nilai))

        baris_rincian = [f"{'Titik awal (rata-rata prediksi model)':44s} "
                         f"{base*100:6.1f}%"]
        for i in urut[:8]:
            baris_rincian.append(
                f"   {NAMA_TAMPIL.get(kolom_shap[i], kolom_shap[i]):41s} "
                f"{nilai[i]*100:+6.1f} poin")
        sisa = nilai[urut[8:]].sum()
        if len(urut) > 8:
            baris_rincian.append(f"   {'faktor lainnya':41s} {sisa*100:+6.1f} poin")
        baris_rincian.append(f"{'Hasil prediksi':44s} {proba*100:6.1f}%")
        st.code("\n".join(baris_rincian), language=None)

        st.caption(
            "Nilai positif mendorong prediksi ke arah risiko lebih tinggi, "
            "negatif ke arah lebih rendah. Ini penjelasan cara model "
            "berpikir, bukan pengaruh di dunia nyata."
        )
