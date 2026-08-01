"""Definisi transformer buatan sendiri yang dipakai di dalam pipeline.

Kelas di sini HARUS identik dengan yang dipakai saat melatih
(Perbaikan Scripts/Private_Project_Hari_Anak_Nasional.ipynb).
joblib hanya menyimpan nama modul + nama kelas, bukan kodenya,
jadi file ini wajib bisa diimpor saat pipeline dimuat.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class IQRCapper(BaseEstimator, TransformerMixin):
    """Jepit nilai ekstrem ke batas IQR. Dipakai hanya untuk kolom kontinu."""

    def fit(self, X, y=None):
        X = np.array(X)
        Q1, Q3 = np.percentile(X, 25, axis=0), np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        self.lower_, self.upper_ = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        return self

    def transform(self, X, y=None):
        X = np.array(X).copy()
        for i in range(X.shape[1]):
            X[:, i] = np.clip(X[:, i], self.lower_[i], self.upper_[i])
        return X


class IndeksKemakmuran(BaseEstimator, TransformerMixin):
    """Ringkas bahan_lantai, bahan_dinding, jumlah_kamar jadi satu skor
    kemakmuran lewat PCA komponen pertama.

    Ditaruh di dalam pipeline supaya pemakai cukup mengisi ketiga kolom asli
    dan skornya dihitung otomatis.
    """

    def __init__(self, kolom=("bahan_lantai", "bahan_dinding", "jumlah_kamar")):
        self.kolom = kolom              # WAJIB disimpan apa adanya (aturan clone sklearn)

    def fit(self, X, y=None):
        self.kolom_ = list(self.kolom)
        M = X[self.kolom_]
        self.scaler_ = StandardScaler().fit(M)
        self.pca_ = PCA(n_components=1).fit(self.scaler_.transform(M))
        self.arah_ = 1.0 if self.pca_.components_[0].sum() > 0 else -1.0
        return self

    def transform(self, X):
        X = X.copy()
        X["kemakmuran_skor"] = self.pca_.transform(
            self.scaler_.transform(X[self.kolom_]))[:, 0] * self.arah_
        return X.drop(columns=self.kolom_)
