# Skrining Prakonsepsi untuk Deteksi Risiko Stunting

> Model machine learning yang memperkirakan risiko stunting pada anak **hanya dari kondisi rumah tangga dan karakteristik orang tua yang sudah ada sebelum anak lahir**, sehingga skrining bisa dilakukan sebelum pasangan memutuskan memiliki anak, saat kondisi masih memungkinkan dibenahi. Data dari **IFLS-5** (Indonesian Family Life Survey, RAND, gelombang 2014-2015).

![python](https://img.shields.io/badge/Python-3.x-blue)
![methodology](https://img.shields.io/badge/methodology-CRISP--DM-informational)
![model](https://img.shields.io/badge/model-Random%20Forest-green)
![app](https://img.shields.io/badge/demo-Streamlit-red)

Project pribadi untuk Hari Anak Nasional.

---

## 📌 Overview

Sebagian besar model risiko stunting melakukan skrining pada anak yang **sudah lahir**. Proyek ini dirancang untuk berjalan lebih awal: memperkirakan risiko dari faktor yang sudah ada **sebelum kelahiran**, yaitu karakteristik orang tua, kondisi ekonomi keluarga, sanitasi dan air bersih, serta lingkungan sekitar.

Nilai jual utamanya adalah bingkai **prakonsepsi**: karena tidak memakai satu pun variabel dari anak, alat ini bisa dipakai ketika kondisi rumah tangga masih bisa diperbaiki, bukan setelah dampaknya terjadi.

Alat ini bersifat **skrining, bukan diagnosis**. Keluarannya menunjukkan seberapa besar perhatian yang dibutuhkan sebuah kasus, bukan kepastian tentang seorang anak.

---

## 🧠 Keputusan Desain 

### Antropometri anak sengaja tidak dipakai (menghindari label leakage)
Ini keputusan terpenting di proyek, dan semua konsekuensi lain mengalir darinya. Status stunting **dihitung dari tinggi anak** (Height-for-Age Z-score). Model yang memasukkan tinggi atau berat anak sebagai fitur sedang **menghitung ulang rumus labelnya sendiri**, bukan memprediksi apa pun. Karena itu seluruh fitur anak (tinggi, berat, HAZ, umur, jenis kelamin) dibuang.

Kolom yang dibuang karena leakage atau turunan label: tinggi anak, berat anak, `haz`, metode ukur, dan `ibu_umur` (karena `ibu_usia_prakonsepsi` diturunkan darinya). Umur dan jenis kelamin anak juga dibuang karena keputusan bingkai: **keduanya belum ada di titik skrining prakonsepsi**.

### Setiap fitur harus bisa diisi oleh pemakai alatnya
"Lantai rumah Anda dari apa?" bisa dijawab. "Berapa skor indeks kekayaan rumah tangga Anda?" tidak. Prinsip ini menentukan bentuk formulir aplikasi: pemakai menjawab dalam bahasa sehari-hari, aplikasi menghitung turunannya di belakang layar.

### Konsekuensi
Angka performa jauh lebih rendah daripada literatur yang memakai antropometri anak.

---

## 🗂️ Dataset

Sumber: **IFLS-5 (RAND Corporation)**, gelombang 2014-2015. Enam file digabung lewat kunci penaut rumah tangga dan orang tua.

| File IFLS-5 | Dipakai untuk |
|---|---|
| `bk_ar1.csv` | Roster rumah tangga; kunci penghubung anak ke orang tua (`ar10` ayah, `ar11` ibu) |
| `bus_us.csv` | Tinggi & berat, untuk label stunting dan tinggi orang tua |
| `bk_krk.csv` | Observasi pewawancara: bahan bangunan, kondisi lingkungan |
| `b2_kr.csv` | WASH dan riwayat renovasi rumah |
| `bk_sc1.csv` | Lokasi, untuk urban/rural |
| `b3b_km.csv` | Riwayat merokok ayah |

**Ukuran data:**

- **Dataset:** 3.659 baris
- **Prevalensi stunting:** sekitar 28%

**Label:** stunting bila Height-for-Age Z-score di bawah -2, dihitung terhadap WHO Child Growth Standards lewat library `pygrowup`.

> ⚠️ **Catatan lisensi data:** data IFLS-5 **tidak boleh didistribusikan ulang** sesuai ketentuan registrasi RAND. Repositori ini tidak menyertakan data mentah. Data dapat diakses secara gratis setelah registrasi resmi di RAND: https://www.rand.org/health/surveys/FLS/IFLS/access.html

---

## 🧩 Fitur

21 fitur dalam empat domain, semuanya kondisi yang sudah ada sebelum kelahiran:

- **Karakteristik orang tua (7):** `ibu_tinggi`, `ayah_tinggi`, `ibu_usia_konsepsi`, `ibu_pendidikan`, `ayah_pendidikan`, `ayah_merokok`, `ibu_bekerja`
- **Kondisi hunian (3):** `kemakmuran_skor`, `ventilasi_cukup`, `dapur_kamar_menyatu`
- **Air dan sanitasi (9):** `air_minum_layak`, `air_mck_layak`, `air_mck_sama`, `jamban_layak`, `sampah_aman`, `tumpukan_sampah`, `limbah_aman`, `limbah_dekat_rumah`, `air_tergenang`
- **Lingkungan sekitar (2):** `urban`, `dekat_kandang`

Pemilihan empat domain mengacu pada **Kepmenkes No. HK.01.07/MENKES/1928/2022 tentang Pedoman Nasional Pelayanan Kedokteran Tata Laksana Stunting (PNPK Stunting)**, yang memuat kerangka faktor risiko WHO. Pengelompokan air dan sanitasi (layak, meragukan, tidak layak) mengikuti WHO/JMP Core Questions on Drinking Water and Sanitation.

**Indeks kemakmuran** disusun dengan PCA komponen pertama atas tiga kolom kualitas material bangunan (`bahan_lantai`, `bahan_dinding`, `jumlah_kamar`). Wealth Index disusun dari kolom kolom tersebut dengan alasan data pendapatan tidak bisa dipakai karena masalah cross sectional data. Indeks ini **ditempatkan di dalam pipeline** sebagai transformer.

---

## 🔀 Metodologi

**Cabang 1: Prediksi**
1. Pipeline berisi transformer IndeksKemakmuran, IQRCapper, dan Scaler, diikuti classifier.
2. GridSearchCV 5 lipatan atas Random Forest, SVM, dan XGBoost.
3. Random Forest terpilih, lalu penyetelan ambang, lalu SHAP untuk penjelasan.

**Cabang 2: Penemuan pola (Association Rule Mining)**
1. Diskretisasi variabel, lalu FP-Growth untuk mencari aturan menuju `stunting=1`.
2. Aturan ditambang **dari data latih saja**, lalu divalidasi ulang di data uji lewat `lift`.

Hasil ARM **tidak dimasukkan ke model** (alasan di bawah). Perannya adalah **menceritakan kombinasi** yang saling berkumpul di satu rumah, sesuatu yang bisa dipakai pohon untuk menebak tetapi tidak bisa dituliskan, dan yang secara matematis tidak bisa ditunjukkan SHAP.

---

## ⚙️ Keputusan Metodologis dan Alasannya

- **SMOTE tidak dipakai.** Prevalensi sekitar 28% bukan ketidakseimbangan parah. Pengujian menunjukkan SMOTE tidak menaikkan kemampuan model membedakan (PR-AUC praktis sama), hanya menggeser titik operasi, dan pada data campuran biner-kontinu ia mengarang nilai pecahan di kolom biner. Penggantinya: `class_weight='balanced'` plus penyetelan ambang.
- **Ambang disetel, bukan dibiarkan di 0,5.** Melewatkan anak berisiko jauh lebih mahal daripada alarm palsu. Target recall ditetapkan lebih dulu (0,65), ambang dicari **di data latih** lewat cross-validation, lalu diterapkan apa adanya ke data uji.
- **ARM tidak dijadikan fitur model.** Pengujian menunjukkan pohon mendapat nol perbaikan; aturannya jenis "dan" biasa yang tiap penyusunnya sudah bersinyal sendiri; dan fitur-aturan memasukkan informasi dua kali sehingga membuat SHAP sulit dibaca.
- **ARM ditambang dari data latih saja.** Karena 15 aturan dipilih dari ribuan kandidat berdasarkan lift, sebagian yang di puncak naik karena keberuntungan. Validasi di data uji memisahkan yang nyata dari yang kebetulan (satu aturan terbukti runtuh dari lift 1,92 menjadi 0,91).

---

## 📊 Hasil

Metrik utama adalah **PR-AUC**, bukan accuracy, karena kelasnya timpang dan yang penting adalah kemampuan menemukan anak berisiko.

| Model | Test PR-AUC | Recall (ambang) | Precision (ambang) |
|---|---|---|---|
| **Random Forest** (terpilih) | **0,4534** | 0,632 | 0,376 |
| XGBoost | 0,4309 | 0,639 | 0,366 |
| SVM | 0,4283 | 0,590 | 0,380 |

Ambang keputusan Random Forest: **0,418** (dari target recall 0,65). Prevalensi dasar sekitar 0,28.

**Terjemahan praktisnya:** dari sekitar 521 keluarga yang dikunjungi, model menandai 196 anak berisiko, sementara pemilihan acak dengan tenaga yang sama menemukan sekitar 147. Selisih sekitar 49 anak.

**Porsi kepentingan fitur (SHAP), per domain:**

| Domain | Porsi |
|---|---|
| Karakteristik orang tua | 62,4% |
| Air dan sanitasi | 20,9% |
| Kondisi hunian | 8,4% |
| Lingkungan sekitar | 8,3% |

Tinggi ibu (20,6%) dan tinggi ayah (19,5%) menempati dua posisi teratas.

---

## 📝 Keterbatasan

- **Cross-sectional.** Data satu titik waktu, tidak bisa menunjukkan arah sebab-akibat maupun menelusuri perubahan.
- **Ketidaksesuaian waktu.** Kondisi tercatat 2014, sedangkan masa kritis sebagian anak terjadi 2011 sampai 2013. Mitigasi: survei yang sama menanyakan riwayat renovasi sejak 2007, dan hitungan menunjukkan kurang dari 1% rumah tangga mengubah sistem pipa airnya, sehingga asumsi kestabilan dapat dipertahankan untuk faktor yang berubah lambat, **tidak untuk yang berubah cepat**.
- **Faktor setelah kelahiran tidak tercakup.** Pola makan, penyakit berulang, ASI, dan pengasuhan tidak ada di data. Ini batas struktural.
- **Bias seleksi.** Complete case membuang sekitar 28% baris, terutama karena data ayah yang hilang.
- **SHAP menjelaskan model, bukan dunia.** Porsi kepentingan adalah atribusi terhadap cara model berpikir, bukan atribusi kausal terhadap kejadian stunting.
- **Ambang `ayah_tinggi` (160, 170 cm) belum memiliki sumber primer** dan saat ini dipilih dari klasifikasi umum, bukan pedoman resmi.
- **Indeks kemakmuran menunjukkan posisi, bukan lintasan.** Indeks disusun dari kualitas material bangunan, yaitu kondisi yang berubah pelan, sehingga klaim pra-kelahirannya bergantung pada asumsi bahwa kualitas hunian 2014 mendekati kondisi 2011 sampai 2013. Uji renovasi menunjukkan asumsi ini masuk akal untuk sebagian besar sampel, tetapi rumah tangga yang membaik dan yang memburuk tetap terlihat sama selama posisi akhirnya sama.

---

> **Catatan teknis:** kelas `IQRCapper` dan `IndeksKemakmuran` harus didefinisikan di `transformers.py` **sebelum** pickle dibuat. Kalau didefinisikan di notebook, modulnya tercatat sebagai `__main__` dan `joblib.load` akan gagal di Streamlit. Versi `scikit-learn` juga perlu dipatok persis di `requirements.txt`, karena pickle model bisa gagal dimuat di versi berbeda.

---

## 📚 References

- **IFLS-5 (Indonesian Family Life Survey), RAND Corporation**, gelombang 2014-2015. Sumber seluruh data. Halaman resmi: https://www.rand.org/well-being/social-and-behavioral-policy/data/FLS/IFLS.html (unduhan memerlukan registrasi di https://www.rand.org/health/surveys/FLS/IFLS/access.html). Sitasi resmi harus diambil dari halaman unduhan RAND, dan data tidak boleh didistribusikan ulang.
- **WHO Child Growth Standards**, dipakai untuk ambang stunting (HAZ di bawah -2) lewat library `pygrowup`.
- **WHO/JMP Core Questions on Drinking Water and Sanitation**, dipakai untuk pengelompokan WASH.
- **Kementerian Kesehatan RI.** Keputusan Menteri Kesehatan Nomor HK.01.07/MENKES/1928/2022 tentang Pedoman Nasional Pelayanan Kedokteran Tata Laksana Stunting (PNPK Stunting). Ditetapkan 25 November 2022. Dipakai sebagai dasar pemilihan empat domain faktor risiko.
- **Utamima, A., Kanedi, F. J., & Sohel, F. (2026).** Environmental drivers of bushfire warning categories: insights from official records, meteorology, and drought data in Western Australia. *Environmental Challenges, 23*, 101509. https://doi.org/10.1016/j.envc.2026.101509. Dipakai sebagai rujukan struktur diagram alur dan pembanding metodologi ARM.

---

## 👤 Author

**Muhammad Faishal Ardiansyah**, [@Ishalllll](https://github.com/Ishalllll)

---
