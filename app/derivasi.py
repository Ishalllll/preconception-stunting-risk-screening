# -*- coding: utf-8 -*-
"""Lapisan penerjemah: jawaban formulir (kode mentah IFLS-5) -> kolom model.

Aturan klasifikasi DISALIN, bukan dikarang, dari:
  - scripts/Collect_Data_Hari_Anak_Nasional (3).ipynb
      TAHAP awal : peta pendidikan JENJANG (ar16 -> jenjang 0-4)
      TAHAP 9    : pengelompokan WASH (AIR_LAYAK/RAGU/TAK, JAMBAN,
                   LIMBAH_AMAN, SAMPAH_AMAN, pola lompat kr16)
  - Perbaikan Scripts/Private_Project_Hari_Anak_Nasional.ipynb
      pembalikan arah bahan_lantai (7 - kode) dan bahan_dinding (4 - kode),
      recode biner 1/3 -> 1/0

Label kode mentah diambil dari Codebook/ifls2014_hhd_B2.txt (KR13, KR16,
KR17, KR20, KR21, KR22) dan Codebook/ifls2014_hhd_BK.txt (KRK02*, KRK06,
KRK08, KRK09).

Kesetiaan aturan terhadap dataset latih diverifikasi oleh uji_derivasi.py
(hasil: LAPORAN_VERIFIKASI.md).
"""

# ============================================================================
# Aturan klasifikasi (salinan persis dari script ekstraksi)
# ============================================================================

# air: 2 = layak, 1 = ambigu (IFLS tak cukup detail utk aturan JMP), 0 = tidak
AIR_LAYAK = {1, 2, 5, 10}       # pipa, sumur pompa(asumsi), hujan, kemasan
AIR_RAGU  = {3, 4, 8}           # sumur timba, mata air, bak penampungan
AIR_TAK   = {6, 7}              # sungai, kolam (air permukaan)

# jamban: 2 = basic, 1 = limited (layak tapi berbagi), 0 = tidak layak/terbuka
JAMBAN_BASIC    = {1}
JAMBAN_LIMITED  = {3, 4}
JAMBAN_TAKLAYAK = {2}
JAMBAN_TERBUKA  = {5, 6, 7, 9, 10, 11}
JAMBAN = {**{k: 2 for k in JAMBAN_BASIC},
          **{k: 1 for k in JAMBAN_LIMITED},
          **{k: 0 for k in JAMBAN_TAKLAYAK | JAMBAN_TERBUKA}}

LIMBAH_AMAN = {1, 3, 8}         # selokan mengalir, lubang permanen, lubang
SAMPAH_AMAN = {1, 2, 5}         # tempat sampah, dibakar, lubang

# pendidikan: ar16 -> jenjang 0-4 (kode asli IFLS tidak berurutan)
JENJANG = {
    1: 0,  90: 0,
    2: 1,  11: 1,  72: 1,
    3: 2,  4: 2,   12: 2,  73: 2,
    5: 3,  6: 3,   15: 3,  74: 3,
    13: 4, 60: 4,  61: 4,  62: 4,  63: 4,
}

# ============================================================================
# Label manusiawi tiap kode mentah (dari Codebook)
# ============================================================================

LABEL_AIR_MINUM = {                       # KR13
    1:  "Air ledeng / PAM (pipa)",
    2:  "Sumur dengan pompa",
    3:  "Sumur timba (tanpa pompa)",
    4:  "Mata air",
    5:  "Air hujan",
    6:  "Sungai / kali",
    7:  "Kolam",
    8:  "Bak penampungan air",
    10: "Air kemasan / isi ulang",
}
# KR17 tidak punya kode 10 (kemasan), tak ada rumah tangga yang mandi air kemasan
LABEL_AIR_MCK = {k: v for k, v in LABEL_AIR_MINUM.items() if k != 10}

LABEL_JAMBAN = {                          # KR20
    1:  "Jamban sendiri dengan septic tank",
    2:  "Jamban sendiri tanpa septic tank",
    3:  "Jamban bersama / komunal",
    4:  "Jamban umum",
    5:  "Sungai / selokan / parit",
    6:  "Kebun / ladang",
    7:  "Got / saluran pembuangan",
    9:  "Kolam",
    10: "Kandang ternak",
    11: "Laut",
}

LABEL_LIMBAH = {                          # KR21
    1:  "Selokan / got yang mengalir",
    2:  "Selokan / got tergenang",
    3:  "Lubang penampungan permanen",
    4:  "Dibuang ke sungai",
    5:  "Dibuang di pekarangan",
    7:  "Kolam",
    8:  "Lubang galian",
    9:  "Sawah",
    11: "Laut",
}

LABEL_SAMPAH = {                          # KR22
    1: "Dibuang di tempat sampah, diangkut petugas",
    2: "Dibakar",
    3: "Dibuang ke sungai / kali",
    4: "Dibuang di pekarangan, dibiarkan membusuk",
    5: "Dibuang ke lubang",
    7: "Dibuang ke hutan",
    8: "Dibuang ke laut",
    9: "Dibuang ke sawah",
}

LABEL_LANTAI = {                          # KRK08 (kode asli, sebelum dibalik)
    1: "Keramik / marmer / granit / batu",
    2: "Ubin / tegel / teraso",
    3: "Semen / bata merah",
    4: "Kayu / papan",
    5: "Bambu",
    6: "Tanah",
}

LABEL_DINDING = {                         # KRK09 (kode asli, sebelum dibalik)
    1: "Tembok (semen / batako)",
    2: "Kayu / papan / triplek",
    3: "Bambu / bilik anyaman",
}

# pendidikan ditanyakan langsung per jenjang (peta JENJANG many-to-one,
# pemakai tahu jenjangnya sendiri tanpa perlu kode ar16)
LABEL_PENDIDIKAN = {
    0: "Tidak sekolah / TK",
    1: "SD / MI sederajat",
    2: "SMP / MTs sederajat",
    3: "SMA / SMK / MA sederajat",
    4: "Perguruan tinggi",
}

# kolom formulir yang sudah berformat model (tidak butuh penerjemahan)
PASSTHROUGH = [
    "ibu_tinggi", "ayah_tinggi", "ibu_usia_konsepsi", "jumlah_kamar",
    "ibu_pendidikan", "ayah_pendidikan",
    "urban", "ibu_bekerja", "ayah_merokok",
    "limbah_dekat_rumah", "tumpukan_sampah", "air_tergenang",
    "dekat_kandang", "ventilasi_cukup", "dapur_kamar_menyatu",
]


def gol_air(kode):
    """Klasifikasi sumber air (salinan gol_air di TAHAP 9 Collect_Data)."""
    if kode in AIR_LAYAK:
        return 2
    if kode in AIR_RAGU:
        return 1
    if kode in AIR_TAK:
        return 0
    return None


def turunkan(jawaban_mentah: dict) -> dict:
    """Ubah jawaban formulir menjadi 23 kolom yang dipahami model.

    Kunci masukan:
      - semua nama di PASSTHROUGH (sudah numerik siap pakai)
      - kr13         : kode sumber air minum
      - air_mck_sama : 1 kalau sumber air mandi/cuci = air minum (KR16)
      - kr17         : kode sumber air mandi/cuci (diabaikan jika sama)
      - kr20, kr21, kr22 : kode jamban, limbah, sampah
      - krk08, krk09     : kode bahan lantai & dinding (asli, belum dibalik)
    """
    kr13 = jawaban_mentah["kr13"]
    sama = int(jawaban_mentah["air_mck_sama"])
    # pola lompat KR16: kalau sumbernya sama, kode MCK = kode air minum
    kode_mck = kr13 if sama == 1 else jawaban_mentah["kr17"]

    hasil = {k: jawaban_mentah[k] for k in PASSTHROUGH}
    hasil.update({
        "air_minum_layak": gol_air(kr13),
        "air_mck_layak":   gol_air(kode_mck),
        "air_mck_sama":    sama,
        "jamban_layak":    JAMBAN[jawaban_mentah["kr20"]],
        "limbah_aman":     1 if jawaban_mentah["kr21"] in LIMBAH_AMAN else 0,
        "sampah_aman":     1 if jawaban_mentah["kr22"] in SAMPAH_AMAN else 0,
        # arah dibalik saat preprocessing: angka besar = kondisi lebih baik
        "bahan_lantai":    7 - jawaban_mentah["krk08"],
        "bahan_dinding":   4 - jawaban_mentah["krk09"],
    })
    return hasil
