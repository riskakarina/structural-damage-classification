# Tugas Kaggle Competition
Tugas Praktikum Machine Learning untuk mengklasifikasikan tingkat kerusakan bangunan (`damage_grade`) menggunakan algoritma **LightGBM** dan teknik *Target Encoding Out-of-Fold*.

## Latar Belakang
Gempa bumi merupakan salah satu bencana alam yang paling destruktif dan seringkali memicu kerusakan infrastruktur skala besar dalam waktu singkat. Penilaian kerentanan bangunan secara akurat sangat krusial untuk mitigasi risiko bencana, perencanaan tata kota, serta evaluasi keselamatan struktur bangunan pasca-bencana. 

## Tujuan Proyek
Tujuan dari proyek kompetisi Machine Learning ini adalah membangun model prediktif yang handal untuk mengklasifikasikan tingkat kerusakan bangunan (*damage_grade*) secara presisi. Model ini dirancang untuk mempelajari pola dari berbagai atribut fisik bangunan, material konstruksi, serta hierarki wilayah geografis guna memprediksi tingkat keparahan kerusakan yang dialami oleh suatu bangunan.

---

## Data

Dataset yang digunakan dalam proyek ini bersumber dari data survei kerusakan bangunan pasca-bencana. Data dibagi menjadi data latih (*train*) dan data uji (*test*).

### Rincian Fitur Utama:
* **Identifikasi & Wilayah:** 
  * `building_id`: ID unik yang merepresentasikan setiap bangunan.
  * `district_id`, `vdcmun_id`, `ward_id`: Informasi administratif wilayah lokasi bangunan.
* **Atribut Fisik & Geometri Bangunan:**
  * `count_floors_pre_eq`: Jumlah lantai sebelum gempa terjadi.
  * `age_building`: Usia bangunan (tahun).
  * `plinth_area_sq_ft`: Luas tapak bangunan dalam satuan kaki persegi ($sq\_ft$).
  * `height_ft_pre_eq`: Tinggi bangunan sebelum gempa.
* **Fitur Kategorikal & Material (Superstructure):**
  * `foundation_type`, `roof_type`, `ground_floor_type`, `other_floor_type`: Jenis material pondasi, atap, dan lantai.
  * `land_surface_condition`: Kondisi permukaan tanah di sekitar bangunan.
  * `position` & `plan_configuration`: Posisi bangunan dan konfigurasi denah.
  * `superstructure`: Informasi keberadaan berbagai jenis material penguat struktur (seperti batu bata, kayu, semen, dll).
* **Target Variabel:**
  * `damage_grade`: Tingkat kerusakan bangunan yang dikategorikan dalam skala 1 hingga 5 (atau `Grade 1` sampai `Grade 5`).

---

## Metodologi Pemodelan

Pendekatan pemodelan dirancang secara modular dan terstruktur untuk memaksimalkan performa evaluasi, khususnya dalam mengoptimalkan metrik **Macro F1-Score**.

### 1. Preprocessing & Feature Engineering (`preprocess.py` & `feature_engineering.py`)
* **Pembersihan Data:** Menangani nilai anomali atau *placeholder* pada data usia bangunan (misalnya nilai 900+ dikonversi menjadi *missing value* / `NaN` dan diberi indikator biner `age_missing`).
* **Rekayasa Fitur Geometri:** Mengekstrak informasi tambahan untuk membantu model memahami proporsi fisik bangunan, seperti:
  * `area_per_floor` (Rasio luas lantai terhadap jumlah lantai)
  * `floor_height` (Rasio tinggi bangunan terhadap jumlah lantai)
  * `vol` (Estimasi volume bangunan: luas tapak $\times$ tinggi)
  * `hxa` (Rasio geometris tinggi terhadap akar kuadrat luas tapak)
* **Ekstraksi Hierarki ID (`building_id`):** Memecah string `building_id` menjadi prefiks hierarki wilayah (panjang karakter 6, 8, 9, dan 10) karena bangunan yang berada dalam satu klaster wilayah cenderung memiliki pola kerusakan yang serupa.
* **Target Encoding Out-of-Fold (OOF):** Menerapkan teknik *Target Encoding* dengan *smoothing* pada fitur-fitur kategori berdimensi tinggi (seperti prefiks wilayah dan ward ID) tanpa mengalami *data leakage* menggunakan strategi *Stratified K-Fold*.

### 2. Algoritma & Validasi Model (`train.py`)
* **Algoritma:** Menggunakan **LightGBM (Light Gradient Boosting Machine)**, yaitu algoritma *ensemble* berbasis pohon yang sangat cepat dan akurat untuk data tabular multikelas. *(Catatan: Proyek ini tidak menggunakan Neural Network).*
* **Strategi Validasi:** Menggunakan pembagian data *Holdout Validation* yang dikombinasikan dengan *Early Stopping* untuk mencegah *overfitting*.

### 3. Evaluasi & Penyetelan Bobot Kelas (`Search Class Weights`)
* Karena kompetisi ini berfokus pada metrik **Macro F1-Score** (yang sensitif terhadap ketimpangan distribusi kelas), dilakukan optimasi pasca-prediksi menggunakan metode *coordinate ascent*. Metode ini mencari bobot optimal untuk probabilitas setiap kelas agar prediksi akhir lebih seimbang dan skor Macro F1 mencapai titik maksimal.

---

## Struktur Repositori
- `data/`: Menyimpan dataset mentah (train, test, sample submission).
- `notebooks/`: Tempat eksperimen eksplorasi data (EDA) dan modeling iteratif.
- `scripts/`: Berisi kode modular (`preprocess.py`, `feature_engineering.py`, `train.py`, `predict.py`).
- `models/`: Menyimpan file biner model hasil training.
- `submissions/`: Menyimpan hasil akhir prediksi format CSV untuk dikirim ke sistem.

## Cara Menjalankan Proyek
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
