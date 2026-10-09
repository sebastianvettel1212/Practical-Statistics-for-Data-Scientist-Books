# Practical Statistics for Data Scientists — Bab 1–7
Python Code that I reproduce from Practical Statistics for Data Scientists books (O'Reilly).

<img width="339" height="445" alt="image" src="https://github.com/user-attachments/assets/565d456c-7a09-47fc-820e-a30bd4b2190e" />

----------------------------------------------------


|           |                   |
| --------- | ----------------- |
| **Nama**  | Zacky Yusup Hakim |
| **NIM**   | 101032300183      |
| **Kelas** | BS1TK-47-REG-G13  |

Repositori ini berisi rangkuman dan implementasi kode Python untuk **Bab 1 sampai Bab 7** dari buku *Practical Statistics for Data Scientists* (Peter Bruce, Andrew Bruce & Peter Gedeck, O'Reilly, edisi ke-2, 2020). Setiap bab disajikan dalam satu Jupyter Notebook yang menggabungkan **penjelasan teori (Bahasa Indonesia)**, **reproduksi kode** dari buku, **interpretasi hasil**, dan **ringkasan (Key Ideas)**. Seluruh output sudah tersimpan di notebook, sehingga dapat dibaca tanpa dijalankan ulang.

Buku ini menjembatani **statistika klasik** dan **praktik data science**: mulai dari eksplorasi data, sampling, eksperimen dan uji signifikansi, hingga regresi, klasifikasi, *statistical machine learning*, dan *unsupervised learning*.

Repositori ini dibuat untuk tugas **Enrichment for Machine Learning Classes: Code Reproduction + Theoretical Deep-Dive**.

---

## Daftar Isi

1. [Struktur Proyek](#struktur-proyek)
2. [Instalasi & Cara Menjalankan](#instalasi--cara-menjalankan)
3. [Bab 1 — Exploratory Data Analysis](#bab-1--exploratory-data-analysis)
4. [Bab 2 — Data and Sampling Distributions](#bab-2--data-and-sampling-distributions)
5. [Bab 3 — Statistical Experiments and Significance Testing](#bab-3--statistical-experiments-and-significance-testing)
6. [Bab 4 — Regression and Prediction](#bab-4--regression-and-prediction)
7. [Bab 5 — Classification](#bab-5--classification)
8. [Bab 6 — Statistical Machine Learning](#bab-6--statistical-machine-learning)
9. [Bab 7 — Unsupervised Learning](#bab-7--unsupervised-learning)
10. [Dataset](#dataset)
11. [Library yang Digunakan](#library-yang-digunakan)
12. [Referensi](#referensi)

---

## Struktur Proyek

```
Practical Statistics for Data Scientists/
├── README.md
├── data/                      (opsional, lihat bagian Dataset)
└── notebooks/
    ├── Chapter_01_Exploratory_Data_Analysis.ipynb
    ├── Chapter_02_Data_and_Sampling_Distributions.ipynb
    ├── Chapter_03_Statistical_Experiments_and_Significance_Testing.ipynb
    ├── Chapter_04_Regression_and_Prediction.ipynb
    ├── Chapter_05_Classification.ipynb
    ├── Chapter_06_Statistical_Machine_Learning.ipynb
    └── Chapter_07_Unsupervised_Learning.ipynb
```

## Instalasi & Cara Menjalankan

```
pip install pandas numpy scipy statsmodels scikit-learn seaborn matplotlib wquantiles
pip install imbalanced-learn pygam dmba xgboost prince
cd notebooks
jupyter notebook
```

Seluruh dataset berasal dari repositori resmi buku ([gedeck/practical-statistics-for-data-scientists](https://github.com/gedeck/practical-statistics-for-data-scientists)). Tiap notebook memakai fungsi `path_data()` yang **otomatis memakai file lokal di folder `data/` bila ada, atau mengunduhnya dari GitHub bila tidak ada**, sehingga notebook dapat langsung dijalankan (termasuk di Google Colab) tanpa menyiapkan data secara manual.

Beberapa penyesuaian terhadap kode buku, karena versi library yang lebih baru (pandas 3, scikit-learn 1.8, XGBoost 3), dicatat di dalam notebook. Sel yang berlabel **"Tidak ada di buku"** adalah demonstrasi tambahan (mis. verifikasi rumus secara manual, kurva lift, bagging manual) yang melengkapi teori.

---

## Bab 1 — Exploratory Data Analysis

📓 [`Chapter_01_Exploratory_Data_Analysis.ipynb`](Chapter_01_Exploratory_Data_Analysis.ipynb)

Langkah pertama dan terpenting dalam proyek data science: **melihat datanya**. Bab ini berangkat dari gagasan John Tukey tentang *exploratory data analysis* dan membahas cara meringkas serta memvisualisasikan data.

**Materi:**

1. Elemen data terstruktur (numerik, kategorikal, biner, ordinal)
2. Data *rectangular* (data frame, fitur, record, index)
3. Estimasi lokasi: *mean*, *trimmed mean*, *weighted mean*, median, *weighted median*, outlier
4. Estimasi variabilitas: variansi, standar deviasi, MAD, range, persentil, IQR
5. Mengeksplorasi distribusi: boxplot, tabel frekuensi, histogram, density plot
6. Data biner dan kategorikal: mode, *expected value*, bar chart
7. Korelasi: koefisien Pearson, matriks korelasi, scatterplot
8. Dua variabel atau lebih: *hexagonal binning*, kontur, tabel kontingensi, violin plot, *conditioning* (facet)

**Rangkuman:**

- **Mean sensitif terhadap outlier**; median dan *trimmed mean* lebih *robust*. Pada populasi negara bagian AS, mean (6.162.876) lebih besar daripada *trimmed mean* (4.783.697) dan median karena ditarik negara bagian berpopulasi besar.
- Estimasi lokasi harus mempertimbangkan bobot bila kelompok tidak sama besar (*weighted mean* tingkat pembunuhan ≈ 4,45 vs mean biasa ≈ 4,07).
- Standar deviasi sensitif terhadap outlier; **MAD** dan **IQR** lebih robust.
- Untuk data besar, scatterplot tidak memadai; gunakan *hexagonal binning* atau plot kontur. *Conditioning* dapat mengungkap struktur yang tersembunyi.

---

## Bab 2 — Data and Sampling Distributions

📓 [`Chapter_02_Data_and_Sampling_Distributions.ipynb`](Chapter_02_Data_and_Sampling_Distributions.ipynb)

Bagaimana kita menarik kesimpulan tentang populasi dari **sampel**, dan seberapa besar ketidakpastiannya.

**Materi:**

1. *Random sampling* dan *sample bias* (kasus *Literary Digest* 1936)
2. *Selection bias*, *data snooping*, *vast search effect*, *regression to the mean*
3. Distribusi sampling sebuah statistik, *Central Limit Theorem*, *standard error*
4. *The Bootstrap*
5. *Confidence interval*
6. Distribusi normal, QQ-Plot, dan distribusi *long-tailed*
7. Distribusi-t Student
8. Distribusi binomial
9. Distribusi chi-square dan F
10. Distribusi Poisson, eksponensial, dan Weibull

**Rangkuman:**

- **Kualitas data lebih penting daripada kuantitas**; *random sampling* melindungi dari bias.
- *Standard error* = $s/\sqrt{n}$, sehingga presisi dua kali lipat membutuhkan sampel **empat kali lipat**. Hasil simulasi cocok dengan rumus.
- **Bootstrap** menghasilkan estimasi *standard error* dan *confidence interval* tanpa asumsi distribusi.
- Data riil sering **berekor panjang** (QQ-Plot saham Netflix menyimpang dari normal); jangan otomatis mengasumsikan normal.

---

## Bab 3 — Statistical Experiments and Significance Testing

📓 [`Chapter_03_Statistical_Experiments_and_Significance_Testing.ipynb`](Chapter_03_Statistical_Experiments_and_Significance_Testing.ipynb)

Merancang eksperimen dan menilai apakah efek yang teramati **nyata atau hanya kebetulan**.

**Materi:**

1. A/B testing
2. Uji hipotesis (hipotesis nol vs alternatif, satu arah vs dua arah)
3. *Resampling*: uji permutasi
4. Signifikansi statistik, *p-value*, *Type 1* dan *Type 2 error*
5. Uji-t
6. Pengujian berganda (*multiple testing*)
7. Derajat kebebasan
8. ANOVA dan statistik-F
9. Uji chi-square dan uji eksak Fisher
10. *Multi-arm bandit*
11. *Power* dan ukuran sampel

**Rangkuman:**

- Tetapkan **hipotesis dan statistik uji sebelum** melihat data; **pengacakan** adalah fondasi inferensi.
- **Uji permutasi** tidak membutuhkan asumsi distribusi dan hasilnya serupa dengan uji klasik (uji-t, F, chi-square).
- ***P-value* bukan peluang hipotesis nol benar** dan bukan ukuran pentingnya efek.
- Pengujian berganda menaikkan peluang *false positive*; pada 20 uji dengan α = 0,05, peluang minimal satu hasil "signifikan" semu mencapai ≈ 64%.
- *Multi-arm bandit* memusatkan lalu lintas ke alternatif terbaik; analisis *power* menentukan ukuran sampel agar efek kecil dapat terdeteksi.

---

## Bab 4 — Regression and Prediction

📓 [`Chapter_04_Regression_and_Prediction.ipynb`](Chapter_04_Regression_and_Prediction.ipynb)

Regresi untuk **memprediksi nilai numerik** dan **menjelaskan hubungan** antar variabel, dengan contoh harga rumah di King County.

**Materi:**

1. Regresi linier sederhana (garis kuadrat terkecil, residual)
2. Regresi linier berganda (RMSE, RSE, R², *t-statistic*)
3. Prediksi dengan regresi (bahaya ekstrapolasi, interval kepercayaan vs interval prediksi)
4. Variabel faktor dalam regresi (*dummy*, *reference coding*, faktor dengan banyak level, faktor ordinal)
5. Menginterpretasi persamaan regresi (korelasi antar prediktor, multikolinearitas, *confounding*, interaksi)
6. Diagnostik regresi (outlier, nilai berpengaruh, heteroskedastisitas, plot residual parsial)
7. Regresi polinomial dan spline (*GAM*)

**Rangkuman:**

- Koefisien regresi diinterpretasi **dengan variabel lain dianggap tetap**; korelasi antar prediktor dan variabel pengganggu (*confounder*) dapat membalik tanda koefisien.
- **Diagnostik** (outlier, nilai berpengaruh, heteroskedastisitas) sama pentingnya dengan ukuran kecocokan model; fokus pada **prediksi** bukan hanya R².
- Hubungan non-linear ditangani dengan **polinomial, spline, atau GAM**.

---

## Bab 5 — Classification

📓 [`Chapter_05_Classification.ipynb`](Chapter_05_Classification.ipynb)

Memprediksi **kategori** (mis. pinjaman *default* atau *paid off*) memakai data Lending Club.

**Materi:**

1. Naive Bayes (Teorema Bayes, independensi bersyarat)
2. Analisis diskriminan (LDA, fungsi diskriminan linear)
3. Regresi logistik (*logit*, *odds*, *odds ratio*, GLM, residual parsial dan spline)
4. Mengevaluasi model klasifikasi (*confusion matrix*, *precision*, *recall*, *specificity*, kurva ROC, AUC, *lift*)
5. Strategi untuk data tak seimbang (*undersampling*, *oversampling*, pembobotan, SMOTE/ADASYN, klasifikasi berbasis biaya)
6. Menjelajahi prediksi (perbandingan batas keputusan pohon, LDA, regresi logistik, dan GAM)

**Rangkuman:**

- Koefisien regresi logistik adalah **log odds ratio**; probabilitas hasil verifikasi manual dengan fungsi sigmoid identik dengan `predict_proba`.
- Pada 45.342 pinjaman seimbang, regresi logistik mencapai **akurasi 63,65%** dan **AUC ≈ 0,69**; desil teratas memiliki *lift* ≈ 1,56.
- **Akurasi tidak cukup** pada data tak seimbang. Pada data penuh (18,9% *default*), model awal hanya memprediksi ≈ 1% pinjaman sebagai *default*; pembobotan menaikkannya menjadi ≈ 61%, SMOTE/ADASYN menjadi ≈ 28–29%.
- *Cutoff* 0,5 bukan hukum: bila *false negative* 10× lebih mahal, *cutoff* optimal bergeser ke ≈ 0,35.

---

## Bab 6 — Statistical Machine Learning

📓 [`Chapter_06_Statistical_Machine_Learning.ipynb`](Chapter_06_Statistical_Machine_Learning.ipynb)

Metode **berbasis data** yang belajar langsung dari data tanpa bentuk fungsi tetap: KNN dan keluarga ensemble pohon.

**Materi:**

1. K-Nearest Neighbors (metrik jarak, standardisasi, pemilihan K, KNN sebagai *feature engine*)
2. Model pohon (partisi rekursif, impurity Gini dan entropi, aturan berhenti dan *pruning*, pohon regresi)
3. Bagging dan *random forest* (*out-of-bag*, *variable importance*, hiperparameter)
4. Boosting (AdaBoost, *gradient boosting*, XGBoost, regularisasi, validasi silang untuk hiperparameter)

**Rangkuman:**

- **Standardisasi wajib** untuk KNN; tanpa itu, jarak didominasi variabel berskala besar seperti `revol_bal`. **K** dipilih lewat validasi silang.
- Pohon tunggal mudah dibaca (aturan "jika-maka") tetapi tidak stabil dan mudah *overfit*.
- ***Random forest*** menurunkan variansi lewat rata-rata banyak pohon yang beragam; skor **OOB** memberi estimasi galat tanpa data uji terpisah.
- ***Boosting* (XGBoost)** sering paling akurat tetapi mudah *overfit*; **regularisasi**, *learning rate* kecil, dan validasi silang diperlukan.

---

## Bab 7 — Unsupervised Learning

📓 [`Chapter_07_Unsupervised_Learning.ipynb`](Chapter_07_Unsupervised_Learning.ipynb)

Mengekstrak makna dari data **tanpa variabel hasil** yang diketahui: mereduksi dimensi dan mengelompokkan data.

**Materi:**

1. *Principal Components Analysis* (PCA), *scree plot*, dan *correspondence analysis*
2. K-Means *clustering* (memilih jumlah klaster)
3. *Hierarchical clustering* (dendrogram, metrik ketidakmiripan)
4. *Model-based clustering* (*Gaussian mixture*, pemilihan jumlah klaster dengan BIC)
5. Penskalaan dan variabel kategorikal (standardisasi, jarak Gower)

**Rangkuman:**

- **PCA** menggabungkan variabel yang berkorelasi menjadi sedikit komponen yang menangkap sebagian besar variansi.
- **K-Means** cepat dan skalabel tetapi memerlukan jumlah klaster yang ditentukan (*elbow method*); *hierarchical clustering* memberi gambaran struktur klaster secara menyeluruh; *model-based clustering* menyediakan landasan statistik.
- **Penskalaan** sangat memengaruhi hasil klaster, dan data campuran numerik-kategorikal memerlukan ukuran jarak khusus.

---

## Dataset

Seluruh dataset berasal dari repositori resmi buku dan diunduh otomatis oleh notebook bila tidak ada di folder `data/`.

| Bab | Dataset | Keterangan |
| --- | ------- | ---------- |
| 1 | `state.csv`, `dfw_airline.csv`, `airline_stats.csv`, `sp500_data.csv.gz`, `sp500_sectors.csv`, `kc_tax.csv.gz`, `lc_loans.csv` | Populasi dan tingkat pembunuhan per negara bagian AS, keterlambatan penerbangan, return saham S&P 500, pajak properti King County, pinjaman Lending Club |
| 2 | `loans_income.csv`, `sp500_data.csv.gz` | Pendapatan tahunan pemohon pinjaman, return saham |
| 3 | `web_page_data.csv`, `four_sessions.csv`, `click_rates.csv`, `imanishi_data.csv` | Waktu sesi halaman web, klik headline, distribusi digit |
| 4 | `house_sales.csv`, `LungDisease.csv` | Harga rumah King County, kapasitas paru-paru |
| 5 | `loan3000.csv`, `loan_data.csv.gz`, `full_train_set.csv.gz` | Pinjaman Lending Club (sampel 3.000, 45.342 seimbang, dan ≈120 ribu tak seimbang) |
| 6 | `loan200.csv`, `loan3000.csv`, `loan_data.csv.gz` | Pinjaman Lending Club |
| 7 | `sp500_data.csv.gz`, `sp500_sectors.csv`, `loan_data.csv.gz`, `housetasks.csv` | Return saham, data pinjaman, pembagian tugas rumah tangga (*correspondence analysis*) |

## Library yang Digunakan

`numpy` · `pandas` · `scipy` · `statsmodels` · `scikit-learn` · `seaborn` · `matplotlib` · `wquantiles` · `imbalanced-learn` · `pygam` · `dmba` · `xgboost` · `prince`

## Referensi

- Bruce, P., Bruce, A., & Gedeck, P. (2020). *Practical Statistics for Data Scientists* (2nd ed.). O'Reilly Media.
- Repositori kode resmi: <https://github.com/gedeck/practical-statistics-for-data-scientists>
