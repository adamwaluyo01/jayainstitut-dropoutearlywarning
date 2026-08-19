# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan

## Business Understanding
Jaya Jaya Institut merupakan institusi pendidikan tinggi yang ingin menekan kasus mahasiswa yang tidak menyelesaikan studi. Proyek ini membangun pendekatan berbasis data untuk memonitor faktor yang berkaitan dengan dropout dan menyediakan early warning machine learning agar tim akademik dapat memprioritaskan intervensi.

### Permasalahan Bisnis
1. Belum ada mekanisme berbasis data untuk memonitor faktor penting yang berkaitan dengan dropout.
2. Institusi membutuhkan identifikasi mahasiswa berisiko sedini mungkin agar dukungan akademik/finansial dapat diberikan lebih cepat.
3. Tim akademik membutuhkan prototype yang mudah digunakan dan dapat diakses secara remote.

### Cakupan Proyek
- EDA dan analisis faktor dropout.
- Data preparation untuk dashboard dan machine learning.
- Binary classification **`Dropout` vs `Graduate`** menggunakan hanya mahasiswa dengan outcome final. Status `Enrolled` dikeluarkan dari training karena outcome akhirnya belum diketahui.
- Business dashboard menggunakan Looker Studio.
- Prototype prediction menggunakan Streamlit dan deployment ke Streamlit Community Cloud.
- Rekomendasi action items berbasis temuan data.

### Persiapan
Sumber data: dataset resmi Dicoding **Students' Performance**  
`https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance`

Dataset yang digunakan berisi **4.424 baris**, **36 fitur**, dan satu target/status. Tidak terdapat missing value maupun baris duplikat pada file yang dianalisis.

Setup environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Untuk menjalankan notebook:

```bash
jupyter notebook notebook.ipynb
```

Notebook memprioritaskan `data/data.csv`, dan dapat menggunakan dataset resmi dari GitHub sebagai fallback.

## Business Dashboard
Business dashboard digunakan untuk memonitor tingkat dropout dan faktor penting seperti status pembayaran biaya kuliah, status debtor, program studi, kelompok usia, dan performa semester pertama.

**Link Looker Studio:** `https://datastudio.google.com/reporting/c3ac6a82-7495-4d5a-b8be-8cc0bef83493`

KPI utama:
- Total Student: **4.424**
- Total Dropout: **1.421**
- Dropout Rate: **32,12%**
- Total Graduate: **2.209**

Visual yang disarankan:
- Dropout Rate menurut Tuition Status
- Dropout Rate menurut Debtor Status
- Dropout Rate menurut Course
- Dropout Rate menurut Age Group
- Dropout Rate menurut First Semester Approval Band

File sumber dashboard dibuat oleh notebook pada:
`data/dashboard_student_performance.csv`

Screenshot dashboard final harus diberi nama:
`adam_waluyo_7laR-dashboard.png`

## Menjalankan Sistem Machine Learning
Notebook menyimpan artefak model ke:

```text
model/dropout_model.joblib
```

Jalankan prototype secara lokal:

```bash
streamlit run app.py
```

**Link Streamlit Community Cloud:** `https://jayainstitut.streamlit.app`

Prototype meminta informasi yang tersedia setelah semester pertama dan menampilkan probabilitas serta tier risiko dropout. Hasil prediksi merupakan **decision-support**, bukan keputusan otomatis terhadap mahasiswa.

### Ringkasan Model
> **Validasi target:** training hanya menggunakan mahasiswa dengan outcome final `Graduate` dan `Dropout`. Status `Enrolled` tidak dimasukkan ke training maupun test karena outcome akhirnya belum diketahui.

Dataset modeling difilter menjadi **3.630 mahasiswa dengan outcome final**: 1.421 `Dropout` dan 2.209 `Graduate`. Sebanyak 794 mahasiswa `Enrolled` **tidak digunakan untuk training**. Target dipetakan menjadi `Dropout = 1` dan `Graduate = 0`. Model hanya menggunakan fitur pendaftaran, faktor finansial/demografi, serta performa semester pertama.

Dengan train-test split 80:20 (`random_state=42`) dan threshold klasifikasi **0,50**, model terpilih adalah **Random Forest** dengan performa test set:

| Metrik | Hasil |
|---|---:|
| Accuracy | **89,39%** |
| Precision - Dropout | **86,32%** |
| Recall - Dropout | **86,62%** |
| F1-score - Dropout | **86,47%** |
| ROC-AUC | **94,83%** |

Confusion matrix: **TN=403, FP=39, FN=38, TP=246**.

## Conclusion
Dropout rate pada dataset adalah **32,12%** (1.421 dari 4.424 mahasiswa). Analisis menunjukkan beberapa segmen yang perlu mendapat perhatian lebih:

- Tuition tidak up to date: **86,55% dropout**, dibanding **24,74%** untuk tuition up to date.
- Debtor: **62,03% dropout**, dibanding **28,28%** untuk non-debtor.
- Tanpa beasiswa: **38,71% dropout**, dibanding **12,19%** untuk penerima beasiswa.
- Approval semester pertama 0-25%: **78,51% dropout**, sedangkan approval 76-100%: **11,79%**.

Temuan tersebut bersifat asosiasi/deskriptif, bukan bukti sebab-akibat. Dashboard digunakan untuk monitoring tingkat institusi/segmen, sedangkan model digunakan untuk membantu prioritas intervensi pada level mahasiswa.

### Rekomendasi Action Items
1. Prioritaskan monitoring mahasiswa dengan biaya kuliah tidak up to date atau status debtor, lalu arahkan ke dukungan finansial yang sesuai.
2. Buat trigger intervensi untuk mahasiswa dengan approval rate/nilai semester pertama rendah melalui tutoring, konseling akademik, dan review beban studi.
3. Gunakan tier risiko sebagai antrean prioritas follow-up dosen wali/konselor, bukan sebagai keputusan otomatis.
4. Pantau performa model dari waktu ke waktu dan lakukan retraining ketika kualitas prediksi menurun atau profil mahasiswa berubah.
5. Gabungkan sinyal model dengan asesmen manusia dan data kualitatif agar intervensi tetap adil dan kontekstual.
