import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Jaya Jaya Institut - Dropout Early Warning",
    page_icon="🎓",
    layout="wide",
)

st.markdown("""
<style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px;}
    .hero {padding: 1.4rem 1.6rem; border-radius: 16px; background: linear-gradient(135deg,#eef5ff,#f8fbff); border: 1px solid #dbe7f7; margin-bottom: 1rem;}
    .hero h1 {margin:0; font-size:2rem; color:#1f2937;}
    .hero p {margin:.35rem 0 0 0; color:#506070;}
    .risk-card {padding:1.4rem; border-radius:16px; border:1px solid #dde3ea; background:#ffffff; box-shadow:0 2px 8px rgba(20,40,80,.05);}
    .small-note {color:#667085; font-size:.9rem;}
</style>
""", unsafe_allow_html=True)

DATA_URL = "https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/main/students_performance/data.csv"
MODEL_PATH = Path("model/dropout_model.joblib")
THRESHOLD = 0.50

COURSE_MAP = {
    33: 'Biofuel Production Technologies',
    171: 'Animation and Multimedia Design',
    8014: 'Social Service (evening)',
    9003: 'Agronomy',
    9070: 'Communication Design',
    9085: 'Veterinary Nursing',
    9119: 'Informatics Engineering',
    9130: 'Equinculture',
    9147: 'Management',
    9238: 'Social Service',
    9254: 'Tourism',
    9500: 'Nursing',
    9556: 'Oral Hygiene',
    9670: 'Advertising & Marketing Management',
    9773: 'Journalism & Communication',
    9853: 'Basic Education',
    9991: 'Management (evening)',
}
COURSE_NAME_TO_CODE = {v: k for k, v in COURSE_MAP.items()}

FEATURES = [
    'Course','Daytime_evening_attendance','Admission_grade','Displaced','Debtor',
    'Tuition_fees_up_to_date','Gender','Scholarship_holder','Age_at_enrollment',
    'International','Curricular_units_1st_sem_enrolled',
    'Curricular_units_1st_sem_evaluations','Curricular_units_1st_sem_approved',
    'Curricular_units_1st_sem_grade','Curricular_units_1st_sem_without_evaluations'
]
CATEGORICAL_FEATURES = [
    'Course','Daytime_evening_attendance','Displaced','Debtor',
    'Tuition_fees_up_to_date','Gender','Scholarship_holder','International'
]
NUMERIC_FEATURES = [c for c in FEATURES if c not in CATEGORICAL_FEATURES]

@st.cache_resource(show_spinner="Menyiapkan model early warning...")
def load_or_train_model():
    if MODEL_PATH.exists():
        artifact = joblib.load(MODEL_PATH)
        return artifact

    # Fallback untuk deployment awal jika artefak model belum dikomit.
    df = pd.read_csv(DATA_URL, sep=';')
    # Gunakan hanya outcome final untuk training: Graduate vs Dropout.
    # Enrolled belum memiliki outcome akhir dan tidak boleh dianggap sebagai kelas negatif.
    model_df = df[df['Status'].isin(['Dropout', 'Graduate'])].copy()
    X = model_df[FEATURES].copy()
    y = (model_df['Status'] == 'Dropout').astype(int)

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])
    pre = ColumnTransformer([
        ('num', num_pipe, NUMERIC_FEATURES),
        ('cat', cat_pipe, CATEGORICAL_FEATURES),
    ])
    model = Pipeline([
        ('preprocessor', pre),
        ('model', RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2,
            class_weight='balanced', random_state=42, n_jobs=-1
        )),
    ])
    model.fit(X, y)
    return {
        'model': model,
        'threshold': THRESHOLD,
        'features': FEATURES,
        'model_name': 'Random Forest (deployment fallback)',
        'training_statuses': ['Graduate', 'Dropout'],
        'target_definition': 'Graduate=0, Dropout=1',
    }

artifact = load_or_train_model()
model = artifact['model']
threshold = float(artifact.get('threshold', THRESHOLD))
model_name = artifact.get('model_name', 'Dropout Classifier')

st.markdown("""
<div class="hero">
  <h1>🎓 Student Dropout Early Warning</h1>
  <p>Prototype decision-support untuk membantu Jaya Jaya Institut memprioritaskan mahasiswa yang membutuhkan dukungan lebih awal.</p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.35, 1], gap="large")

with left:
    st.subheader("Data Mahasiswa")
    st.caption("Gunakan data yang tersedia setelah semester pertama. Semua input hanya digunakan untuk estimasi risiko, bukan keputusan otomatis.")

    c1, c2 = st.columns(2)
    with c1:
        course_name = st.selectbox("Program studi", list(COURSE_NAME_TO_CODE.keys()), index=10)
        attendance = st.selectbox("Waktu kuliah", ["Daytime", "Evening"])
        age = st.number_input("Usia saat pendaftaran", min_value=16, max_value=80, value=20, step=1)
        admission = st.number_input("Admission grade (0-200)", min_value=0.0, max_value=200.0, value=130.0, step=1.0)
        gender = st.selectbox("Gender (sesuai encoding dataset)", ["Female", "Male"])
        international = st.selectbox("Mahasiswa internasional", ["No", "Yes"])
        displaced = st.selectbox("Displaced", ["No", "Yes"])
    with c2:
        debtor = st.selectbox("Status debtor", ["No", "Yes"])
        tuition = st.selectbox("Biaya kuliah up to date", ["Yes", "No"])
        scholarship = st.selectbox("Penerima beasiswa", ["No", "Yes"])
        enrolled = st.number_input("Unit semester 1 - enrolled", min_value=0, max_value=30, value=6, step=1)
        evaluations = st.number_input("Unit semester 1 - evaluations", min_value=0, max_value=40, value=8, step=1)
        approved = st.number_input("Unit semester 1 - approved", min_value=0, max_value=30, value=5, step=1)
        grade = st.number_input("Nilai rata-rata semester 1 (0-20)", min_value=0.0, max_value=20.0, value=12.0, step=0.1)
        without_eval = st.number_input("Unit tanpa evaluasi semester 1", min_value=0, max_value=30, value=0, step=1)

    predict = st.button("Analisis Risiko Dropout", type="primary", use_container_width=True)

with right:
    st.subheader("Hasil Early Warning")
    if predict:
        row = pd.DataFrame([{
            'Course': COURSE_NAME_TO_CODE[course_name],
            'Daytime_evening_attendance': 1 if attendance == 'Daytime' else 0,
            'Admission_grade': admission,
            'Displaced': 1 if displaced == 'Yes' else 0,
            'Debtor': 1 if debtor == 'Yes' else 0,
            'Tuition_fees_up_to_date': 1 if tuition == 'Yes' else 0,
            'Gender': 1 if gender == 'Male' else 0,
            'Scholarship_holder': 1 if scholarship == 'Yes' else 0,
            'Age_at_enrollment': age,
            'International': 1 if international == 'Yes' else 0,
            'Curricular_units_1st_sem_enrolled': enrolled,
            'Curricular_units_1st_sem_evaluations': evaluations,
            'Curricular_units_1st_sem_approved': approved,
            'Curricular_units_1st_sem_grade': grade,
            'Curricular_units_1st_sem_without_evaluations': without_eval,
        }])
        prob = float(model.predict_proba(row[FEATURES])[:, 1][0])

        if prob >= 0.65:
            tier, icon = "TINGGI", "🔴"
        elif prob >= threshold:
            tier, icon = "SEDANG", "🟠"
        else:
            tier, icon = "RENDAH", "🟢"

        st.markdown('<div class="risk-card">', unsafe_allow_html=True)
        st.metric("Probabilitas dropout", f"{prob:.1%}")
        st.progress(min(max(prob, 0.0), 1.0))
        st.markdown(f"### {icon} Risiko {tier}")
        st.caption(f"Threshold early warning: {threshold:.0%} | Model: {model_name}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("#### Rekomendasi tindak lanjut")
        actions = []
        if tuition == 'No' or debtor == 'Yes':
            actions.append("Tinjau hambatan finansial dan opsi bantuan/pembayaran bersama mahasiswa.")
        approval_rate = (approved / enrolled) if enrolled > 0 else 0
        if approval_rate < 0.5 or grade < 10:
            actions.append("Prioritaskan konseling akademik, tutoring, dan review beban studi.")
        if without_eval > 0:
            actions.append("Periksa penyebab unit tanpa evaluasi dan potensi masalah kehadiran/administratif.")
        if prob >= threshold:
            actions.append("Masukkan mahasiswa ke daftar follow-up dosen wali/konselor dan lakukan asesmen manusia.")
        if not actions:
            actions.append("Lanjutkan monitoring rutin dan pertahankan dukungan akademik yang tersedia.")
        for item in actions:
            st.write("•", item)
    else:
        st.info("Isi data mahasiswa lalu klik **Analisis Risiko Dropout**.")

st.divider()
st.markdown("**Catatan etis:** prediksi ini adalah alat bantu prioritas. Jangan gunakan skor sebagai satu-satunya dasar untuk membatasi akses, memberikan sanksi, atau membuat keputusan akademik terhadap mahasiswa.")
