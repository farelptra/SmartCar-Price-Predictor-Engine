import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import random
# ---------- KONFIGURASI HALAMAN ----------
st.set_page_config(
    page_title='SmartCar - Prediksi Harga Mobil',
    page_icon='🚗',
    layout='centered',
    initial_sidebar_state='expanded'
)

st.markdown("""
<style>
.result-box {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}
.result-label { color: #a0aec0; font-size: 13px; letter-spacing: 2px; text-transform: uppercase; }
.result-price { color: #68d391; font-size: 48px; font-weight: 800; margin: 0.25rem 0; }
.result-note  { color: #718096; font-size: 12px; }

.badge-green { background:#c6f6d5; color:#276749; border-radius:20px; padding:3px 12px; font-size:12px; font-weight:600; }
.badge-blue  { background:#bee3f8; color:#2a69ac; border-radius:20px; padding:3px 12px; font-size:12px; font-weight:600; }

section[data-testid="stSidebar"] { background-color: #f8fafc; }

[data-testid="stMetricValue"] {
    font-size: 1.4rem !important;
}
[data-testid="stMetricValue"] > div, [data-testid="stMetricLabel"] > div, [data-testid="stMetricLabel"] p {
    white-space: normal !important;
    word-break: break-word !important;
}
</style>
""", unsafe_allow_html=True)


def car_balloons():
    html_code = """
    <style>
    .car-float {
        position: fixed;
        bottom: -50px;
        animation: floatUp 3s ease-in forwards;
        z-index: 999999;
        opacity: 0;
    }
    @keyframes floatUp {
        0% { bottom: -50px; opacity: 1; transform: translateX(0px) rotate(0deg); }
        50% { transform: translateX(20px) rotate(10deg); }
        100% { bottom: 100vh; opacity: 0; transform: translateX(-20px) rotate(-10deg); }
    }
    </style>
    """
    cars = ['🚗', '🚙', '🚕', '🚓', '🏎️']
    for _ in range(20):
        car = random.choice(cars)
        left = random.randint(5, 95)
        delay = random.uniform(0, 2)
        size = random.randint(24, 48)
        html_code += f'<div class="car-float" style="left: {left}%; animation-delay: {delay}s; font-size: {size}px;">{car}</div>'
    st.markdown(html_code, unsafe_allow_html=True)

# ---------- MUAT ARTEFAK ----------
@st.cache_resource
def load_artefak():
    model  = joblib.load('regresi_berganda_v2.pkl')
    scaler = joblib.load('scaler_v2.pkl')
    fitur  = joblib.load('fitur_v2.pkl')
    return model, scaler, fitur

@st.cache_data
def load_data():
    df = pd.read_csv('mobil_mentah.csv')
    df = df[df['Engine_size'] > 0].dropna()
    return df

try:
    model, scaler, FITUR = load_artefak()
    df_ref = load_data()
except Exception as e:
    st.error(f"❌ Gagal memuat file. Pastikan semua file .pkl dan mobil_mentah.csv ada di folder yang sama.\nError: {e}")
    st.stop()

# Ambil daftar maker & genmodel dari data
MAKERS = sorted(df_ref['Maker'].unique().tolist())
FUEL_TYPES_RAW = sorted(df_ref['Fuel_type'].unique().tolist())

# Kolom dummy yang dipakai model
FITUR_ANGKA_MODEL = ['Engine_size', 'Gas_emission', 'Year', 'Engine_size_sq', 'Engine_Gas']


# ---------- SIDEBAR: INPUT BERSIH ----------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Mobil_logo.svg/3840px-Mobil_logo.svg.png", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.header("🚗 Identitas Mobil")

# Dropdown Maker
maker_pilih = st.sidebar.selectbox(
    "Merek (Maker)",
    options=MAKERS,
    index=MAKERS.index('BMW') if 'BMW' in MAKERS else 0
)

# Dropdown Genmodel dinamis berdasarkan Maker
models_available = sorted(df_ref[df_ref['Maker'] == maker_pilih]['Genmodel'].unique().tolist())
genmodel_pilih = st.sidebar.selectbox("Model", options=models_available)

# Dropdown Fuel Type
fuel_pilih = st.sidebar.selectbox("Jenis Bahan Bakar", options=FUEL_TYPES_RAW, index=0)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Spesifikasi Teknis")

year_pilih        = st.sidebar.slider("Tahun Produksi", min_value=1998, max_value=2021, value=2018, step=1)
engine_pilih      = st.sidebar.number_input("Kapasitas Mesin (cc)", min_value=599, max_value=6761, value=1600, step=50)
emission_pilih    = st.sidebar.number_input("Emisi Gas (g/km)", min_value=0, max_value=570, value=120, step=5)

st.sidebar.markdown("---")
tombol = st.sidebar.button("🔮 Prediksi Harga", type="primary", use_container_width=True)


# ---------- MAIN AREA ----------
st.markdown("## 🚗 SmartCar Price Predictor")
st.markdown("Estimasi harga mobil menggunakan **Regresi Linear Berganda** — R² = **94.2%**")
st.divider()

# Ringkasan input
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Merek & Model", f"{maker_pilih} {genmodel_pilih.title()}")
with col2:
    st.metric("Tahun / Bahan Bakar", f"{year_pilih} / {fuel_pilih}")
with col3:
    st.metric("Mesin / Emisi", f"{engine_pilih}cc / {emission_pilih}g")


# ---------- LOGIKA PREDIKSI ----------
if tombol:
    with st.spinner("Menghitung estimasi harga..."):
        try:
            # 1. Feature engineering (sama dengan saat training)
            eng_sq  = engine_pilih ** 2
            eng_gas = engine_pilih * emission_pilih

            # 2. Scale fitur numerik
            arr_num = np.array([[engine_pilih, emission_pilih, year_pilih, eng_sq, eng_gas]])
            arr_scaled = scaler.transform(arr_num)

            # 3. Buat baris kosong dengan semua kolom model
            row = {f: 0.0 for f in FITUR}

            # 4. Isi fitur numerik yang sudah di-scale
            for i, fname in enumerate(FITUR_ANGKA_MODEL):
                if fname in row:
                    row[fname] = arr_scaled[0][i]

            # 5. Isi dummy Fuel_type
            # drop_first=True → kategori pertama (alphabetical) di-drop
            fuel_cols = sorted([f for f in FITUR if 'Fuel_type_' in f])
            fuel_col_name = f'Fuel_type_{fuel_pilih}'
            if fuel_col_name in row:
                row[fuel_col_name] = 1.0

            # 6. Isi dummy Genmodel
            genmodel_col_name = f'Genmodel_{genmodel_pilih}'
            if genmodel_col_name in row:
                row[genmodel_col_name] = 1.0

            # 7. Prediksi
            df_input = pd.DataFrame([row])[FITUR]
            prediksi = model.predict(df_input)[0]
            prediksi = max(prediksi, 0)  # clamp negative

            # ---------- TAMPILKAN HASIL ----------
            car_balloons()
            st.markdown(f"""
            <div class="result-box">
                <div class="result-label">Estimasi Harga Jual</div>
                <div class="result-price">£ {prediksi:,.0f}</div>
                <div class="result-note">Berdasarkan model regresi linear berganda · R² Test = 94.2%</div>
            </div>
            """, unsafe_allow_html=True)

            # Range harga wajar ±10%
            low = prediksi * 0.90
            high = prediksi * 1.10
            st.info(f"📊 Rentang wajar estimasi: **£ {low:,.0f}** – **£ {high:,.0f}** (±10%)")

            # Detail fitur
            with st.expander("🔍 Detail Input & Fitur Turunan"):
                detail = pd.DataFrame([{
                    'Maker': maker_pilih,
                    'Genmodel': genmodel_pilih,
                    'Fuel_type': fuel_pilih,
                    'Year': year_pilih,
                    'Engine_size (cc)': engine_pilih,
                    'Gas_emission (g/km)': emission_pilih,
                    'Engine_size² [poly]': eng_sq,
                    'Engine×Emission [interact]': eng_gas,
                }])
                st.dataframe(detail, use_container_width=True, hide_index=True)

            with st.expander("📊 Top 10 Koefisien Model"):
                df_koef = pd.DataFrame({
                    'Fitur': FITUR,
                    'Koefisien (β)': model.coef_.round(2)
                })
                df_koef['|β|'] = df_koef['Koefisien (β)'].abs()
                top10 = df_koef.sort_values('|β|', ascending=False).head(10).drop(columns='|β|')
                st.dataframe(top10, use_container_width=True, hide_index=True)
                st.caption(f"Intercept (β₀) = {model.intercept_:.2f}")

        except Exception as e:
            st.error(f"⚠️ Error saat prediksi: {e}")

else:
    st.markdown("""
    > **Cara pakai:**
    > 1. Pilih **merek** dan **model** mobil di sidebar kiri
    > 2. Atur **tahun, bahan bakar, mesin**, dan **emisi**
    > 3. Klik tombol **Prediksi Harga**
    """)


# ---------- FOOTER ----------
st.divider()
col_a, col_b = st.columns(2)
with col_a:
    st.caption("Kejuruan Data Analyst — Angkatan 2026")
with col_b:
    st.markdown('<p style="text-align:right;color:gray;font-size:12px;">PPKD Jakarta Selatan</p>', unsafe_allow_html=True)