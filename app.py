import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---------- 1. KONFIGURASI HALAMAN PREMIUM ----------
st.set_page_config(
    page_title='SmartCar - Prediksi Harga Mobil Berganda',
    page_icon='🚗',
    layout='centered',
    initial_sidebar_state='expanded'
)

# Kustomisasi CSS sedikit agar tampilan visual lebih rapi dan profesional
st.markdown("""
    <style>
    .big-font { font-size:24px !important; font-weight: bold; color: #1E3A8A; }
    .metric-box { background-color: #F3F4F6; padding: 15px; border-radius: 10px; border-left: 5px solid #3B82F6; }
    </style>
""", unsafe_allow_html=True)


# ---------- 2. MUAT MODEL & SCALER (CACHED) ----------
@st.cache_resource
def load_artefak():
    # Membaca file pkl yang sudah kamu siapkan di folder
    model  = joblib.load('regresi_berganda.pkl')
    scaler = joblib.load('scaler.pkl')
    fitur  = joblib.load('fitur.pkl')
    return model, scaler, fitur

try:
    model, scaler, FITUR = load_artefak()
except Exception as e:
    st.error(f"❌ Gagal memuat file model/scaler. Pastikan file pkl ada di folder yang sama dengan app.py. Error: {e}")
    st.stop()


# ---------- 3. HEADER UTAMA ----------
st.markdown('<p class="big-font">🚗 SmartCar Price Predictor Engine</p>', unsafe_allow_html=True)
st.markdown('Aplikasi berbasis **Machine Learning (Regresi Linear Berganda)** untuk memprediksi nilai estimasi harga mobil berdasarkan spesifikasi teknis dan fitur.')
st.divider()


# ---------- 4. SIDEBAR INPUT INTERACTIVE ----------
st.sidebar.header('🛠️ Spesifikasi & Fitur Mobil')
st.sidebar.markdown('Sesuaikan indikator di bawah ini:')

input_raw = {}

# Looping pintar: memisahkan input numeric asli dengan input bertipe dummy/kategori binary
for f in FITUR:
    if 'Fuel_type' in f:
        # Jika fiturnya adalah Dummy Variable Bahan Bakar, ubah jadi Selectbox Yes/No yang intuitif
        nama_bersih = f.replace('Fuel_type_', 'Bahan Bakar: ')
        pilihan = st.sidebar.selectbox(f"Apakah {nama_bersih}?", options=['No', 'Yes'], index=0)
        input_raw[f] = 1.0 if pilihan == 'Yes' else 0.0
    else:
        # Jika fitur angka biasa (Engine_size, Gas_emission, Year)
        if f == 'Year':
            input_raw[f] = st.sidebar.number_input(f"📅 Tahun Produksi ({f})", min_value=1980, max_value=2030, value=2018, step=1)
        elif f == 'Engine_size':
            input_raw[f] = st.sidebar.number_input(f"⚡ Kapasitas Mesin ({f})", value=1.6, step=0.1, format='%.2f')
        elif f == 'Gas_emission':
            input_raw[f] = st.sidebar.number_input(f"💨 Tingkat Emisi Gas ({f})", value=120.0, step=1.0, format='%.1f')
        else:
            input_raw[f] = st.sidebar.number_input(f"📊 Nilai Fitur: {f}", value=0.0, step=0.1)


# ---------- 5. TOMBOL AKSI & LOGIKA PREDIKSI ----------
st.sidebar.markdown('---')
tombol_prediksi = st.sidebar.button('🔮 Hitung Estimasi Harga', type='primary', use_container_width=True)

if tombol_prediksi:
    with st.spinner('Model sedang menganalisis data input...'):
        try:
            # Pisahkan fitur numerik untuk di-scale sesuai dengan yang dipelajari scaler saat fit
            fitur_numerik = list(scaler.feature_names_in_)
            df_numerik = pd.DataFrame([[input_raw[f] for f in fitur_numerik]], columns=fitur_numerik)
            
            # Lakukan standardisasi (scaling) hanya pada fitur numerik
            arr_numerik_scaled = scaler.transform(df_numerik)
            
            # Gabungkan kembali fitur numerik yang sudah di-scale dengan fitur dummy (kategori)
            input_siap_prediksi = {}
            for i, f in enumerate(fitur_numerik):
                input_siap_prediksi[f] = arr_numerik_scaled[0][i]
                
            for f in FITUR:
                if f not in fitur_numerik:
                    input_siap_prediksi[f] = input_raw[f]
            
            # Susun DataFrame akhir sesuai urutan list FITUR agar cocok dengan model regresi
            df_final = pd.DataFrame([[input_siap_prediksi[f] for f in FITUR]], columns=FITUR)
            
            # Prediksi harga menggunakan model regresi berganda
            prediksi_harga = model.predict(df_final)[0]
            
            # ---------- 6. DISPLAY HASIL INTERFACE MENARIK ----------
            st.balloons()
            st.subheader('🎯 Hasil Analisis Model')
            
            # Tampilkan harga dalam kotak estetik berukuran besar
            st.markdown(f"""
                <div class="metric-box">
                    <span style="font-size: 14px; color: #555;">ESTIMASI NILAI JUAL MOBIL:</span><br>
                    <span style="font-size: 32px; font-weight: bold; color: #10B981;">$ {prediksi_harga:,.2f}</span>
                </div>
            """, unsafe_allow_html=True)
            st.caption("Note: Nilai di atas didasarkan pada perhitungan bobot regresi terstandarisasi.")
            
            # Tampilkan rangkuman input dalam bentuk tabel yang rapi
            with st.expander("🔍 Lihat Detail Input Data"):
                st.dataframe(pd.DataFrame([input_raw]), use_container_width=True, hide_index=True)
                
            # Tampilkan transparansi koefisien persamaan regresi untuk dosen
            with st.expander("📊 Lihat Struktur Koefisien Model (Transparansi Rumus)"):
                df_koef = pd.DataFrame({
                    'Nama Fitur (Prediktor)': FITUR,
                    'Koefisien Bobot (β)': model.coef_.round(4)
                })
                st.dataframe(df_koef, use_container_width=True, hide_index=True)
                st.info(f"**Nilai Intercept Konstanta (β₀):** {model.intercept_:.4f}")
                
        except Exception as e:
            st.error(f'⚠️ Gagal memproses prediksi. Terjadi kesalahan matematis: {e}')
else:
    # Tampilan awal saat web baru dibuka sebelum tombol ditekan
    st.info('💡 **Petunjuk Penggunaan:** Silakan sesuaikan spesifikasi mobil pada panel kontrol di bagian **sidebar sebelah kiri**, lalu klik tombol **"Hitung Estimasi Harga"** untuk memprediksi.')


# ---------- 7. FOOTER INFORMASI PPKD ----------
st.divider()
col1, col2 = st.columns(2)
with col1:
    st.caption('Kejuruan Data Analyst — Angkatan 2026')
with col2:
    st.markdown('<p style="text-align: right; color: gray; font-size: 12px;">PPKD Jakarta Selatan</p>', unsafe_allow_html=True)