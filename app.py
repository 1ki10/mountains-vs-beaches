# Import library yang diperlukan
import streamlit as st          # Library utama untuk membuat web app
import pandas as pd            # Untuk manipulasi data dalam bentuk dataframe
import numpy as np             # Untuk operasi numerik
import pickle                  # Untuk load model yang sudah disimpan
from sklearn.preprocessing import StandardScaler  # Untuk scaling features

# ===== KONFIGURASI HALAMAN =====
# Mengatur tampilan halaman web (judul di tab browser, icon, layout)
st.set_page_config(
    page_title="Vacation Preference Predictor",  # Judul yang muncul di tab browser
    page_icon="🏔️",                             # Icon yang muncul di tab browser
    layout="wide"                                # Layout lebar penuh (bukan centered)
)

# ===== FUNGSI LOAD MODEL =====
# @st.cache_resource membuat fungsi ini hanya dijalankan sekali
# dan hasilnya disimpan di cache untuk performa lebih baik
@st.cache_resource
def load_model():
    """
    Fungsi untuk load model XGBoost yang sudah di-train sebelumnya.
    Model disimpan dalam format pickle (.pkl)
    """
    try:
        # Buka file model dan load menggunakan pickle
        with open('Vacation_Preference_XGBoost_Model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        # Jika file tidak ditemukan, tampilkan error
        st.error("Model file not found. Please ensure 'Vacation_Preference_XGBoost_Model.pkl' is in the repository.")
        return None

# ===== HEADER DAN DESKRIPSI APLIKASI =====
# Judul utama aplikasi
st.title("🏔️ Mountains vs 🏖️ Beaches Preference Predictor")

# Garis horizontal untuk pemisah
st.markdown("---")

# Deskripsi aplikasi
st.markdown("""
This application predicts whether a person prefers **Mountains** or **Beaches** for their vacation 
based on various demographic and preference factors.
""")

# ===== SIDEBAR UNTUK NAVIGASI =====
# Sidebar adalah panel di sebelah kiri untuk input
st.sidebar.header("Enter Your Information")
st.sidebar.markdown("Please fill in all the fields below:")

# ===== LAYOUT DUA KOLOM UNTUK INPUT =====
# Membagi halaman menjadi 2 kolom agar lebih rapi
col1, col2 = st.columns(2)

# ===== KOLOM 1: INFORMASI DEMOGRAFIS =====
with col1:
    st.subheader("📊 Demographic Information")
    
    # Input umur (number input dengan min/max value)
    age = st.number_input(
        "Age",                    # Label
        min_value=18,            # Nilai minimum
        max_value=100,           # Nilai maksimum
        value=30                 # Nilai default
    )
    
    # Dropdown untuk memilih gender
    gender = st.selectbox(
        "Gender",                                    # Label
        ["male", "female", "non-binary"]            # Pilihan yang tersedia
    )
    
    # Input pendapatan tahunan
    income = st.number_input(
        "Annual Income ($)", 
        min_value=0, 
        max_value=500000, 
        value=50000,             # Nilai default
        step=1000                # Increment/decrement per klik
    )
    
    # Dropdown untuk tingkat pendidikan
    education = st.selectbox(
        "Education Level", 
        ["high school", "bachelor", "master", "doctorate"]  # Urutan dari rendah ke tinggi
    )
    
    # Dropdown untuk tipe lokasi tempat tinggal
    location = st.selectbox(
        "Location Type", 
        ["urban", "suburban", "rural"]
    )

# ===== KOLOM 2: PREFERENSI TRAVEL =====
with col2:
    st.subheader("🎯 Travel Preferences")
    
    # Input frekuensi travel per tahun
    travel_frequency = st.number_input(
        "Travel Frequency (trips per year)", 
        min_value=0, 
        max_value=20, 
        value=2
    )
    
    # Input budget untuk liburan
    vacation_budget = st.number_input(
        "Vacation Budget ($)", 
        min_value=0, 
        max_value=50000, 
        value=2000, 
        step=100
    )
    
    # Dropdown untuk aktivitas yang disukai
    activities = st.selectbox(
        "Preferred Activities", 
        ["hiking", "swimming", "skiing", "sunbathing"]
    )
    
    # Dropdown untuk musim favorit
    season = st.selectbox(
        "Favorite Season", 
        ["summer", "winter", "spring", "fall"]
    )

# Garis pemisah
st.markdown("---")

# ===== LAYOUT DUA KOLOM UNTUK FAKTOR TAMBAHAN =====
col3, col4 = st.columns(2)

# ===== KOLOM 3: INFORMASI JARAK =====
with col3:
    st.subheader("📍 Proximity Information")
    
    # Slider untuk jarak ke gunung (0-500 miles)
    proximity_mountains = st.slider(
        "Distance to Mountains (miles)",  # Label
        0,                               # Nilai minimum
        500,                             # Nilai maksimum
        100                              # Nilai default
    )
    
    # Slider untuk jarak ke pantai
    proximity_beaches = st.slider(
        "Distance to Beaches (miles)", 
        0, 
        500, 
        100
    )

# ===== KOLOM 4: FAKTOR LAINNYA =====
with col4:
    st.subheader("🌟 Other Factors")
    
    # Checkbox untuk kepemilikan hewan peliharaan
    pets = st.checkbox("Do you have pets?")
    
    # Checkbox untuk kepedulian lingkungan
    environmental_concerns = st.checkbox("Do you have environmental concerns?")

# ===== KONVERSI CHECKBOX KE BINARY =====
# Model membutuhkan input 0 atau 1, bukan True/False
pets_binary = 1 if pets else 0
env_concerns_binary = 1 if environmental_concerns else 0

# ===== TOMBOL PREDIKSI =====
# Tombol dengan style primary (warna berbeda)
if st.button("🔮 Predict My Preference", type="primary"):
    try:
        # ===== STEP 1: MEMBUAT DATAFRAME INPUT =====
        # Buat dataframe dengan semua numerical features
        input_data = pd.DataFrame({
            'Age': [age],
            'Income': [income],
            'Travel_Frequency': [travel_frequency],
            'Vacation_Budget': [vacation_budget],
            'Proximity_to_Mountains': [proximity_mountains],
            'Proximity_to_Beaches': [proximity_beaches],
            'Pets': [pets_binary],
            'Environmental_Concerns': [env_concerns_binary]
        })
        
        # ===== STEP 2: LABEL ENCODING UNTUK EDUCATION =====
        # Education adalah ordinal data (ada urutannya)
        # Kita konversi ke angka: high school=0, bachelor=1, dst
        education_mapping = {
            'high school': 0,
            'bachelor': 1,
            'master': 2,
            'doctorate': 3
        }
        input_data['Education_Level'] = education_mapping[education]
        
        # ===== STEP 3: ONE-HOT ENCODING UNTUK CATEGORICAL FEATURES =====
        # Categorical features perlu di-encode menjadi binary columns
        # Contoh: Gender (male/female/non-binary) menjadi 3 kolom binary
        
        # One-hot encoding untuk Gender
        input_data['Gender_female'] = 1 if gender == 'female' else 0
        input_data['Gender_male'] = 1 if gender == 'male' else 0
        input_data['Gender_non-binary'] = 1 if gender == 'non-binary' else 0
        
        # One-hot encoding untuk Preferred Activities
        input_data['Preferred_Activities_hiking'] = 1 if activities == 'hiking' else 0
        input_data['Preferred_Activities_skiing'] = 1 if activities == 'skiing' else 0
        input_data['Preferred_Activities_sunbathing'] = 1 if activities == 'sunbathing' else 0
        input_data['Preferred_Activities_swimming'] = 1 if activities == 'swimming' else 0
        
        # One-hot encoding untuk Location
        input_data['Location_rural'] = 1 if location == 'rural' else 0
        input_data['Location_suburban'] = 1 if location == 'suburban' else 0
        input_data['Location_urban'] = 1 if location == 'urban' else 0
        
        # One-hot encoding untuk Favorite Season
        input_data['Favorite_Season_fall'] = 1 if season == 'fall' else 0
        input_data['Favorite_Season_spring'] = 1 if season == 'spring' else 0
        input_data['Favorite_Season_summer'] = 1 if season == 'summer' else 0
        input_data['Favorite_Season_winter'] = 1 if season == 'winter' else 0
        
        # ===== STEP 4: PASTIKAN URUTAN KOLOM SAMA DENGAN TRAINING =====
        # SANGAT PENTING: Urutan kolom harus sama persis dengan saat training
        feature_columns = [
            'Age', 'Income', 'Education_Level', 'Travel_Frequency', 
            'Vacation_Budget', 'Proximity_to_Mountains', 'Proximity_to_Beaches',
            'Pets', 'Environmental_Concerns', 'Gender_female', 'Gender_male',
            'Gender_non-binary', 'Preferred_Activities_hiking',
            'Preferred_Activities_skiing', 'Preferred_Activities_sunbathing',
            'Preferred_Activities_swimming', 'Location_rural', 'Location_suburban',
            'Location_urban', 'Favorite_Season_fall', 'Favorite_Season_spring',
            'Favorite_Season_summer', 'Favorite_Season_winter'
        ]
        
        # Susun ulang kolom sesuai urutan yang benar
        input_data = input_data[feature_columns]
        
        # ===== STEP 5: FEATURE SCALING =====
        # StandardScaler untuk normalisasi data (mean=0, std=1)
        # CATATAN: Idealnya gunakan scaler yang sama dengan training
        scaler = StandardScaler()
        input_scaled = scaler.fit_transform(input_data)
        
        # ===== STEP 6: LOAD MODEL DAN PREDIKSI =====
        # Load model yang sudah di-train
        model = load_model()
        if model is None:
            st.error("Model could not be loaded. Please check the model file.")
            return
        
        # Lakukan prediksi
        prediction = model.predict(input_scaled)[0]  # 0 atau 1
        
        # Dapatkan probability untuk setiap class
        prediction_proba = model.predict_proba(input_scaled)[0]  # [prob_beach, prob_mountain]
        
        # ===== STEP 7: TAMPILKAN HASIL =====
        st.markdown("---")
        st.subheader("🎯 Prediction Results")
        
        # Buat 3 kolom untuk centering hasil
        col_result1, col_result2, col_result3 = st.columns([1,2,1])
        
        with col_result2:  # Kolom tengah
            if prediction == 1:  # Prefer Mountains
                # Tampilkan hasil dengan warna hijau (success)
                st.success("### 🏔️ You prefer **MOUNTAINS**!")
                
                # Tampilkan confidence level
                confidence = prediction_proba[1] * 100
                st.info(f"Confidence: {confidence:.1f}%")
                
                # Deskripsi hasil
                st.markdown("""
                Based on your preferences, you seem to enjoy:
                - Fresh mountain air and scenic views
                - Hiking trails and outdoor adventures
                - Cooler temperatures and peaceful environments
                - Activities like hiking, skiing, or mountain biking
                """)
            else:  # Prefer Beaches (prediction == 0)
                # Tampilkan hasil dengan warna biru (info)
                st.info("### 🏖️ You prefer **BEACHES**!")
                
                # Tampilkan confidence level
                confidence = prediction_proba[0] * 100
                st.info(f"Confidence: {confidence:.1f}%")
                
                # Deskripsi hasil
                st.markdown("""
                Based on your preferences, you seem to enjoy:
                - Sandy beaches and ocean waves
                - Swimming and water sports
                - Warm weather and sunshine
                - Activities like sunbathing, surfing, or beach volleyball
                """)
        
        # ===== STEP 8: VISUALISASI PROBABILITAS =====
        st.markdown("---")
        st.subheader("📊 Preference Probability Distribution")
        
        # Buat dataframe untuk visualisasi
        prob_df = pd.DataFrame({
            'Preference': ['Beaches', 'Mountains'],
            'Probability': [prediction_proba[0] * 100, prediction_proba[1] * 100]
        })
        
        # Tampilkan bar chart menggunakan Streamlit
        st.bar_chart(prob_df.set_index('Preference'))
        
    except Exception as e:
        # ===== ERROR HANDLING =====
        # Jika ada error, tampilkan pesan error yang informatif
        st.error(f"An error occurred: {str(e)}")
        st.error("Please make sure the model file 'Vacation_Preference_XGBoost_Model.pkl' is in the same directory as this app.")

# ===== FOOTER =====
# Informasi tambahan di bagian bawah halaman
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Created for Data Science Final Project - Digital Skola Batch 47</p>
    <p>Group 10: Mountains vs Beaches Preferences</p>
</div>
""", unsafe_allow_html=True)  # unsafe_allow_html untuk menggunakan HTML tags