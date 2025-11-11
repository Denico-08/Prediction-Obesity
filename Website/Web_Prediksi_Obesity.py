import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import os
from matplotlib import pyplot as plt
from catboost import CatBoostClassifier
from lime.lime_tabular import LimeTabularExplainer
import dice_ml
from dice_ml import Dice
from sklearn.preprocessing import LabelEncoder
import traceback

# ======================================================================================
# 1. KONFIGURASI & SETUP APLIKASI
# ======================================================================================

st.set_page_config(page_title="Prediksi Obesitas (XAI)", layout="wide")

# --- Path ke Aset Model (SESUAI DENGAN REPO ANDA) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR,"Model") 

# PERBAIKAN: Menggunakan file .cbm seperti yang Anda minta
MODEL_PATH = os.path.join(MODEL_DIR, "catboost_obesity_model.cbm") 
TARGET_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder_y_v3.pkl")
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "feature_names_v3.pkl")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names_y_v3.pkl")

DATA_PATH = r"C:\Users\LENOVO\Documents\DENICO\Skripsi\Python\Dataset\combined_dataset.csv"

# --- Konfigurasi Tipe Fitur (Berdasarkan Notebook) ---
TARGET_NAME = 'NObeyesdad'

# Definisikan kolom-kolom berdasarkan tipe data
CONTINUOUS_COLS = ['Age', 'Height', 'Weight']
CATEGORICAL_COLS = [
    'Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE', 
    'family_history_with_overweight', 'CAEC', 'MTRANS'
]
ORDINAL_COLS = ['FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

# Semua kolom kategorikal (termasuk ordinal)
ALL_CATEGORICAL_COLS = CATEGORICAL_COLS + ORDINAL_COLS

# Daftar semua fitur dalam urutan yang digunakan model
ALL_FEATURES = CONTINUOUS_COLS + ALL_CATEGORICAL_COLS

# --- Mapping untuk UI (hanya untuk tampilan, tidak untuk preprocessing) ---
# PERBAIKAN: Ubah nilai (value) menjadi integer
GENDER_MAP = {'Female': 0, 'Male': 1}
FAMILY_HISTORY_MAP = {'no': 0, 'yes': 1}

# Label yang lebih mudah dibaca untuk setiap fitur
FEATURE_LABELS = {
    'Age': 'Umur',
    'Gender': 'Jenis Kelamin',
    'Height': 'Tinggi Badan',
    'Weight': 'Berat Badan',
    'CALC': 'Konsumsi Alkohol',
    'FAVC': 'Konsumsi Makanan Tinggi Kalori',
    'FCVC': 'Konsumsi Sayuran',
    'NCP': 'Jumlah Makan Utama per Hari',
    'SCC': 'Pemantauan Kalori',
    'SMOKE': 'Status Merokok',
    'CH2O': 'Konsumsi Air',
    'family_history_with_overweight': 'Riwayat Keluarga dengan Berat Berlebih',
    'FAF': 'Aktivitas Fisik',
    'TUE': 'Waktu Penggunaan Teknologi',
    'CAEC': 'Konsumsi Makanan Ringan',
    'MTRANS': 'Moda Transportasi'
}
FAVC_MAP = {'no': 0, 'yes': 1}
SCC_MAP = {'no': 0, 'yes': 1}
SMOKE_MAP = {'no': 0, 'yes': 1}
CAEC_MAP = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
CALC_MAP = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
MTRANS_MAP = {
    'Walking': 0, 
    'Public_Transportation': 1, 
    'Bike': 2, 
    'Motorbike': 3, 
    'Automobile': 4
}

# ======================================================================================
# 2. FUNGSI PREPROCESSING & PEMUATAN ASET
# ======================================================================================

def preprocess_input_data(input_dict, all_features_list):
    """
    Mengonversi data input (dari UI, mayoritas string) ke format
    numerik penuh (float dan int) yang siap digunakan oleh model.
    """
    df = pd.DataFrame([input_dict])
    if df.empty:
        return None

    try:
        # 1. Konversi kolom numerik (CONTINUOUS)
        for col in CONTINUOUS_COLS:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Penanganan NA (Anda bisa sesuaikan nilai default)
            if df[col].isnull().any() or np.isinf(df[col].values).any():
                if col == 'Age':
                    df[col] = df[col].fillna(25)
                elif col == 'Height':
                    df[col] = df[col].fillna(1.70)
                elif col == 'Weight':
                    df[col] = df[col].fillna(70)
            
            # Bulatkan Age ke integer, sisanya float
            if col == 'Age':
                 df[col] = df[col].round().astype(int)
            else:
                 df[col] = df[col].round(3).astype(float)

        # 2. Konversi kolom KATEGORIKAL (string -> int) menggunakan MAP
        # (Menggunakan mapping global yang sudah diperbaiki)
        df['Gender'] = df['Gender'].map(GENDER_MAP).astype(int)
        df['family_history_with_overweight'] = df['family_history_with_overweight'].map(FAMILY_HISTORY_MAP).astype(int)
        df['FAVC'] = df['FAVC'].map(FAVC_MAP).astype(int)
        df['SCC'] = df['SCC'].map(SCC_MAP).astype(int)
        df['SMOKE'] = df['SMOKE'].map(SMOKE_MAP).astype(int)
        df['CAEC'] = df['CAEC'].map(CAEC_MAP).astype(int)
        df['CALC'] = df['CALC'].map(CALC_MAP).astype(int)
        df['MTRANS'] = df['MTRANS'].map(MTRANS_MAP).astype(int)
        
        # 3. Konversi kolom ORDINAL (string/int dari UI -> int)
        ordinal_defaults = {
            'FCVC': 2, 'NCP': 3, 'CH2O': 2, 'FAF': 1, 'TUE': 1
        }
        
        for col in ORDINAL_COLS:
            # Input dari UI (bisa jadi int atau string)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(
                ordinal_defaults[col]
            ).round().astype(int)

    except Exception as e:
        st.error(f"Error saat preprocessing data input: {e}")
        return None
        
    if not all_features_list:
        st.error("Daftar fitur (ALL_FEATURES) kosong.")
        return None
    
    try:
        # Pastikan urutan kolom sesuai
        df = df[all_features_list]
    except KeyError as e:
        st.error(f"Error: Kolom yang hilang saat preprocessing: {e}")
        return None
        
    return df

@st.cache_resource
def load_all_assets():
    """
    Memuat semua aset dari file .pkl di repositori.
    """
    try:
        # 1. Muat Model (format .cbm)
        model = CatBoostClassifier()
        model.load_model(MODEL_PATH)
        
        target_encoder = joblib.load(TARGET_ENCODER_PATH)
        all_features = joblib.load(FEATURE_NAMES_PATH)
        class_names = joblib.load(CLASS_NAMES_PATH)
        
        encoders = { TARGET_NAME: target_encoder }
        
        # 2. Muat Data Mentah (untuk LIME/DiCE)
        x_train_raw = pd.read_csv(DATA_PATH, usecols=all_features + [TARGET_NAME])
        x_train_raw = x_train_raw.dropna().reset_index(drop=True)
        
        # 3. Buat x_train_encoded (data numerik)
        x_train_processed = x_train_raw.copy()
        
        # Konversi CONTINUOUS ke numeric
        for col in CONTINUOUS_COLS:
            x_train_processed[col] = pd.to_numeric(x_train_processed[col], errors='coerce')
        
        # Konversi KATEGORIKAL (string -> int) menggunakan mapping global
        x_train_processed['Gender'] = x_train_processed['Gender'].map(GENDER_MAP)
        x_train_processed['family_history_with_overweight'] = x_train_processed['family_history_with_overweight'].map(FAMILY_HISTORY_MAP)
        x_train_processed['FAVC'] = x_train_processed['FAVC'].map(FAVC_MAP)
        x_train_processed['SCC'] = x_train_processed['SCC'].map(SCC_MAP)
        x_train_processed['SMOKE'] = x_train_processed['SMOKE'].map(SMOKE_MAP)
        x_train_processed['CAEC'] = x_train_processed['CAEC'].map(CAEC_MAP)
        x_train_processed['CALC'] = x_train_processed['CALC'].map(CALC_MAP)
        x_train_processed['MTRANS'] = x_train_processed['MTRANS'].map(MTRANS_MAP)
        
        # Konversi ORDINAL ke numeric
        for col in ORDINAL_COLS:
            x_train_processed[col] = pd.to_numeric(x_train_processed[col], errors='coerce')

        # Hapus baris yang GAGAL di-mapping atau dikonversi (menjadi NA)
        x_train_processed = x_train_processed.dropna()
        
        # Sekarang, konversi semua kategorikal (termasuk ordinal) ke INT
        # Ini aman karena semua data sudah numerik
        for col in ALL_CATEGORICAL_COLS:
            x_train_processed[col] = x_train_processed[col].round().astype(int)
        
        # Pastikan continuous adalah float (kecuali Age)
        for col in CONTINUOUS_COLS:
            if col == 'Age':
                x_train_processed[col] = x_train_processed[col].round().astype(int)
            else:
                x_train_processed[col] = x_train_processed[col].astype(float)

        x_train_encoded = x_train_processed[all_features]

        return model, encoders, all_features, class_names, x_train_encoded
    
    except FileNotFoundError as e:
        st.error(f"Gagal memuat file aset: {e}. Pastikan file .pkl/.cbm ada di folder 'Model/models/'.")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"Gagal memuat aset penting: {e}")
        return None, None, None, None, None

# ======================================================================================
# 3. FUNGSI LIME (PENDEKATAN NUMERIK)
# ======================================================================================

@st.cache_resource
def initialize_lime_explainer(_X_train_encoded, all_features_list, class_names_list):
    """Inisialisasi LIME explainer menggunakan data yang sudah di-encode (numerik)."""
    training_values = _X_train_encoded[all_features_list].values 

    categorical_feature_indices = [
        _X_train_encoded.columns.get_loc(col) for col in ALL_CATEGORICAL_COLS 
        if col in _X_train_encoded.columns
    ]
    
    categorical_names_map = {}
    if 'Gender' in all_features_list:
        categorical_names_map[all_features_list.index('Gender')] = list(GENDER_MAP.keys())
    if 'family_history_with_overweight' in all_features_list:
        categorical_names_map[all_features_list.index('family_history_with_overweight')] = list(FAMILY_HISTORY_MAP.keys())

    lime_explainer = LimeTabularExplainer(
        training_data=training_values,
        feature_names=all_features_list,
        class_names=class_names_list,
        categorical_features=categorical_feature_indices,
        categorical_names=categorical_names_map,
        mode='classification',
        random_state=42
    )
    return lime_explainer

def predict_proba_catboost_for_lime(data, model_obj, all_features_list):
    """Wrapper prediksi KHUSUS UNTUK LIME (Menerima data numerik)."""
    if isinstance(data, (list, np.ndarray)):
        arr = np.array(data)
        if arr.ndim == 1:
            df = pd.DataFrame([arr], columns=all_features_list)
        else:
            df = pd.DataFrame(arr, columns=all_features_list)
    else:
        df = pd.DataFrame(data)

    input_encoded = df.reindex(columns=all_features_list, fill_value=0)

    # Paksa semua fitur kategorikal menjadi INTEGER
    for col in ALL_CATEGORICAL_COLS:
        if col in input_encoded.columns:
            numeric_col = pd.to_numeric(input_encoded[col], errors='coerce').fillna(0)
            input_encoded[col] = numeric_col.round().astype(int)
            
    for col in CONTINUOUS_COLS:
         if col in input_encoded.columns:
            input_encoded[col] = pd.to_numeric(input_encoded[col], errors='coerce').fillna(0).astype(float)

    proba = model_obj.predict_proba(input_encoded[all_features_list])
    return proba

# ======================================================================================
# 4. FUNGSI DICE (PENDEKATAN STRING)
# ======================================================================================

class DiceCatBoostWrapper:
    """Wrapper KHUSUS UNTUK DiCE. Menerima string, mengonversi ke int."""
    def __init__(self, model, feature_names, continuous_features, categorical_features):
        self.model = model
        self.feature_names = feature_names
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features

    def predict_proba(self, X):
        # Konversi numpy array ke DataFrame jika perlu
        if isinstance(X, np.ndarray):
            if X.size == 0:
                raise ValueError("Empty input array passed to predict_proba")

            # Tangani vektor 1-dim dan matriks 2-dim secara eksplisit
            if X.ndim == 1:
                # Jika panjang cocok dengan jumlah fitur, ubah menjadi 1 x n_features
                if X.shape[0] == len(self.feature_names):
                    X = X.reshape(1, -1)
                else:
                    # Jika tidak cocok, tapi bisa dibagi, coba reshape
                    if X.size % len(self.feature_names) == 0:
                        X = X.reshape(-1, len(self.feature_names))
                    else:
                        raise ValueError(f"Numpy array shape {X.shape} incompatible with feature_names length {len(self.feature_names)}")

            if X.ndim == 2:
                if X.shape[1] != len(self.feature_names):
                    raise ValueError(f"Numpy array has {X.shape[1]} columns but expected {len(self.feature_names)} features")

            X = pd.DataFrame(X, columns=self.feature_names)
        
        # Validasi input
        if X.empty:
            raise ValueError("Empty DataFrame passed to predict_proba")
            
        X_copy = X.copy()

        # Proses continuous features
        for col in self.continuous_features:
            if col in X_copy.columns:
                try:
                    X_copy[col] = pd.to_numeric(X_copy[col], errors='coerce')
                    # Fill NaN dengan mean atau median dari kolom
                    if X_copy[col].isnull().any():
                        median_val = X_copy[col].median()
                        if pd.isnull(median_val):
                            median_val = 0
                        X_copy[col] = X_copy[col].fillna(median_val)
                except Exception as e:
                    raise ValueError(f"Error processing continuous feature {col}: {str(e)}")
        
        # Proses categorical features
        for col in self.categorical_features:
            if col in X_copy.columns:
                try:
                    # Konversi string ke numeric
                    val_numeric = pd.to_numeric(X_copy[col], errors='coerce')
                    # Fill NaN dengan 0
                    val_numeric = val_numeric.fillna(0)
                    # Round ke integer
                    val_int = val_numeric.round().astype(int)
                    X_copy[col] = val_int
                except Exception as e:
                    raise ValueError(f"Error processing categorical feature {col}: {str(e)}")

        # Validasi final
        if not all(col in X_copy.columns for col in self.feature_names):
            raise ValueError(f"Missing columns in processed data. Expected: {self.feature_names}. Got: {X_copy.columns.tolist()}")

        # Reorder columns
        X_copy = X_copy[self.feature_names]
        
        # Final validation
        if X_copy.isnull().any().any():
            raise ValueError("NaN values in processed data")
        
        try:
            return self.model.predict_proba(X_copy)
        except Exception as e:
            raise RuntimeError(f"Model prediction error inside Dice wrapper: {str(e)}")

def get_dice_recommendations(_x_train_encoded, model_obj, encoders, input_df_processed, desired_class_index, all_features_list):
    """
    Menghasilkan rekomendasi DiCE dengan batasan fitur fisik yang ketat
    dan menggunakan metode genetic untuk eksplorasi yang lebih mendalam.
    """
    # Tampilkan progress bar
    progress_text = "Mencari rekomendasi perubahan..."
    my_bar = st.progress(0, text=progress_text)
    
    try:
        # 1. Siapkan DataFrame untuk DiCE (data pelatihan)
        my_bar.progress(10, text="Mempersiapkan data training...")
        df_dice = _x_train_encoded.copy()
        
        # Tambahkan kolom target ter-encode
        string_predictions = model_obj.predict(df_dice[all_features_list]).ravel()
        df_dice[TARGET_NAME] = encoders[TARGET_NAME].transform(string_predictions)

        # 2. Definisikan tipe fitur dan batasan
        my_bar.progress(30, text="Mengatur batasan nilai fitur...")
        
        # Definisi ulang fitur kontinyu dan kategorikal
        dice_continuous_features = []  # Kosongkan continuous features
        categorical_features_for_dice = ALL_FEATURES  # Semua fitur sebagai kategorikal
        
        # Set semua feature sebagai categorical untuk kontrol yang lebih baik
        feature_types = {col: 'categorical' for col in categorical_features_for_dice}
        
        # Siapkan permitted range untuk semua fitur
        permitted_range = {}
        
        # Batasan untuk fitur kategorikal asli (menggunakan nilai numerik)
        categorical_ranges = {
            'Gender': ['0', '1'],  # 0: Female, 1: Male
            'FAVC': ['0', '1'],    # 0: no, 1: yes
            'SCC': ['0', '1'],     # 0: no, 1: yes
            'SMOKE': ['0', '1'],   # 0: no, 1: yes
            'CAEC': ['0', '1', '2', '3'],  # 0: no, 1: Sometimes, 2: Frequently, 3: Always
            'CALC': ['0', '1', '2', '3'],  # 0: no, 1: Sometimes, 2: Frequently, 3: Always
            'MTRANS': ['0', '1', '2', '3', '4'],  # 0: Walking, 1: Public_Transportation, 2: Bike, 3: Motorbike, 4: Automobile
            'family_history_with_overweight': ['0', '1']  # 0: no, 1: yes
        }
        
        # Batasan untuk fitur ordinal (tetap menggunakan nilai numerik)
        ordinal_ranges = {
            'FCVC': ['1', '2', '3'],
            'NCP': ['1', '2', '3', '4'],
            'CH2O': ['1', '2', '3'],
            'FAF': ['0', '1', '2', '3'],
            'TUE': ['0', '1', '2']
        }
        
        # Konversi numerik ke string dengan format yang tepat
        df_dice['Age'] = df_dice['Age'].round().astype(int).astype(str)  # Age sebagai integer
        df_dice['Height'] = df_dice['Height'].apply(lambda x: f"{float(x):.1f}")  # Height dengan 1 desimal
        df_dice['Weight'] = df_dice['Weight'].apply(lambda x: f"{float(x):.1f}")  # Weight dengan 1 desimal

        # Temukan kategori unik dan urutkan
        age_categories = sorted(df_dice['Age'].unique().tolist())
        height_categories = sorted(df_dice['Height'].unique().tolist())
        weight_categories = sorted(df_dice['Weight'].unique().tolist())

    # (removed debug prints for ranges)
    # Tambahkan ke permitted range
        permitted_range['Age'] = age_categories
        permitted_range['Height'] = height_categories
        permitted_range['Weight'] = weight_categories

    # (debug info removed)
        
        # Update permitted range untuk semua fitur
        permitted_range.update(categorical_ranges)
        permitted_range.update(ordinal_ranges)
        
        # Konversi semua kolom ke string
        for col in categorical_features_for_dice:
            df_dice[col] = df_dice[col].astype(str)

        # 3. Siapkan Data Interface DiCE
        my_bar.progress(50, text="Inisialisasi DiCE...")
        data_interface = dice_ml.Data(
            dataframe=df_dice,
            continuous_features=[],  # Tidak ada continuous features
            outcome_name=TARGET_NAME,
            type_of_features=feature_types
        )
        
        # 4. Siapkan Model Interface DiCE
        wrapped_model = DiceCatBoostWrapper(
            model_obj, 
            all_features_list, 
            CONTINUOUS_COLS, 
            ALL_CATEGORICAL_COLS
        )
        model_interface = dice_ml.Model(model=wrapped_model, backend="sklearn", model_type='classifier')
        
        # 5. Inisialisasi Explainer dengan parameter genetik yang dioptimalkan
        my_bar.progress(70, text="Menyiapkan algoritma pencarian...")
        dice_explainer = Dice(
            data_interface, 
            model_interface, 
            method="genetic"  # Gunakan algoritma genetik untuk pencarian
        ) 

        # 6. Siapkan Query Instance (Data Input User)
        query_instance = input_df_processed[all_features_list].copy() 
        
        # Pastikan semua nilai kategorikal adalah string
        for col in categorical_features_for_dice:
            if col in query_instance.columns:
                query_instance[col] = query_instance[col].round().astype(int).astype(str)
        
        # Pastikan nilai kontinyu dalam range yang valid
        for col in dice_continuous_features:
            if col in query_instance.columns:
                query_instance[col] = pd.to_numeric(query_instance[col], errors='coerce')
                
        # 7. Batasan Fitur dan Range
        FIXED_FEATURES = ['Gender', 'Age', 'Height', 'Weight']
        features_to_vary_list = [
            col for col in all_features_list 
            if col not in FIXED_FEATURES
        ]

        # Tambahkan batasan untuk fitur ordinal
        for col in ['FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']:
            if col in permitted_range:
                continue
            if col == 'FCVC':
                permitted_range[col] = ['1', '2', '3']
            elif col == 'NCP':
                permitted_range[col] = ['1', '2', '3', '4']
            elif col == 'CH2O':
                permitted_range[col] = ['1', '2', '3']
            elif col == 'FAF':
                permitted_range[col] = ['0', '1', '2', '3']
            elif col == 'TUE':
                permitted_range[col] = ['0', '1', '2']

        # 8. Hasilkan Counterfactuals dengan parameter yang dioptimalkan
        my_bar.progress(90, text="Mencari rekomendasi terbaik...")
        
        # Coba beberapa kali jika gagal menemukan counterfactuals
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # Sesuaikan parameter berdasarkan percobaan
                total_cfs = 3 + (attempt * 2)  # Meningkatkan jumlah CF setiap percobaan
                proximity_weight = max(0.2, 1.0 - (attempt * 0.4))  # 1.0, 0.6, 0.2
                
                st.info(f"Mencoba menemukan rekomendasi (Percobaan {attempt + 1}/{max_attempts})...")
                
                # Ubah query instance ke format kategorikal numerik
                query_instance_processed = query_instance.copy()
                
                        # Fungsi untuk menemukan nilai terdekat dalam daftar
                def find_closest_value(value, valid_values, is_age=False):
                    value_float = float(value)
                    valid_floats = [float(x) for x in valid_values]
                    closest = min(valid_floats, key=lambda x: abs(x - value_float))
                    if is_age:
                        return str(int(closest))  # Age harus integer
                    return f"{closest:.1f}"  # 1 desimal untuk height/weight

                # Konversi Age ke kategori terdekat (sebagai integer)
                age = query_instance['Age'].iloc[0]
                age_categories = sorted([str(x) for x in _x_train_encoded['Age'].unique()])
                closest_age = find_closest_value(age, age_categories, is_age=True)
                query_instance_processed.loc[0, 'Age'] = closest_age
                # (debug output removed)
                
                # Konversi Height ke kategori terdekat (dari meter ke cm)
                height = query_instance['Height'].iloc[0]
                height_cm = float(height) * 100  # Konversi m ke cm
                height_categories = sorted([str(x) for x in _x_train_encoded['Height'].unique()])
                closest_height = find_closest_value(height_cm, height_categories)
                query_instance_processed.loc[0, 'Height'] = closest_height
                # (debug output removed)
                
                # Konversi Weight ke kategori terdekat
                weight = query_instance['Weight'].iloc[0]
                weight_categories = sorted([str(x) for x in _x_train_encoded['Weight'].unique()])
                closest_weight = find_closest_value(weight, weight_categories)
                query_instance_processed.loc[0, 'Weight'] = closest_weight
                # (debug output removed)
                
                # (debug output removed)
                
                        # Debug informasi tentang nilai di training data
                my_bar.progress(80, text="Memeriksa nilai kategorikal...")
                
                # (values comparison omitted from UI)
                for col in ALL_CATEGORICAL_COLS:
                    train_vals = sorted(_x_train_encoded[col].unique().astype(str))
                    input_val = query_instance_processed[col].iloc[0]
                
                # Konversi kategorikal sesuai dengan data training
                for col in ALL_CATEGORICAL_COLS:
                    if col in query_instance_processed.columns:
                        # Ambil nilai unik dari training data untuk kolom ini
                        valid_values = _x_train_encoded[col].unique().astype(str)
                        current_value = query_instance_processed[col].iloc[0]
                        
                        if col == 'Gender':
                            # Map nilai Gender
                            gender_map = {'Female': '0', 'Male': '1'}
                            if current_value in gender_map:
                                query_instance_processed.loc[0, col] = gender_map[current_value]
                                
                        elif col in ['FAVC', 'SCC', 'SMOKE', 'family_history_with_overweight']:
                            # Map nilai binary
                            binary_map = {'no': '0', 'yes': '1'}
                            if current_value in binary_map:
                                query_instance_processed.loc[0, col] = binary_map[current_value]
                                
                        elif col == 'CAEC' or col == 'CALC':
                            # Map nilai ordinal CAEC/CALC
                            caec_calc_map = {'no': '0', 'Sometimes': '1', 'Frequently': '2', 'Always': '3'}
                            if current_value in caec_calc_map:
                                query_instance_processed.loc[0, col] = caec_calc_map[current_value]
                                
                        elif col == 'MTRANS':
                            # Map nilai MTRANS
                            mtrans_map = {
                                'Walking': '0',
                                'Public_Transportation': '1',
                                'Bike': '2',
                                'Motorbike': '3',
                                'Automobile': '4'
                            }
                            if current_value in mtrans_map:
                                query_instance_processed.loc[0, col] = mtrans_map[current_value]
                                
                        else:
                            # Untuk fitur ordinal lainnya, pastikan dalam bentuk string
                            query_instance_processed.loc[0, col] = str(current_value)
                
                # (debug output removed)
                
                # Pastikan query_instance_processed memiliki semua kolom yang diharapkan
                query_instance_processed = query_instance_processed.reindex(columns=all_features_list, fill_value='0')
                # Konversi semua nilai menjadi string (DiCE mengharapkan tipe string untuk banyak konfigurasi categorical)
                for col in query_instance_processed.columns:
                    query_instance_processed[col] = query_instance_processed[col].astype(str)

                # Validasi ukuran / bentuk sebelum memanggil DiCE
                if query_instance_processed.shape[1] != len(all_features_list):
                    raise ValueError(f"Query instance has {query_instance_processed.shape[1]} features but expected {len(all_features_list)}. Columns: {query_instance_processed.columns.tolist()}")

                # Generate counterfactuals dengan parameter yang disesuaikan
                try:
                    dice_result = dice_explainer.generate_counterfactuals(
                        query_instance_processed,
                        total_CFs=total_cfs,
                        desired_class=desired_class_index,
                        features_to_vary=features_to_vary_list,
                        permitted_range=permitted_range,
                        proximity_weight=proximity_weight
                    )
                except Exception as inner_e:
                    # Tampilkan traceback lengkap agar memudahkan debugging
                    tb = traceback.format_exc()
                    raise RuntimeError(f"DiCE generation failed: {inner_e}\nTraceback:\n{tb}")
                
                # Periksa apakah counterfactuals ditemukan
                found_valid_cf = False
                if dice_result is not None:
                    if hasattr(dice_result, 'cf_examples_list'):
                        if dice_result.cf_examples_list:
                            if hasattr(dice_result.cf_examples_list[0], 'final_cfs_df'):
                                if dice_result.cf_examples_list[0].final_cfs_df is not None:
                                    if not dice_result.cf_examples_list[0].final_cfs_df.empty:
                                        found_valid_cf = True
                
                if found_valid_cf:
                    st.success(f"Berhasil menemukan rekomendasi pada percobaan {attempt + 1}")
                    break
                else:
                    st.warning(f"Percobaan {attempt + 1} tidak menemukan solusi yang sesuai. Mencoba dengan parameter berbeda...")
                    
            except Exception as e:
                st.warning(f"Percobaan {attempt + 1} gagal: {str(e)}")
                if attempt == max_attempts - 1:  # Jika ini percobaan terakhir
                    raise Exception("Tidak dapat menemukan rekomendasi yang sesuai setelah beberapa percobaan.")
        
        my_bar.progress(100, text="Selesai!")
        return dice_result
        
    except Exception as e:
        my_bar.empty()
        st.error(f"Error saat mencari rekomendasi: {str(e)}")
        return None
    
    finally:
        # Hapus progress bar setelah selesai
        my_bar.empty()

def decode_dice_dataframe(df_dice_output, encoders, all_features_list):
    """
    Fungsi khusus untuk mendekode output DiCE.
    """
    df_decoded = df_dice_output.copy()

    # Mapping numerik ke kategorikal
    categorical_decode_maps = {
        'Gender': {
            '0': 'Female', 
            '1': 'Male'
        },
        'family_history_with_overweight': {
            '0': 'no', 
            '1': 'yes'
        },
        'FAVC': {
            '0': 'no', 
            '1': 'yes'
        },
        'SCC': {
            '0': 'no', 
            '1': 'yes'
        },
        'SMOKE': {
            '0': 'no', 
            '1': 'yes'
        },
        'CAEC': {
            '0': 'no', 
            '1': 'Sometimes', 
            '2': 'Frequently', 
            '3': 'Always'
        },
        'CALC': {
            '0': 'no', 
            '1': 'Sometimes', 
            '2': 'Frequently', 
            '3': 'Always'
        },
        'MTRANS': {
            '0': 'Walking', 
            '1': 'Public_Transportation', 
            '2': 'Bike', 
            '3': 'Motorbike', 
            '4': 'Automobile'
        }
    }

    # Decode kategorikal features
    for col_name, mapping in categorical_decode_maps.items():
        if col_name in df_decoded.columns:
            df_decoded[col_name] = df_decoded[col_name].astype(str).map(mapping)

    # Decode continuous features (convert kembali dari string ke float)
    for col in CONTINUOUS_COLS:
        if col in df_decoded.columns:
            df_decoded[col] = pd.to_numeric(df_decoded[col], errors='coerce').round(2)
            
    # Decode ordinal features (convert kembali dari string ke int)
    for col in ORDINAL_COLS:
        if col in df_decoded.columns:
            df_decoded[col] = pd.to_numeric(df_decoded[col], errors='coerce').round().astype(int)

    # Decode target
    if TARGET_NAME in df_decoded.columns:
        df_decoded[TARGET_NAME] = encoders[TARGET_NAME].inverse_transform(
            pd.to_numeric(df_decoded[TARGET_NAME], errors='coerce').round().astype(int)
        )
            
    return df_decoded[all_features_list + [TARGET_NAME]]

# ======================================================================================
# 5. APLIKASI UTAMA STREAMLIT
# ======================================================================================

# Muat semua aset satu kali di awal
loaded_assets = load_all_assets()
model, encoders, ALL_FEATURES, CLASS_NAMES, x_train_encoded = loaded_assets

# Cek jika *salah satu* aset adalah None
if model is None or encoders is None or ALL_FEATURES is None or CLASS_NAMES is None or x_train_encoded is None:
    st.error("Gagal memuat aset penting. Aplikasi berhenti. Periksa path file aset.")
    st.stop() # Hentikan eksekusi jika aset gagal dimuat
else:
    # Aset berhasil dimuat, inisialisasi LIME
    lime_explainer = initialize_lime_explainer(x_train_encoded, ALL_FEATURES, CLASS_NAMES)

# --- Inisialisasi Session State ---
if 'prediction_done' not in st.session_state:
    st.session_state.prediction_done = False
    st.session_state.user_input_raw = None

# --- UI Sidebar ---
st.sidebar.title("Informasi")
st.sidebar.write("Aplikasi ini memprediksi tingkat obesitas berdasarkan kebiasaan sehari-hari menggunakan model Machine Learning.")
st.sidebar.markdown("---")
if st.sidebar.button("Hapus Cache & Muat Ulang"):
    st.cache_resource.clear()
    st.rerun()

# --- UI Utama ---
st.title("Prediksi Tingkat Obesitas & Rekomendasi Perubahan")
st.markdown("Masukkan data Anda di bawah ini untuk mendapatkan prediksi dan penjelasan mengapa model memberikan hasil tersebut (XAI).")

st.header("Masukkan Data Anda")
col1, col2, col3 = st.columns(3)

# --- Form Input (Sinkron dengan mapping) ---
with col1:
    st.subheader("Fisik & Riwayat")
    gender = st.selectbox('Gender', list(GENDER_MAP.keys()), key='gender')
    age = st.number_input('Umur', min_value=1, max_value=100, value=25, step=1, key='age')
    height_cm = st.number_input('Tinggi Badan (cm)', min_value=100, max_value=250, value=170, help="Masukkan tinggi badan dalam sentimeter.", key='height_cm')
    weight = st.number_input('Berat Badan (kg)', min_value=30, max_value=200, value=70, key='weight')
    family_history_with_overweight = st.selectbox('Riwayat keluarga dengan berat badan berlebih?', list(FAMILY_HISTORY_MAP.keys()), format_func=lambda x: 'Tidak ada' if x == 'no' else 'Ada', key='family')

with col2:
    st.subheader("Kebiasaan Makan")
    favc = st.selectbox('Sering makan makanan tinggi kalori?', list(FAVC_MAP.keys()), format_func=lambda x: 'Tidak Pernah' if x == 'no' else 'Sering (2-3 seminggu)', key='favc')
    fcvc = st.selectbox('Frekuensi makan sayur', [1, 2, 3], format_func=lambda x: {1: 'Tidak Pernah', 2: 'Kadang-kadang', 3: 'Setiap Makan'}[x], key='fcvc')
    ncp = st.number_input('Berapa kali makan utama per hari?', min_value=1, max_value=4, value=3, step=1, key='ncp')
    
    caec_choice = st.selectbox('Makan cemilan di luar waktu makan?', 
                               list(CAEC_MAP.keys()), 
                               format_func=lambda x: {'no':'Tidak Pernah', 'Sometimes':'1-2 hari', 'Frequently':'3-5 hari', 'Always':'6-7 hari'}.get(x, x), 
                               key='caec')
    calc_choice = st.selectbox('Seberapa sering minum alkohol?', 
                               list(CALC_MAP.keys()), 
                               format_func=lambda x: {'no':'Tidak Pernah', 'Sometimes':'Kadang-kadang', 'Frequently':'Sering', 'Always':'Selalu'}.get(x, x), 
                               key='calc')

with col3:
    st.subheader("Aktivitas & Lainnya")
    smoke = st.selectbox('Apakah Anda merokok?', list(SMOKE_MAP.keys()), format_func=lambda x: 'Tidak' if x == 'no' else 'Ya', key='smoke')
    ch2o = st.selectbox('Berapa banyak air yang diminum setiap hari?', [1, 2, 3], format_func=lambda x: {1: '<1L', 2: '1-2L', 3: '>2L'}[x], key='ch2o')
    scc = st.selectbox('Memantau kalori yang Anda makan?', list(SCC_MAP.keys()), format_func=lambda x: 'Tidak' if x == 'no' else 'Ya', key='scc')
    faf = st.selectbox('Aktivitas fisik (menit/hari)?', [0, 1, 2, 3], format_func=lambda x: {0: '< 15 menit', 1: '15-30 menit', 2: '30-60 menit', 3: '> 60 menit'}[x], key='faf')
    tue = st.selectbox('Waktu penggunaan perangkat teknologi', [0, 1, 2], format_func=lambda x: {0: '<1 jam', 1: '1-2 jam', 2: '>2 jam'}[x], key='tue')
    mtrans = st.selectbox('Transportasi yang biasa digunakan', list(MTRANS_MAP.keys()), key='mtrans')


if st.button("Dapatkan Prediksi dan Penjelasan", type="primary"):
    try:
        # Validasi dan konversi input numerik
        height = float(height_cm) / 100.0  # Konversi ke float dan ke meter
        if not (0.5 <= height <= 2.5):  # Validasi range tinggi badan
            st.error("Tinggi badan harus antara 50cm dan 250cm")
            st.stop()
            
        weight = float(weight)  # Konversi ke float
        if not (30 <= weight <= 200):  # Validasi range berat badan
            st.error("Berat badan harus antara 30kg dan 200kg")
            st.stop()
            
        age = int(age)  # Konversi ke integer
        if not (1 <= age <= 100):  # Validasi range umur
            st.error("Umur harus antara 1 dan 100 tahun")
            st.stop()

        # Jika semua validasi berhasil, simpan input
        st.session_state.user_input_raw = {
            'Age': age, 
            'Gender': str(gender), 
            'Height': height, 
            'Weight': weight, 
            'CALC': str(calc_choice),
            'FAVC': str(favc), 
            'FCVC': str(fcvc), 
            'NCP': str(ncp), 
            'SCC': str(scc), 
            'SMOKE': str(smoke),
            'CH2O': str(ch2o), 
            'family_history_with_overweight': str(family_history_with_overweight),
            'FAF': str(faf), 
            'TUE': str(tue), 
            'CAEC': str(caec_choice), 
            'MTRANS': str(mtrans)
        }
        
        st.session_state.prediction_done = True
        st.rerun()
            
    except (ValueError, TypeError) as e:
        st.error(f"Error pada input: Pastikan semua nilai numerik valid. Detail: {str(e)}")
        st.stop()
    try:
        # Validasi dan konversi input numerik
        height = float(height_cm) / 100.0  # Konversi ke float dan ke meter
        if not (0.5 <= height <= 2.5):  # Validasi range tinggi badan
            st.error("Tinggi badan harus antara 50cm dan 250cm")
            st.stop()
            
        weight = float(weight)  # Konversi ke float
        if not (30 <= weight <= 200):  # Validasi range berat badan
            st.error("Berat badan harus antara 30kg dan 200kg")
            st.stop()
            
        age = int(age)  # Konversi ke integer
        if not (1 <= age <= 100):  # Validasi range umur
            st.error("Umur harus antara 1 dan 100 tahun")
            st.stop()

        # Jika semua validasi berhasil, simpan input
        st.session_state.user_input_raw = {
            'Age': age, 
            'Gender': str(gender), 
            'Height': height, 
            'Weight': weight, 
            'CALC': str(calc_choice),
            'FAVC': str(favc), 
            'FCVC': str(fcvc), 
            'NCP': str(ncp), 
            'SCC': str(scc), 
            'SMOKE': str(smoke),
            'CH2O': str(ch2o), 
            'family_history_with_overweight': str(family_history_with_overweight),
            'FAF': str(faf), 
            'TUE': str(tue), 
            'CAEC': str(caec_choice), 
            'MTRANS': str(mtrans)
        }
        
        st.session_state.prediction_done = True
        st.rerun()
            
    except (ValueError, TypeError) as e:
        st.error(f"Error pada input: Pastikan semua nilai numerik valid. Detail: {str(e)}")
        st.stop()# ======================================================================================
# BLOK UTAMA (TAMPILAN HASIL)
# ======================================================================================
if st.session_state.prediction_done:
    # --- Persiapan Data ---
    user_input_raw = st.session_state.user_input_raw
    input_df_processed = preprocess_input_data(user_input_raw, ALL_FEATURES)

    if input_df_processed is None:
        st.error("Gagal memproses data input.")
        st.stop()

    # --- Prediksi (Sinkron dengan Notebook) ---
    prediction_proba = model.predict_proba(input_df_processed[ALL_FEATURES])
    predicted_class_index = np.argmax(prediction_proba[0]) 
    predicted_class = encoders[TARGET_NAME].classes_[predicted_class_index]

    # --- Tampilan Hasil Prediksi ---
    st.header("Hasil Analisis")
    color_map = {"Normal": "green", "Overweight": "orange", "Obesity": "red", "Insufficient": "blue"}
    color = "gray"
    for key, clr in color_map.items():
        if key in predicted_class:
            color = clr
            break
    st.subheader("Prediksi Model")
    st.markdown(f"Berdasarkan data yang Anda berikan, model memprediksi status berat badan Anda adalah: **<span style='color:{color};'>{predicted_class.replace('_', ' ')}</span>**", unsafe_allow_html=True)

    st.subheader("Tingkat Keyakinan Model")
    proba_df = pd.DataFrame(
        prediction_proba,
        columns=CLASS_NAMES,
        index=["Probabilitas"]
    ).T.sort_values("Probabilitas", ascending=False)
    proba_df.columns = ["Tingkat Keyakinan"]
    st.dataframe(proba_df.style.format("{:.2%}"))

    # --- XAI Tabs ---
    st.markdown("---")
    st.subheader("Penjelasan & Rekomendasi (XAI)")
    tab1, tab2 = st.tabs(["Mengapa Model Berpikir Demikian? (LIME)", ":bulb: Bagaimana Cara Mencapai Target? (DiCE)"])

    # --- LIME Tab ---
    with tab1:
        st.subheader("Penjelasan dengan LIME")
        if lime_explainer is not None:
            with st.spinner("Membuat penjelasan LIME..."):
                try:
                    def lime_pred_fn(data):
                        return predict_proba_catboost_for_lime(data, model, ALL_FEATURES)

                    lime_exp = lime_explainer.explain_instance(
                        input_df_processed[ALL_FEATURES].values[0],
                        lime_pred_fn,
                        num_features=10,
                        top_labels=1
                    )

                    st.write("#### Kontribusi Fitur (Plot)")
                    fig = lime_exp.as_pyplot_figure(label=predicted_class_index)
                    st.pyplot(fig, use_container_width=True)
                    st.info("Grafik di atas menunjukkan fitur-fitur yang paling berpengaruh. **Hijau** mendukung prediksi, **Merah** menentang.")

                    st.markdown("---")
                    st.write("#### Detail Penjelasan (Tabel Interaktif)")
                    html = lime_exp.as_html(labels=[predicted_class_index])
                    html_with_bg = f'<div style="background-color:white; color:black; padding: 10px; border-radius: 5px;">{html}</div>'
                    components.html(html_with_bg, height=400, scrolling=True)

                except Exception as e:
                    st.error(f"Gagal membuat penjelasan LIME: {e}")
        else:
            st.error("LIME Explainer tidak dapat diinisialisasi.")

    with tab2:
        st.subheader("Rekomendasi Perubahan")
        if predicted_class == 'Normal_Weight':
            st.success("Selamat! Berat badan Anda saat ini sudah **Normal**.")
            st.balloons()
        else:
            desired_target_class = 'Normal_Weight'
            st.info(f"Rekomendasi di bawah ini bertujuan untuk membantu Anda mencapai target: **{desired_target_class.replace('_', ' ')}**")

            with st.spinner(f"Mencari rekomendasi untuk mencapai **{desired_target_class.replace('_', ' ')}**..."):
                try:
                    desired_target_index = list(CLASS_NAMES).index(desired_target_class)
                    
                    dice_result = get_dice_recommendations(
                        x_train_encoded, 
                        model, 
                        encoders, 
                        input_df_processed,
                        desired_target_index,
                        ALL_FEATURES
                    )

                    if dice_result and dice_result.cf_examples_list and dice_result.cf_examples_list[0].final_cfs_df is not None:
                        cf_df_output = dice_result.cf_examples_list[0].final_cfs_df
                        
                        if not cf_df_output.empty:
                            st.markdown("---")
                            # Judul ini mirip dengan di gambar Anda
                            st.write(f"#### Kumpulan Rekomendasi (Counterfactuals) untuk Mencapai: **{desired_target_class.replace('_', ' ')}**")
                            
                            # --- MODIFIKASI DIMULAI ---
                            
                            # 1. Decode hasil counterfactual (rekomendasi)
                            cf_df_decoded = decode_dice_dataframe(cf_df_output, encoders, ALL_FEATURES)
                            
                            # 2. Siapkan data asli Anda (untuk baris pertama tabel)
                            # Data ini menggunakan input string asli dari user
                            orig_df = pd.DataFrame([user_input_raw])
                            # Tambahkan kolom prediksi asli
                            orig_df['NObeyesdad'] = predicted_class.replace('_', ' ')
                            
                            # 3. Gabungkan data asli dengan data rekomendasi
                            # Ini akan membuat tabel persis seperti di gambar Anda
                            combined_df = pd.concat([orig_df, cf_df_decoded], ignore_index=True)

                            # 4. Tampilkan tabel gabungan
                            st.info("Baris pertama adalah data asli Anda. Baris-baris berikutnya adalah rekomendasi perubahan.")
                            
                            # Pastikan urutan kolom sesuai
                            display_columns = ALL_FEATURES + [TARGET_NAME]
                            st.dataframe(combined_df[display_columns])
                            
                        else:
                            st.warning(f"DiCE tidak dapat menemukan rekomendasi perubahan untuk mencapai **{desired_target_class.replace('_', ' ')}**.")
                    else:
                        st.warning(f"Maaf, DiCE tidak dapat menemukan rekomendasi perubahan yang realistis untuk mencapai **{desired_target_class.replace('_', ' ')}**.")

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat menghasilkan rekomendasi DiCE: {e}")