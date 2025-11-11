import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import os
from catboost import CatBoostClassifier
from lime.lime_tabular import LimeTabularExplainer
import dice_ml
from dice_ml import Dice
import warnings
# Menekan peringatan yang tidak relevan selama menjalankan Streamlit
warnings.filterwarnings('ignore', category=FutureWarning) 
warnings.filterwarnings('ignore', category=DeprecationWarning) 

# ======================================================================================
# 1. KONFIGURASI & SETUP APLIKASI
# ======================================================================================
st.set_page_config(
    page_title="Prediksi Obesitas (CatBoost + XAI)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Path ke Aset Model (Sesuaikan jika struktur folder berbeda) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "catboost_model_v3.pkl")
TARGET_ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder_y_v3.pkl")
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, "feature_names_v3.pkl")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names_y_v3.pkl")
DATA_PATH = r"C:\Users\LENOVO\Documents\DENICO\Skripsi\Dataset\combined_dataset.csv"

# --- Konfigurasi Tipe Fitur (Berdasarkan Notebook) ---
TARGET_NAME = 'NObeyesdad'
CONTINUOUS_COLS = ['Age', 'Height', 'Weight']
ORDINAL_INT_COLS = ['FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
CATEGORICAL_STRING_COLS = [
    'Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE', 'family_history_with_overweight',
    'CAEC', 'MTRANS'
]
ALL_FEATURES = CONTINUOUS_COLS + ORDINAL_INT_COLS + CATEGORICAL_STRING_COLS 
ALL_CATEGORICAL_COLS = ORDINAL_INT_COLS + CATEGORICAL_STRING_COLS

# --- Daftar Manual untuk UI ---
UI_MAPS = {
    'Gender': ['Female', 'Male'],
    'family_history_with_overweight': ['yes', 'no'],
    'FAVC': ['yes', 'no'],
    'SCC': ['yes', 'no'],
    'SMOKE': ['yes', 'no'],
    'CAEC': ['Sometimes', 'Frequently', 'Always', 'no'],
    'CALC': ['Sometimes', 'Frequently', 'Always', 'no'],
    'MTRANS': ['Public_Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike'],
    'FCVC': ['1', '2', '3'], 
    'NCP': ['1', '2', '3', '4'], 
    'CH2O': ['1', '2', '3'], 
    'FAF': ['0', '1', '2', '3'], 
    'TUE': ['0', '1', '2'], 
}

# --- Reverse Mapping untuk Encoder (LIME/DiCE) ---
INV_UI_MAPS = {
    'Gender': {'Female': 0, 'Male': 1},
    'family_history_with_overweight': {'no': 0, 'yes': 1},
    'FAVC': {'no': 0, 'yes': 1},
    'SCC': {'no': 0, 'yes': 1},
    'SMOKE': {'no': 0, 'yes': 1},
    'CAEC': {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3},
    'CALC': {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3},
    'MTRANS': {'Public_Transportation': 0, 'Walking': 1, 'Automobile': 2, 'Motorbike': 3, 'Bike': 4}, 
}


# ======================================================================================
# 2. FUNGSI LOADING ASET & WRAPPER DICE
# ======================================================================================

@st.cache_resource
def load_all_assets():
    """Memuat model CatBoost, encoders, dan data training yang sudah di-encoded."""
    try:
        model = joblib.load(MODEL_PATH)
        target_encoder = joblib.load(TARGET_ENCODER_PATH)
        all_features = joblib.load(FEATURE_NAMES_PATH)
        class_names = joblib.load(CLASS_NAMES_PATH)
        encoders = {TARGET_NAME: target_encoder}

        df_raw = pd.read_csv(DATA_PATH, usecols=all_features + [TARGET_NAME])
        
        for col in CONTINUOUS_COLS: 
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce').astype(float) 
        for col in ORDINAL_INT_COLS: 
            df_raw[col] = df_raw[col].astype(str)
        for col in CATEGORICAL_STRING_COLS:
            df_raw[col] = df_raw[col].astype(str)

        df_raw = df_raw.dropna(subset=all_features).reset_index(drop=True)
        
        # --- Buat x_train_encoded (data numerik) untuk LIME/DiCE Explainer ---
        x_train_encoded = df_raw.copy()
        
        for col, mapping in INV_UI_MAPS.items():
            x_train_encoded[col] = x_train_encoded[col].map(mapping)
        
        for col in ORDINAL_INT_COLS:
             x_train_encoded[col] = pd.to_numeric(x_train_encoded[col], errors='coerce').round()
        
        for col in CONTINUOUS_COLS:
             x_train_encoded[col] = pd.to_numeric(x_train_encoded[col], errors='coerce').astype(float)

        x_train_encoded = x_train_encoded.dropna(subset=all_features)

        # Konversi akhir ke FLOAT (untuk konsistensi LIME)
        for col in ALL_CATEGORICAL_COLS: 
            x_train_encoded[col] = x_train_encoded[col].astype(float) 
        
        x_train_encoded = x_train_encoded[all_features]
        
        return model, encoders, all_features, class_names, x_train_encoded, df_raw

    except FileNotFoundError as e:
        st.error(f"Gagal memuat file aset. Pastikan semua file (pkl dan csv) tersedia di folder yang benar: {e}")
        return None, None, None, None, None, None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat aset penting: {e}")
        return None, None, None, None, None, None

# --- Custom Wrapper untuk CatBoost di DiCE ---
class DiceCatBoostWrapper:
    def __init__(self, model, feature_names, continuous_features, categorical_features):
        self.model = model
        self.feature_names = feature_names
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features

    def predict_proba(self, X):
        X_copy = X.copy()
        for col in self.continuous_features:
            X_copy[col] = pd.to_numeric(X_copy[col], errors='coerce').astype(float) 
        for col in self.categorical_features:
             X_copy[col] = X_copy[col].astype(str)
        X_copy = X_copy[self.feature_names]
        return self.model.predict_proba(X_copy)


# ======================================================================================
# 3. FUNGSI XAI (LIME & DICE)
# ======================================================================================

# --- LIME Explainer ---
@st.cache_resource
def initialize_lime_explainer(_X_train_encoded, all_features_list, class_names_list):
    """Inisialisasi LIME explainer menggunakan data yang sudah di-encoded (numerik)."""
    
    # PERBAIKAN: Ubah _X_train_encoded menjadi array NumPy (akan bertipe float)
    training_values = _X_train_encoded[all_features_list].values 

    categorical_feature_indices = [
        _X_train_encoded.columns.get_loc(col) for col in ALL_CATEGORICAL_COLS if col in _X_train_encoded.columns
    ]
    
    categorical_names_map = {}
    
    for col, mapping in INV_UI_MAPS.items():
        if col in all_features_list:
            col_idx = all_features_list.index(col)
            # Nilai kategorikal di LIME harus float (0.0, 1.0)
            categorical_names_map[col_idx] = sorted([float(v) for v in mapping.values()]) 
            
    for col in ORDINAL_INT_COLS:
        if col in all_features_list:
            col_idx = all_features_list.index(col)
            # Nilai ordinal di LIME harus float (1.0, 2.0)
            categorical_names_map[col_idx] = [float(v) for v in UI_MAPS[col]]
    
    # --- PERBAIKAN KUNCI ---
    # 1. Kembalikan `training_data=training_values` untuk memperbaiki TypeError
    # 2. Hapus `discretize_continuous=False` agar LIME menggunakan default (True), 
    #    yang akan mengatasi error 'data is numpy array of floating point'
    explainer = LimeTabularExplainer(
        training_data=training_values,    # <-- DIKEMBALIKAN
        feature_names=all_features_list,
        class_names=class_names_list,
        categorical_features=categorical_feature_indices,
        categorical_names=categorical_names_map,
        mode='classification',
        random_state=42
        # `discretize_continuous=False` DIHAPUS (membiarkan LIME melakukan binning)
    )
    return explainer

def get_lime_explanation(explainer, model, input_df_encoded, class_names):
    """Menghasilkan penjelasan LIME dan merender visualisasinya."""
    
    explanation = explainer.explain_instance(
        input_df_encoded.values[0].astype(float), # Pastikan input instance bertipe float
        model.predict_proba,
        num_features=10,
        num_samples=1000
    )

    html_content = explanation.as_html(labels=class_names).replace('<h1>', '###').replace('<h2>', '####')
    return html_content

# --- DiCE Explainer ---
def initialize_dice_explainer(model, encoders, _df_raw, all_features_list, class_names_list):
    """Inisialisasi objek DiCE."""
    df_dice = _df_raw.copy()
    
    permitted_range = {}
    features_to_vary = []
    
    EXCLUDE_VARY = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight']
    
    for col in all_features_list:
        if col not in EXCLUDE_VARY:
            features_to_vary.append(col)
            if col in UI_MAPS:
                 permitted_range[col] = UI_MAPS[col] 
    
    feature_types = {}
    for col in all_features_list:
        if col in CONTINUOUS_COLS:
            feature_types[col] = 'continuous'
        elif col in ALL_CATEGORICAL_COLS:
             feature_types[col] = 'categorical'

    data_interface = dice_ml.Data(
        dataframe=df_dice[all_features_list + [TARGET_NAME]], 
        continuous_features=CONTINUOUS_COLS,
        outcome_name=TARGET_NAME,
        type_of_features=feature_types,
        outcome_type='categorical', 
        outcome_classes=class_names_list 
    )
    
    wrapped_model = DiceCatBoostWrapper(
        model, all_features_list, CONTINUOUS_COLS, ALL_CATEGORICAL_COLS
    )
    model_interface = dice_ml.Model(model=wrapped_model, backend="sklearn", model_type='classifier')
    
    dice_explainer = Dice(data_interface, model_interface, method="random")
    
    return dice_explainer, features_to_vary, permitted_range

def get_dice_recommendations(explainer, input_df_raw, predicted_class, desired_class, features_to_vary, permitted_range):
    """Menghasilkan rekomendasi DiCE."""
    try:
        if predicted_class == desired_class:
            return None, "Hasil prediksi sudah sesuai target yang diinginkan."
        
        dice_result = explainer.generate_counterfactuals(
            input_df_raw.copy(), 
            total_CFs=5, 
            desired_class=desired_class,
            features_to_vary=features_to_vary,
            permitted_range=permitted_range
        )
        
        return dice_result, None

    except Exception as e:
        return None, f"Terjadi kesalahan saat menghasilkan rekomendasi DiCE: {e}"

# --- Fungsi untuk memformat Output DiCE ---
def decode_dice_dataframe(dice_result, all_features_list):
    """Dekode output DiCE ke format yang lebih mudah dibaca."""
    try:
        cf_df_output = dice_result.cf_examples_list[0].final_cfs_df
        if cf_df_output is None or cf_df_output.empty:
            return None

        if TARGET_NAME in cf_df_output.columns:
            cf_df_output[TARGET_NAME] = cf_df_output[TARGET_NAME].str.replace('_', ' ')
        
        for col in CONTINUOUS_COLS:
            if col in cf_df_output.columns:
                cf_df_output[col] = pd.to_numeric(cf_df_output[col], errors='coerce').round(2)
        
        return cf_df_output[all_features_list + [TARGET_NAME]]
        
    except Exception as e:
        st.error(f"Gagal mendekode output DiCE: {e}")
        return None


# ======================================================================================
# 4. APLIKASI UTAMA STREAMLIT
# ======================================================================================
# Muat semua aset satu kali di awal
loaded_assets = load_all_assets()
if loaded_assets[0] is None:
    st.stop()
    
model, encoders, ALL_FEATURES, CLASS_NAMES_LIST, x_train_encoded, df_raw = loaded_assets
CLASS_NAMES_DISPLAY = [name.replace('_', ' ') for name in CLASS_NAMES_LIST]

# Inisialisasi Explainer LIME dan DiCE (satu kali)
lime_explainer = initialize_lime_explainer(x_train_encoded, ALL_FEATURES, CLASS_NAMES_LIST)
dice_explainer, dice_features_to_vary, dice_permitted_range = initialize_dice_explainer(
    model, encoders, df_raw, ALL_FEATURES, CLASS_NAMES_LIST 
)


st.title("Prediksi Tingkat Obesitas (CatBoost + XAI) 📊")
st.markdown("---")

# --- Bagian Input Pengguna (Sidebar) ---
with st.sidebar:
    st.header("Input Data Pasien")
    st.markdown("Masukkan data untuk memprediksi tingkat obesitas dan menganalisis faktor-faktornya.")
    
    # 1. Input Numerik
    st.subheader("Data Fisik")
    age = st.number_input("Age (Tahun)", min_value=14, max_value=75, value=25, step=1)
    height = st.number_input("Height (Tinggi Badan dalam cm)", min_value=120, max_value=200, value=170, step=1)
    weight = st.number_input("Weight (Berat Badan dalam kg)", min_value=39, max_value=173, value=75, step=1)
    
    # 2. Input Kategorikal/Ordinal (Selectbox)
    st.subheader("Kebiasaan & Gaya Hidup")
    gender = st.selectbox("Gender", options=UI_MAPS['Gender'])
    family_history_with_overweight = st.selectbox(
        "Family History with Overweight?", 
        options=UI_MAPS['family_history_with_overweight'], 
        format_func=lambda x: 'Ada' if x == 'yes' else 'Tidak Ada'
    )
    FAVC = st.selectbox("FAVC (Sering makan makanan tinggi kalori?)", options=UI_MAPS['FAVC'], format_func=lambda x: 'Ya' if x == 'yes' else 'Tidak')
    FCVC = st.selectbox("FCVC (Frekuensi makan sayur) (1=Tidak Pernah, 2=Kadang, 3=Selalu)", options=UI_MAPS['FCVC'])
    NCP = st.selectbox("NCP (Berapa kali makan utama/hari) (1-4)", options=UI_MAPS['NCP'])
    CAEC = st.selectbox("CAEC (Seberapa sering makan cemilan)", options=UI_MAPS['CAEC'])
    SMOKE = st.selectbox("SMOKE (Merokok?)", options=UI_MAPS['SMOKE'], format_func=lambda x: 'Ya' if x == 'yes' else 'Tidak')
    CH2O = st.selectbox("CH2O (Berapa banyak air/hari) (1=<1L, 2=1-2L, 3=>2L)", options=UI_MAPS['CH2O'])
    SCC = st.selectbox("SCC (Memantau kalori?)", options=UI_MAPS['SCC'], format_func=lambda x: 'Ya' if x == 'yes' else 'Tidak')
    FAF = st.selectbox("FAF (Aktivitas fisik/minggu) (0=Tidak pernah, 1=1-2 hari, 2=2-4 hari, 3=4-5 hari/Hampir setiap hari)", options=UI_MAPS['FAF'])
    TUE = st.selectbox("TUE (Waktu penggunaan perangkat teknologi) (0=0-2 jam, 1=3-5 jam, 2=>5 jam)", options=UI_MAPS['TUE'])
    CALC = st.selectbox("CALC (Seberapa sering minum alkohol)", options=UI_MAPS['CALC'])
    MTRANS = st.selectbox("MTRANS (Transportasi)", options=UI_MAPS['MTRANS'])

    predict_button = st.button("Prediksi dan Analisis", type="primary")

# --- Fungsi Prediksi ---
def get_predictions(model, raw_input_df):
    """Menghasilkan prediksi dan probabilitas."""
    raw_input_df = raw_input_df[ALL_FEATURES].copy()
    
    for col in CONTINUOUS_COLS:
        raw_input_df[col] = pd.to_numeric(raw_input_df[col], errors='coerce').astype(float)
    for col in ALL_CATEGORICAL_COLS:
        raw_input_df[col] = raw_input_df[col].astype(str)

    prediction_proba = model.predict_proba(raw_input_df)[0]
    predicted_index = np.argmax(prediction_proba)
    predicted_class_name = CLASS_NAMES_LIST[predicted_index]
    
    return predicted_class_name, prediction_proba[predicted_index], prediction_proba

# --- Fungsi untuk menyiapkan input untuk LIME (Encoded/Numerical) ---
def get_lime_input(raw_input_df):
    """Mengubah input mentah (string) menjadi numerik/encoded untuk LIME Explainer."""
    lime_input = raw_input_df.copy()
    
    # 1. Encode String Kategorikal
    for col, mapping in INV_UI_MAPS.items():
        lime_input[col] = lime_input[col].map(mapping)
        
    # 2. Convert Ordinal/Count Features (yang saat ini string) ke Numerik
    for col in ORDINAL_INT_COLS:
        lime_input[col] = pd.to_numeric(lime_input[col], errors='coerce').round()
        
    # 3. Pastikan kolom kontinu adalah float
    for col in CONTINUOUS_COLS:
        lime_input[col] = pd.to_numeric(lime_input[col], errors='coerce').astype(float)
        
    # 4. Konversi Akhir ke FLOAT (Wajib agar konsisten dengan array numpy float)
    for col in ALL_CATEGORICAL_COLS:
         lime_input[col] = lime_input[col].astype(float)
        
    return lime_input[ALL_FEATURES]


# --- Tampilkan Hasil di Main Area ---
if predict_button:
    # 1. Kumpulkan Input
    user_input_raw = pd.DataFrame([{
        'Age': age, 'Gender': gender, 'Height': height, 'Weight': weight,
        'CALC': CALC, 'FAVC': FAVC, 'FCVC': FCVC, 'NCP': NCP, 
        'SCC': SCC, 'SMOKE': SMOKE, 'CH2O': CH2O, 'family_history_with_overweight': family_history_with_overweight,
        'FAF': FAF, 'TUE': TUE, 'CAEC': CAEC, 'MTRANS': MTRANS
    }])
    
    # 2. Prediksi
    predicted_class, predicted_prob, prediction_proba_all = get_predictions(model, user_input_raw)
    predicted_class_display = predicted_class.replace('_', ' ')
    
    st.subheader(f"Hasil Prediksi Tingkat Obesitas: **{predicted_class_display}** (Keyakinan: {predicted_prob:.2%})")
    st.markdown("---")
    
    # Tampilkan Data Input
    st.write("#### Data Input Anda")
    st.dataframe(user_input_raw[ALL_FEATURES])

    # 3. Tampilkan XAI dalam Tabs (Perbedaan Tampilan)
    tab1, tab2 = st.tabs(["📊 Analisis LIME (Faktor Penting)", "🔄 Rekomendasi DiCE (Counterfactuals)"])

    # === TAB 1: LIME ===
    with tab1:
        if lime_explainer:
            st.write("#### Faktor yang Paling Mempengaruhi Prediksi (LIME)")
            st.info(f"LIME menjelaskan mengapa model memprediksi **{predicted_class_display}** untuk input spesifik Anda.")
            
            # Siapkan input LIME (numerik/encoded)
            try:
                user_input_encoded = get_lime_input(user_input_raw)
            except Exception as e:
                st.error(f"Gagal menyiapkan input untuk LIME. Pastikan semua pilihan kategori valid: {e}")
                user_input_encoded = None

            # Hasilkan dan tampilkan LIME
            if user_input_encoded is not None:
                try:
                    lime_html = get_lime_explanation(model, user_input_encoded, CLASS_NAMES_DISPLAY)
                    components.html(lime_html, height=500, scrolling=True)
                except Exception as e:
                     st.error(f"Gagal menghasilkan penjelasan LIME: {e}")
        else:
            st.warning("LIME explainer belum terinisialisasi.")

    # === TAB 2: DiCE ===
    with tab2:
        if dice_explainer:
            st.write("#### Saran untuk Mencapai Kelas Berat Badan Lain (DiCE)")
            
            target_class_options_raw = [c for c in CLASS_NAMES_LIST if c != predicted_class]
            target_options_map = {name.replace('_', ' '): name for name in target_class_options_raw}
            
            desired_target_display = st.selectbox(
                "Pilih Tingkat Obesitas Target:", 
                options=list(target_options_map.keys()),
                index=0 if target_class_options_raw else None
            )
            desired_target_class = target_options_map.get(desired_target_display) 
            
            if desired_target_class:
                st.info(f"Rekomendasi perubahan minimum pada kebiasaan yang dapat mengubah prediksi menjadi: **{desired_target_display}**")
                
                with st.spinner(f"Mencari rekomendasi untuk mencapai '{desired_target_display}'..."):
                    dice_result, error_message = get_dice_recommendations(
                        dice_explainer, 
                        user_input_raw, 
                        predicted_class, 
                        desired_target_class, 
                        dice_features_to_vary, 
                        dice_permitted_range
                    )

                if error_message:
                    st.error(f"DiCE Error: {error_message}")
                elif dice_result:
                    cf_df_decoded = decode_dice_dataframe(dice_result, ALL_FEATURES)
                    if cf_df_decoded is not None and not cf_df_decoded.empty:
                        st.write("##### Rekomendasi Perubahan (Counterfactuals)")
                        st.dataframe(cf_df_decoded)

                        st.markdown("---")
                        st.write("##### Ringkasan Perubahan")
                        
                        summary_data = []
                        original_instance = user_input_raw.iloc[0].to_dict()
                        
                        for cf_idx, cf_row in cf_df_decoded.iterrows():
                            changes = {}
                            for feature in dice_features_to_vary:
                                original_value = original_instance[feature]
                                cf_value = cf_row[feature]
                                if str(original_value) != str(cf_value):
                                    changes[feature] = f"Ubah dari **{original_value}** menjadi **{cf_value}**"
                            
                            if changes:
                                changes_list = [f"- {k.replace('_', ' ').capitalize()}: {v}" for k, v in changes.items()]
                                summary_data.append(f"**Rekomendasi {cf_idx + 1}** (Target: {cf_row[TARGET_NAME]}):\n" + "\n".join(changes_list))

                        if summary_data:
                            for summary in summary_data:
                                st.markdown(summary)
                        else:
                             st.warning("DiCE tidak dapat menemukan perubahan fitur non-fisik yang menghasilkan kelas target yang diinginkan.")
                    else:
                        st.warning(f"DiCE tidak dapat menemukan rekomendasi perubahan untuk mencapai **{desired_target_display}**.")
                else:
                    st.info(f"Prediksi saat ini sudah **{predicted_class_display}**. Tidak ada rekomendasi perubahan yang diperlukan.")
        else:
            st.warning("DiCE explainer belum terinisialisasi.")