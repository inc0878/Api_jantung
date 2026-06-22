from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from catboost import CatBoostClassifier

# Inisialisasi Aplikasi
app = FastAPI(title="Heart Disease Prediction API")

# Muat Model AI
model = CatBoostClassifier()
model.load_model("catboost_model.cbm")

# Definisikan Struktur Data Input (Mencegah error jika user memasukkan data salah)
class PatientData(BaseModel):
    age: float
    sex: float
    chest_pain_type: float
    resting_bp_s: float
    cholesterol: float
    fasting_blood_sugar: float
    resting_ecg: float
    max_heart_rate: float
    exercise_angina: float
    oldpeak: float
    st_slope: float

@app.get("/")
def read_root():
    return {"message": "API Prediksi Penyakit Jantung Aktif!"}

@app.post("/predict")
def predict_heart_disease(data: PatientData):
    # Ubah input menjadi DataFrame dengan nama kolom aslinya
    df = pd.DataFrame([{
        'age': data.age,
        'sex': data.sex,
        'chest pain type': data.chest_pain_type,
        'resting bp s': data.resting_bp_s,
        'cholesterol': data.cholesterol,
        'fasting blood sugar': data.fasting_blood_sugar,
        'resting ecg': data.resting_ecg,
        'max heart rate': data.max_heart_rate,
        'exercise angina': data.exercise_angina,
        'oldpeak': data.oldpeak,
        'ST slope': data.st_slope
    }])
    
    # Lakukan Prediksi
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1] # Probabilitas kelas 1 (Sakit)
    
    return {
        "prediksi_kelas": int(prediction),
        "probabilitas_sakit": round(float(probability) * 100, 2),
        "status": "Terindikasi Penyakit Jantung" if prediction == 1 else "Normal"
    }