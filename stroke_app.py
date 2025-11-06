from fastapi import FastAPI
import joblib
import numpy as np

# load once at cold start
model = joblib.load('stroke_model.joblib')
pipe, thr = model['pipe'], model['threshold']

# Feature ordering from training
feature_order = [
    "age", "avg_glucose_level", "bmi", "hypertension", "heart_disease",
    "smoking_status", "gender", "ever_married", "Residence_type", "work_type"
]

# ---------- prediction helper ----------
def predict_from_user_inputs(age, height_cm, weight_kg, blood_sugar,
                             hypertension, heart_disease,
                             smoking_status, gender, marital_status,
                             residence_type, work_type):
    # compute BMI
    bmi = weight_kg / ((height_cm/100)**2)

    # build raw feature row
    row = {
        "age": age,
        "avg_glucose_level": blood_sugar,
        "bmi": bmi,
        "hypertension": int(hypertension),
        "heart_disease": int(heart_disease),
        "smoking_status": smoking_status,
        "gender": gender,
        "ever_married": marital_status,
        "Residence_type": residence_type,
        "work_type": work_type
    }

    # convert dict to list in correct order
    raw_input = np.array([[row[f] for f in feature_order]])

    # transform using the pipeline preprocessor
    X_processed = pipe.named_steps['preprocessor'].transform(raw_input)

    # predict
    p = pipe.named_steps['model'].predict_proba(X_processed)[0][1]
    y = int(p >= thr)

    return round(float(p), 4), y, round(float(row['bmi']), 2)

# ---------- FastAPI app ----------
app = FastAPI(title="Stroke Risk API")

@app.get("/")
def root():
    return {"message": "Send a POST request to /predict with patient data."}

@app.post("/predict")
def predict(payload: dict):
    prob, pred, bmi = predict_from_user_inputs(**payload)
    return {
        "probability": prob,
        "prediction": pred,
        "calculated_BMI": bmi
    }

from mangum import Mangum
handler = Mangum(app)
