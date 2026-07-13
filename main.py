from fastapi import FastAPI
import joblib
from pydantic import BaseModel

app = FastAPI()
# Load trained pipeline once, when the API starts
model = joblib.load('diabetes_model.pkl')


class PatientData(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int


import pandas as pd

@app.post("/predict")
def predict(data: PatientData):
    input_data = pd.DataFrame([{
        "Pregnancies": data.Pregnancies,
        "Glucose": data.Glucose,
        "BloodPressure": data.BloodPressure,
        "SkinThickness": data.SkinThickness,
        "Insulin": data.Insulin,
        "BMI": data.BMI,
        "DiabetesPedigreeFunction": data.DiabetesPedigreeFunction,
        "Age": data.Age
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    result = "Diabetic" if prediction == 1 else "Not Diabetic"

    return {
        "prediction": result,
        "probability": round(float(probability), 2)
    }
