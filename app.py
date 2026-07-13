import streamlit as st
import requests
import pandas as pd

st.title("Diabetes Risk Prediction")
st.write("Enter patient details below to get a prediction.")
st.set_page_config(page_title="Diabetes Risk Prediction", layout="wide")

# Load and display feature importance
importance_df = pd.read_csv('feature_importance.csv')

with st.expander("ℹ️ What factors does this model consider most important?"):
    display_df = importance_df.copy()
    display_df['Importance'] = (display_df['Importance'] * 100).round(1)
    st.bar_chart(display_df.set_index('Feature'))

# Input fields for patient data
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
with col2:
    glucose = st.number_input("Glucose", min_value=0, max_value=300, value=100)

col3, col4 = st.columns(2)

with col3:
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
with col4:
    skin_thickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=20.0)

col5, col6 = st.columns(2)

with col5:
    insulin = st.number_input("Insulin", min_value=0.0, max_value=900.0, value=80.0)
with col6:
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)

col7, col8 = st.columns(2)

with col7:
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
with col8:
    age = st.number_input("Age", min_value=1, max_value=120, value=30)


st.markdown("---")

if st.button("Predict"):
    input_data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }

    response = requests.post("http://127.0.0.1:8000/predict", json=input_data)

    if response.status_code == 200:
        result = response.json()
        st.subheader(f"Prediction: {result['prediction']}")
        st.write(f"Probability of diabetes: {result['probability']}")
    else:
        st.error("Something went wrong. Please try again.")

