import streamlit as st
import numpy as np
import pickle
import kagglehub
import os

# Page configuration
st.set_page_config(page_title="AI Diabetes Risk Assessment", layout="centered")

# Load model from KaggleHub
@st.cache_resource
def load_model():
    path = kagglehub.model_download("gsaha123/diabetes-risk-assessment/scikitLearn/2gd/1")
    model_file = os.path.join(path, "diabetesmodel.pkl")
    return pickle.load(open(model_file, 'rb'))

model = load_model()

# Title
st.title("🩺 AI Diabetes Risk Assessment")

# Background (optional, not natively supported — can use HTML/CSS with `st.markdown` but let’s focus on function)

# Input form
st.header("Enter your health info")

glucose = st.number_input("Glucose", min_value=50.0, max_value=300.0, step=1.0)
blood_pressure = st.number_input("Blood Pressure", min_value=50.0, max_value=200.0, step=1.0)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, step=0.1)
age = st.number_input("Age", min_value=5, max_value=120, step=1)

if st.button("Predict"):
    input_data = np.array([[glucose, blood_pressure, bmi, age]])
    prediction = model.predict(input_data)

    if prediction[0] == 0:
        st.success("✅ You are NOT likely to have diabetes.")
        with st.expander("💡 Health Suggestions"):
            st.markdown("""
            - ✅ Stay active daily  
            - 🍽️ Eat balanced meals  
            - 🚫 Limit sugar intake  
            - 🩺 Get regular checkups  
            - 😴 Sleep well  
            """)
    else:
        st.error("⚠️ You ARE likely to have diabetes.")
        with st.expander("💡 Suggestions to reduce risk"):
            st.markdown("""
            - 🚫 Cut back on sugary drinks  
            - 🌾 Eat fibre-rich food  
            - 🏃‍♀️ Exercise 30 mins/day  
            - 🔬 Monitor glucose  
            - 🧘 Reduce stress  
            """)
