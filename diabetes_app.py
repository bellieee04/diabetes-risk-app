import streamlit as st
import numpy as np
import pickle
import base64
from datetime import datetime
import matplotlib.pyplot as plt

# --- Page Config ---
st.set_page_config(
    page_title="AI Diabetes Risk Assessment",
    layout="centered",
    page_icon="🩺"
)

# --- Load Model ---
@st.cache_resource
def load_model():
    return pickle.load(open("diabetesmodel.pkl", "rb"))

model = load_model()

# --- Background Setup ---
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(0, 0, 0, 0.55), rgba(0, 0, 0, 0.55)), url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }}
        </style>
    """, unsafe_allow_html=True)

set_background("diabetes background.jpeg")

# --- Custom Button Styling ---
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #FF4B4B;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 1.1em;
    border: none;
    transition: background-color 0.3s;
}
div.stButton > button:hover {
    background-color: #ff3333;
}
</style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown("<h1 style='text-align: center;'>🩺 AI Diabetes Risk Assessment</h1>", unsafe_allow_html=True)
st.subheader("Know your risk using health indicators")

# --- Personal Info ---
st.markdown("### 🧑‍⚕️ Personal Information")
name = st.text_input("Full Name")
gender = st.radio("Gender", ["Male", "Female", "Prefer not to say"])
height = st.number_input("📏 Height (cm)", min_value=50.0, max_value=250.0)
weight = st.number_input("⚖️ Weight (kg)", min_value=10.0, max_value=300.0)

# --- Input Tips ---
with st.expander("📊 Recommended Input Ranges"):
    st.markdown("""
    - **Glucose (mg/dL)**: 70 – 140  
    - **Blood Pressure (mmHg)**: 80 – 120  
    - **BMI (Body Mass Index)**: 18.5 – 24.9  
    - **Age (years)**: 10 – 90  
    """)

# --- Main Inputs ---
st.markdown("### 📝 Enter Your Health Information")
glucose = st.number_input("🧪 Glucose (mg/dL)", min_value=50.0, max_value=300.0)
blood_pressure = st.number_input("💓 Blood Pressure (mmHg)", min_value=50.0, max_value=200.0)
age = st.number_input("🎂 Age", min_value=5, max_value=120)

# --- Additional Inputs ---
with st.expander("➕ Additional Health Metrics"):
    st.number_input("🤰 Pregnancies", min_value=0, max_value=15, help="Higher pregnancy counts may increase risk.")
    st.number_input("💉 Insulin Level", min_value=0.0, max_value=900.0, help="Normal fasting insulin: 2.6–24.9 µIU/mL")
    st.number_input("🩻 Skin Thickness", min_value=0.0, max_value=100.0, help="Normal: 20–35 mm for triceps skinfold")

# --- Prediction Logic ---
if st.button("🌟 Predict"):
    bmi = weight / ((height / 100) ** 2)
    if bmi < 18.5:
        bmi_status = "Underweight"
    elif 18.5 <= bmi < 24.9:
        bmi_status = "Normal"
    elif 25 <= bmi < 29.9:
        bmi_status = "Overweight"
    else:
        bmi_status = "Obese"

    input_data = np.array([[glucose, blood_pressure, bmi, age]])
    prediction = model.predict_proba(input_data)
    score = prediction[0][1] * 100
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Result
    if score < 50:
        st.success(f"✅ {name}, you are NOT likely to have diabetes. (Risk Score: {score:.2f}%)")
        st.image("healthy background.jpeg", caption="💚 Keep up the healthy lifestyle!", use_container_width=True)
    else:
        st.error(f"⚠️ {name}, you ARE likely to have diabetes. (Risk Score: {score:.2f}%)")
        st.image("unhealthy background.jpeg", caption="❤️ Take steps to lower your risk", use_container_width=True)

    st.markdown(f"**Calculated BMI:** `{bmi:.2f}` – **Status:** `{bmi_status}`")

    # Risk Pie Chart
    fig, ax = plt.subplots()
    ax.pie([score, 100 - score], labels=["Risk", "No Risk"], autopct='%1.1f%%',
           colors=["#FF4B4B", "#90EE90"], startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

    # Downloadable Report
    report = f"""
🩺 Diabetes Risk Report – {timestamp}

Patient Name: {name}
Gender: {gender}
Age: {age}

Input Summary:
- Glucose: {glucose}
- Blood Pressure: {blood_pressure}
- Weight: {weight} kg
- Height: {height} cm
- BMI: {bmi:.2f} ({bmi_status})
- Risk Score: {score:.2f}%

Outcome: {"Low risk ✅" if score < 50 else "High risk ⚠️"}
"""
    st.download_button("📤 Download Report", report, file_name="diabetes_report.txt")
