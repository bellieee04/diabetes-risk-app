import streamlit as st
import numpy as np
import pickle
import base64

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

set_background("diabetes background.jpeg")  # Make sure this file is in the correct location!

# --- Title ---
st.markdown("<h1 style='text-align: center;'>🩺 AI Diabetes Risk Assessment</h1>", unsafe_allow_html=True)
st.subheader("Know your risk using health indicators")

# --- Personal Info ---
st.markdown("### 👤 Personal Information")
name = st.text_input("Full Name")
gender = st.radio("Gender", ["Male", "Female", "Prefer not to say"])
height_cm = st.number_input("📏 Height (cm)", min_value=100, max_value=250)
weight_kg = st.number_input("⚖️ Weight (kg)", min_value=30.0, max_value=200.0)

# --- Input Tips ---
with st.expander("📊 Recommended Input Ranges"):
    st.markdown("""
    - **Glucose (mg/dL)**: 70 – 140  
    - **Blood Pressure (mmHg)**: 80 – 120  
    - **BMI (Body Mass Index)**: 18.5 – 24.9  
    - **Age (years)**: 10 – 90  
    """)

# --- Health Inputs ---
st.markdown("### 📝 Enter Your Health Information")
glucose = st.number_input("🧪 Glucose (mg/dL)", min_value=50.0, max_value=300.0, step=1.0)
blood_pressure = st.number_input("💓 Blood Pressure (mmHg)", min_value=50.0, max_value=200.0, step=1.0)
age = st.number_input("🎂 Age", min_value=5, max_value=120, step=1)

# --- BMI Calculation ---
height_m = height_cm / 100 if height_cm else 0.01
bmi = round(weight_kg / (height_m ** 2), 2)
st.text_input("Calculated BMI", value=bmi, disabled=True)

# --- Optional Inputs ---
with st.expander("➕ Additional Health Metrics"):
    st.number_input("🤰 Pregnancies (Only if applicable)", min_value=0, max_value=15, help="Number of times pregnant (affects insulin resistance)")
    st.number_input("💉 Insulin Level", min_value=0.0, max_value=900.0, help="Normal range: 2-25 μIU/mL")
    st.number_input("🩻 Skin Thickness", min_value=0.0, max_value=100.0, help="Often measured in mm during health checks")

# --- Styled Button ---
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

# --- Predict Logic ---
if st.button("🌟 Predict"):
    input_data = np.array([[glucose, blood_pressure, bmi, age]])
    prediction = model.predict_proba(input_data)
    score = prediction[0][1] * 100  # %

    if score < 50:
        st.success(f"✅ You are NOT likely to have diabetes. (Risk Score: {score:.2f}%)")
        st.image("healthy background.jpeg", caption="💚 Keep up the healthy lifestyle!", use_container_width=True)
        with st.expander("💡 Health Suggestions"):
            st.markdown("""
            - ✅ Stay active daily  
            - 🍽️ Eat balanced meals  
            - 🚫 Limit sugar intake  
            - 🩺 Get regular checkups  
            - 😴 Sleep well  
            """)
    else:
        st.error(f"⚠️ You ARE likely to have diabetes. (Risk Score: {score:.2f}%)")
        st.image("unhealthy background.jpeg", caption="❤️ Take steps to lower your risk", use_container_width=True)
        with st.expander("💡 Suggestions to reduce risk"):
            st.markdown("""
            - 🚫 Cut back on sugary drinks  
            - 🌾 Eat fibre-rich food  
            - 🏃‍♀️ Exercise 30 mins/day  
            - 🔬 Monitor glucose  
            - 🧘 Reduce stress  
            """)

    # --- Report ---
    report = f"""
🩺 Diabetes Risk Report

Name: {name}
Gender: {gender}
Height: {height_cm} cm
Weight: {weight_kg} kg
BMI: {bmi}

Input Summary:
- Glucose: {glucose} mg/dL
- Blood Pressure: {blood_pressure} mmHg
- Age: {age}
- Risk Score: {score:.2f}%

Outcome: {"Low risk ✅" if score < 50 else "High risk ⚠️"}
"""
    st.download_button("📤 Download Report", report, file_name=f"{name}_diabetes_report.txt")
