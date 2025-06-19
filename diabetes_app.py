import streamlit as st
import numpy as np
import pickle
import base64

# --- Page Setup ---
st.set_page_config(page_title="AI Diabetes Risk Assessment", layout="centered", page_icon="🩺")

# --- Load Model ---
@st.cache_resource
def load_model():
    return pickle.load(open("diabetesmodel.pkl", "rb"))
model = load_model()

# --- Background ---
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.55), rgba(0, 0, 0, 0.55)), url("data:image/jpeg;base64,{encoded}");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }}
    </style>
    """, unsafe_allow_html=True)
set_background("diabetes background.jpeg")

# --- Title ---
st.markdown("<h1 style='text-align: center;'>🩺 AI Diabetes Risk Assessment</h1>", unsafe_allow_html=True)
st.subheader("Know your risk using health indicators")

# --- Personal Info ---
st.markdown("## 🧑‍⚕️ Personal Information")
name = st.text_input("Full Name")
gender = st.radio("Gender", ["Male", "Female", "Prefer not to say"])
height_cm = st.number_input("📝 Height (cm)", min_value=100, max_value=250)
weight_kg = st.number_input("⚖️ Weight (kg)", min_value=20.0, max_value=300.0)

bmi = round(weight_kg / ((height_cm / 100) ** 2), 2) if height_cm > 0 else 0

# --- Input Tips ---
with st.expander("📊 Recommended Input Ranges"):
    st.markdown("""
    - **Glucose (mg/dL)**: 70 – 140 *(Normal)*
    - **Blood Pressure (mmHg)**: 80 – 120 *(Normal)*
    - **BMI (kg/m²)**: 18.5 – 24.9 *(Healthy weight)*
    - **Age (years)**: 10 – 90 *(Adolescents to elderly)*
    """)

# --- Core Health Inputs ---
st.markdown("## 🧾 Enter Your Health Information")
glucose = st.number_input("🧪 Glucose (mg/dL)", min_value=50.0, max_value=300.0, help="70–140 is considered normal")
blood_pressure = st.number_input("💕 Blood Pressure (mmHg)", min_value=50.0, max_value=200.0, help="80–120 is normal range")
age = st.number_input("🎂 Age", min_value=5, max_value=120, step=1, help="You must be at least 5 years old")

# --- Additional Metrics ---
with st.expander("➕ Additional Health Metrics"):
    st.number_input("🤰 Pregnancies", min_value=0, max_value=15, help="Total number of past pregnancies")
    st.number_input("🦩 Insulin Level", min_value=0.0, max_value=900.0, help="Normal: 16–166 mIU/mL")
    st.number_input("🧻 Skin Thickness", min_value=0.0, max_value=100.0, help="Normal range: 10–50 mm")

# --- Custom Styled Button ---
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #ff4747;
    color: white;
    font-weight: bold;
    border-radius: 12px;
    height: 3.2em;
    width: 100%;
    font-size: 1.15em;
    border: none;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
    transition: 0.3s ease;
}
div.stButton > button:hover {
    background-color: #e83e3e;
    box-shadow: 0px 6px 14px rgba(0,0,0,0.4);
}
</style>
""", unsafe_allow_html=True)

# --- Prediction ---
if st.button("🌟 Predict"):
    input_data = np.array([[glucose, blood_pressure, bmi, age]])
    prediction = model.predict_proba(input_data)
    score = prediction[0][1] * 100

    if score < 50:
        st.success(f"✅ {name}, you are NOT likely to have diabetes. (Risk Score: {score:.2f}%)")
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
        st.error(f"⚠️ {name}, you ARE likely to have diabetes. (Risk Score: {score:.2f}%)")
        st.image("unhealthy background.jpeg", caption="❤️ Take steps to lower your risk", use_container_width=True)
        with st.expander("💡 Suggestions to reduce risk"):
            st.markdown("""
            - 🚫 Cut back on sugary drinks
            - 🌾 Eat fibre-rich food
            - 🏃‍♀️ Exercise 30 mins/day
            - 🔬 Monitor glucose
            - 🧘 Reduce stress
            """)

    # --- Download Report ---
    report = f"""
🩺 Diabetes Risk Report

Personal Info:
- Name: {name}
- Gender: {gender}
- Height: {height_cm} cm
- Weight: {weight_kg} kg
- BMI: {bmi}

Health Data:
- Glucose: {glucose}
- Blood Pressure: {blood_pressure}
- Age: {age}

Risk Score: {score:.2f}%
Outcome: {"Low risk ✅" if score < 50 else "High risk ⚠️"}
"""
    st.download_button("📤 Download Report", report, file_name="diabetes_report.txt")
