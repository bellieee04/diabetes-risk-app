import streamlit as st
import numpy as np
import pickle
import base64
import datetime

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
            background-image: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)), url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
            color: white;
        }}
        </style>
    """, unsafe_allow_html=True)

set_background("diabetes background.jpeg")

# --- Title ---
st.markdown("<h1 style='text-align: center;'>🩺 AI Diabetes Risk Assessment</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Your personalised assistant for better health awareness</h4>", unsafe_allow_html=True)

# --- Health Tips Slider ---
with st.expander("💡 Health Tips of the Day"):
    st.markdown("""
    - 🏃‍♂️ Move your body at least 30 minutes daily  
    - 🍎 Choose fruits over sugar  
    - 💧 Stay hydrated  
    - 🧘 Practice deep breathing to reduce stress  
    - 🩺 Monitor health metrics regularly  
    """)

# --- Login Section ---
st.markdown("## 👤 Personal Details")
name = st.text_input("Full Name")
gender = st.selectbox("Gender", ["Prefer not to say", "Male", "Female", "Other"])
height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, step=0.5)
weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, step=0.5)

# BMI calculation
bmi = round(weight / ((height / 100) ** 2), 2) if height > 0 else 0
if bmi:
    if bmi < 18.5:
        bmi_status = "Underweight"
    elif 18.5 <= bmi < 25:
        bmi_status = "Normal"
    elif 25 <= bmi < 30:
        bmi_status = "Overweight"
    else:
        bmi_status = "Obese"
    st.markdown(f"**Calculated BMI**: `{bmi}` ({bmi_status})")

# --- Health Info Inputs ---
st.markdown("## 📝 Enter Your Health Indicators")
glucose = st.number_input("🧪 Glucose (mg/dL)", min_value=50.0, max_value=300.0, step=1.0, help="Normal range: 70–140 mg/dL")
blood_pressure = st.number_input("💓 Blood Pressure (mmHg)", min_value=50.0, max_value=200.0, step=1.0, help="Normal: 80–120 mmHg")
age = st.number_input("🎂 Age", min_value=5, max_value=120, step=1)

# --- Additional Optional Metrics ---
with st.expander("➕ Additional Health Metrics (Optional)"):
    st.markdown("Provide if known, else you may skip.")
    st.number_input("🤰 Pregnancies", min_value=0, max_value=15, help="Relevant for women of childbearing age")
    st.number_input("💉 Insulin Level (µU/mL)", min_value=0.0, max_value=900.0, help="Typically 16–166 µU/mL")
    st.number_input("🩻 Skin Thickness (mm)", min_value=0.0, max_value=100.0, help="Used in clinical screening for insulin resistance")

# --- Custom Styled Button ---
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #FF4B4B;
    color: white;
    font-weight: bold;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 1.1em;
    border: none;
    transition: 0.3s ease-in-out;
}
div.stButton > button:hover {
    background-color: #d43f3f;
    transform: scale(1.02);
}
</style>
""", unsafe_allow_html=True)

# --- Prediction Logic ---
if st.button("🌟 Predict My Diabetes Risk"):
    input_data = np.array([[glucose, blood_pressure, bmi, age]])
    prediction = model.predict_proba(input_data)
    score = prediction[0][1] * 100

    st.markdown("---")
    if score < 50:
        st.success(f"✅ Low Risk! (Score: {score:.2f}%)")
        st.image("healthy background.jpeg", caption="💚 Keep up the healthy lifestyle!", use_container_width=True)
        with st.expander("💡 Healthy Living Tips"):
            st.markdown("""
            - ✅ Stay active daily  
            - 🍽️ Eat balanced meals  
            - 🚫 Limit sugar intake  
            - 🩺 Get regular checkups  
            - 😴 Prioritise quality sleep  
            """)
    else:
        st.error(f"⚠️ High Risk! (Score: {score:.2f}%)")
        st.image("unhealthy background.jpeg", caption="❤️ Time to prioritise your health", use_container_width=True)
        with st.expander("💡 Tips to Lower Risk"):
            st.markdown("""
            - 🚫 Avoid sweetened beverages  
            - 🥗 Eat more veggies & whole grains  
            - 🏃‍♀️ Exercise daily (30 mins)  
            - 🔬 Monitor glucose levels  
            - 🧘 Practise relaxation  
            """)

    # --- Downloadable Report ---
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"""
🩺 AI Diabetes Risk Assessment Report

Date: {now}
Name: {name}
Gender: {gender}
Age: {age}

Height: {height} cm
Weight: {weight} kg
BMI: {bmi} ({bmi_status})

Glucose Level: {glucose} mg/dL
Blood Pressure: {blood_pressure} mmHg

Estimated Diabetes Risk: {score:.2f}%% → {"Low ✅" if score < 50 else "High ⚠️"}

Thank you for using our AI Health Assistant. Stay healthy!
"""
    st.download_button("📤 Download Personal Health Report", report, file_name=f"{name}_diabetes_report.txt")
