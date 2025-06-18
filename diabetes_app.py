import streamlit as st
import numpy as np
import pickle
import os
import base64

# Page configuration
st.set_page_config(
    page_title="AI Diabetes Risk Assessment",
    layout="centered",
    page_icon="🩺"
)

# Light/Dark Mode Toggle
mode = st.selectbox("🌗 Choose Theme Mode", ["Light", "Dark"])
if mode == "Dark":
    st.markdown("<style>body { background-color: #0e1117; color: white; }</style>", unsafe_allow_html=True)
else:
    st.markdown("<style>body { background-color: white; color: black; }</style>", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    return pickle.load(open("diabetesmodel.pkl", "rb"))

model = load_model()

# Background setup
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
        }}
        </style>
    """, unsafe_allow_html=True)

set_background("diabetes background.jpeg")

# Title
st.markdown("<h1 style='text-align: center;'>🩺 AI Diabetes Risk Assessment</h1>", unsafe_allow_html=True)
st.subheader("Know your risk using health indicators")

# Tips
with st.expander("📊 Recommended Input Ranges"):
    st.markdown("""
    - **Glucose (mg/dL)**: 70 – 140  
    - **Blood Pressure (mmHg)**: 80 – 120  
    - **BMI (Body Mass Index)**: 18.5 – 24.9  
    - **Age (years)**: 10 – 90  
    """)

# Inputs
st.markdown("### 📝 Enter Your Health Information")
glucose = st.number_input("🧪 Glucose (mg/dL)", min_value=50.0, max_value=300.0, step=1.0)
blood_pressure = st.number_input("💓 Blood Pressure (mmHg)", min_value=50.0, max_value=200.0, step=1.0)
bmi = st.number_input("⚖️ BMI", min_value=10.0, max_value=60.0, step=0.1)
age = st.number_input("🎂 Age", min_value=5, max_value=120, step=1)

# Extra inputs (optional)
with st.expander("➕ Additional Health Metrics"):
    pregnancies = st.number_input("🤰 Pregnancies", min_value=0, max_value=15)
    insulin = st.number_input("💉 Insulin Level", min_value=0.0, max_value=900.0)
    skin = st.number_input("🩻 Skin Thickness", min_value=0.0, max_value=100.0)

# Action buttons
col1, col2 = st.columns(2)
with col1:
    predict = st.button("🌟 Predict", use_container_width=True)
with col2:
    reset = st.button("🔄 Reset", use_container_width=True)

# Reset form
if reset:
    st.experimental_rerun()

# Prediction logic
if predict:
    input_data = np.array([[glucose, blood_pressure, bmi, age]])
    prediction = model.predict_proba(input_data)
    score = prediction[0][1] * 100

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

    # Report download
    report = f"""
    🩺 Diabetes Risk Report

    Input Summary:
    - Glucose: {glucose}
    - Blood Pressure: {blood_pressure}
    - BMI: {bmi}
    - Age: {age}
    - Risk Score: {score:.2f}%

    Outcome: {"Low risk ✅" if score < 50 else "High risk ⚠️"}
    """
    st.download_button("📤 Download Report", report, file_name="diabetes_report.txt")

# Optional next features
st.markdown("---")
st.caption("Let us know if you'd like:")
st.markdown("- 🌓 Light/dark mode toggle (already added!)  
- 📈 More detailed feedback (Risk Score shown!)  
- 🩺 Additional health metrics (included in expandable section)")
