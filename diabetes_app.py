import streamlit as st
import numpy as np
import pickle
import pandas as pd
from datetime import datetime
import os

# ---------- Page Setup ----------
st.set_page_config(page_title="AI Diabetes Risk Assessment", layout="centered")

# ---------- Background Styling ----------
def set_background(image_file, overlay_opacity=0.6):
    st.markdown(f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(0,0,0,{overlay_opacity}), rgba(0,0,0,{overlay_opacity})),
                        url("data:image/jpeg;base64,{image_file}");
            background-size: cover;
        }}
        </style>
    """, unsafe_allow_html=True)

def get_base64(file_path):
    import base64
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

default_bg = get_base64("diabetes background.jpeg")
healthy_bg = get_base64("healthy background.jpeg")
unhealthy_bg = get_base64("unhealthy background.jpeg")

set_background(default_bg)

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    with open("diabetesmodel.pkl", 'rb') as file:
        return pickle.load(file)

model = load_model()

# ---------- UI ----------
st.markdown("## 🩺 AI Diabetes Risk Assessment")
st.markdown("### Know your risk using health indicators")

# Expandable guide
with st.expander("📊 Recommended Input Ranges"):
    st.markdown("""
    - **Glucose:** 70–140 mg/dL  
    - **Blood Pressure:** 80–120 mmHg  
    - **BMI:** 18.5–24.9  
    - **Age:** 10–90 years  
    """)

# Input form
st.markdown("### 📝 Enter Your Health Information")

col1, col2 = st.columns([3, 1])
with col1:
    glucose = st.number_input("🧪 Glucose (mg/dL)", min_value=50.0, max_value=300.0, value=100.0, step=1.0)
    blood_pressure = st.number_input("💓 Blood Pressure (mmHg)", min_value=50.0, max_value=200.0, value=90.0, step=1.0)
    bmi = st.number_input("⚖️ BMI", min_value=10.0, max_value=60.0, value=22.0, step=0.1)
    age = st.number_input("🎂 Age", min_value=5, max_value=120, value=30, step=1)

# Session state to hold report data
if 'report_data' not in st.session_state:
    st.session_state.report_data = []

# ---------- Predict Button ----------
if st.button("🧠 Predict"):
    input_data = np.array([[glucose, blood_pressure, bmi, age]])
    prediction = model.predict(input_data)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.report_data.append({
        "Timestamp": now,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "BMI": bmi,
        "Age": age,
        "Prediction": "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
    })

    # Change background based on result
    if prediction[0] == 0:
        st.success("✅ You are NOT likely to have diabetes.")
        set_background(healthy_bg)
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
        set_background(unhealthy_bg)
        with st.expander("💡 Suggestions to Reduce Risk"):
            st.markdown("""
            - 🚫 Cut back on sugary drinks  
            - 🌾 Eat fibre-rich food  
            - 🏃‍♀️ Exercise 30 mins/day  
            - 🔬 Monitor glucose  
            - 🧘 Reduce stress  
            """)

# ---------- Reset Button ----------
if st.button("🔁 Reset"):
    st.experimental_rerun()

# ---------- Download Report ----------
if st.session_state.report_data:
    df = pd.DataFrame(st.session_state.report_data)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📤 Download Report", csv, "diabetes_report_history.csv", "text/csv")

    st.markdown("### 📈 Health Tracking Graph")
    st.line_chart(df.set_index("Timestamp")[["Glucose", "BloodPressure", "BMI"]])
