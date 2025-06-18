import streamlit as st
import numpy as np
import pickle
import os
from datetime import datetime
import pandas as pd

# Page config
st.set_page_config(
    page_title="AI Diabetes Risk Assessment",
    layout="centered",
    page_icon="🩺"
)

# ======================= Custom Theme Toggle ===========================
mode = st.selectbox("🌗 Choose Theme Mode", ["Dark", "Light"], index=0)

if mode == "Dark":
    st.markdown("""
        <style>
        .stApp {
            background-color: #0e1117;
            color: white;
        }
        .stButton>button {
            background-color: #2c2f35;
            color: white;
            border-radius: 10px;
            padding: 0.5em 2em;
            font-weight: 600;
            border: none;
        }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .stApp {
            background-color: #f0f2f6;
            color: black;
        }
        .stButton>button {
            background-color: #e5e5e5;
            color: black;
            border-radius: 10px;
            padding: 0.5em 2em;
            font-weight: 600;
            border: none;
        }
        </style>
    """, unsafe_allow_html=True)

# ======================= Model ===========================
@st.cache_resource
def load_model():
    return pickle.load(open("diabetesmodel.pkl", "rb"))

model = load_model()

# ======================= Background ===========================
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = f.read().hex()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
            filter: brightness(0.6);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("diabetes background.jpeg")

# ======================= App Title ===========================
st.markdown("<h1 style='text-align: center;'>🩺 AI Diabetes Risk Assessment</h1>", unsafe_allow_html=True)
st.subheader("Know your risk using health indicators")

# ======================= Ranges ===========================
with st.expander("📊 Recommended Input Ranges"):
    st.markdown("""
    - **Glucose (mg/dL)**: 70 – 140  
    - **Blood Pressure (mmHg)**: 80 – 120  
    - **BMI**: 18.5 – 24.9  
    - **Age**: 10 – 90  
    """)

# ======================= Inputs ===========================
st.markdown("### 📝 Enter Your Health Information")

glucose = st.number_input("🥼 Glucose (mg/dL)", min_value=50.0, max_value=300.0, step=1.0)
blood_pressure = st.number_input("💓 Blood Pressure (mmHg)", min_value=50.0, max_value=200.0, step=1.0)
bmi = st.number_input("⚖️ BMI", min_value=10.0, max_value=60.0, step=0.1)
age = st.number_input("🎂 Age", min_value=5, max_value=120, step=1)

# ======================= Additional Health Metrics ===========================
with st.expander("➕ Additional Health Metrics"):
    st.markdown("Future metrics like insulin, skin thickness, etc. coming soon!")

# ======================= Buttons ===========================
col1, col2 = st.columns(2)

result = None
graph_data = []

with col1:
    if st.button("🔆 Predict"):
        input_data = np.array([[glucose, blood_pressure, bmi, age]])
        prediction = model.predict(input_data)
        result = prediction[0]

        # Graph tracking
        log_file = "diabetes_predictions.csv"
        new_entry = pd.DataFrame({
            "Timestamp": [datetime.now()],
            "Glucose": [glucose],
            "BP": [blood_pressure],
            "BMI": [bmi],
            "Age": [age],
            "Prediction": ["No Diabetes" if result == 0 else "Diabetes"]
        })

        if os.path.exists(log_file):
            df = pd.read_csv(log_file)
            df = pd.concat([df, new_entry], ignore_index=True)
        else:
            df = new_entry

        df.to_csv(log_file, index=False)
        graph_data = df

with col2:
    if st.button("🔄 Reset"):
        st.experimental_rerun()

# ======================= Result Output ===========================
if result is not None:
    if result == 0:
        st.success("✅ You are NOT likely to have diabetes.")
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
        st.error("⚠️ You ARE likely to have diabetes.")
        st.image("unhealthy background.jpeg", caption="❤️ Take steps to lower your risk", use_container_width=True)
        with st.expander("💡 Suggestions to reduce risk"):
            st.markdown("""
            - 🚫 Cut back on sugary drinks  
            - 🌾 Eat fibre-rich food  
            - 🏃‍♀️ Exercise 30 mins/day  
            - 🔬 Monitor glucose  
            - 🧘 Reduce stress  
            """)

    # Downloadable Report
    with open("diabetes_predictions.csv", "rb") as file:
        st.download_button("📤 Download Report", file, file_name="diabetes_report.csv", mime="text/csv")

# ======================= Tracking Graph ===========================
if os.path.exists("diabetes_predictions.csv"):
    st.markdown("### 📅 Diabetes Risk Tracker")
    graph_data = pd.read_csv("diabetes_predictions.csv")
    st.line_chart(graph_data[["Glucose", "BP", "BMI"]])
