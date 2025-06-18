import streamlit as st
import numpy as np
import pickle
from PIL import Image
import base64

# ---------- CONFIGURATION ----------
st.set_page_config(page_title="AI Diabetes Risk Assessment", layout="centered")

# ---------- FUNCTION: BACKGROUND IMAGE ----------
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            transition: background 0.5s ease-in-out;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Show default background
set_background("diabetes background.jpeg")

# ---------- FUNCTION: LOAD MODEL ----------
@st.cache_resource
def load_model():
    with open("diabetesmodel.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ---------- HEADER ----------
st.markdown("<h1 style='text-align: center;'>🩺 AI Diabetes Risk Assessment</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Know your risk using health indicators</h3>", unsafe_allow_html=True)
st.write("")

# ---------- EXPANDER: GUIDE ----------
with st.expander("📊 Recommended Input Ranges"):
    st.markdown("""
    - 🔬 **Glucose**: 70 – 140 mg/dL  
    - 💓 **Blood Pressure**: 80 – 120 mmHg  
    - ⚖️ **BMI**: 18.5 – 24.9  
    - 🎂 **Age**: All ages accepted, but risk increases with age
    """)

# ---------- USER INPUTS ----------
st.subheader("📝 Enter Your Health Information")

glucose = st.number_input("🔬 Glucose (mg/dL)", min_value=50.0, max_value=300.0, step=1.0, help="Normal range: 70–140")
blood_pressure = st.number_input("💓 Blood Pressure (mmHg)", min_value=50.0, max_value=200.0, step=1.0, help="Normal range: 80–120")
bmi = st.number_input("⚖️ BMI", min_value=10.0, max_value=60.0, step=0.1, help="Normal range: 18.5–24.9")
age = st.number_input("🎂 Age", min_value=5, max_value=120, step=1, help="Age in years")

# ---------- PREDICTION ----------
if st.button("🧠 Predict"):
    input_data = np.array([[glucose, blood_pressure, bmi, age]])
    prediction = model.predict(input_data)

    if prediction[0] == 0:
        st.success("✅ You are NOT likely to have diabetes.")
        st.image("healthy background.jpeg", caption="💚 Keep up the healthy lifestyle!", use_column_width=True)
        with st.expander("💡 Health Tips"):
            st.markdown("""
            - 🏃 Stay active daily  
            - 🥗 Eat balanced meals  
            - 🚫 Limit sugar intake  
            - 🩺 Get regular checkups  
            - 😴 Prioritise sleep  
            """)
    else:
        st.error("⚠️ You ARE likely to have diabetes.")
        st.image("unhealthy background.jpeg", caption="❤️ Take steps to lower your risk", use_column_width=True)
        with st.expander("💡 Suggestions to Improve"):
            st.markdown("""
            - 🚫 Reduce sugary drinks  
            - 🌾 Eat more fibre-rich foods  
            - 🏃‍♀️ Exercise 30 mins daily  
            - 🔍 Monitor blood sugar  
            - 🧘‍♀️ Manage stress levels  
            """)
