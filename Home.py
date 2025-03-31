import streamlit as st
import pandas as pd
import joblib
import os
from fpdf import FPDF
from io import BytesIO

# Load model and preprocessing files
model_path = os.path.join(os.path.dirname(__file__), "ckd_model.pkl")
scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
features_path = os.path.join(os.path.dirname(__file__), "selected_features.pkl")

if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(features_path):
    st.error("Model, scaler, or feature selection file is missing!")
    st.stop()

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
selected_features = joblib.load(features_path)

# Set page config
st.set_page_config(page_title="CKD Predictor", layout="centered")

# Inject custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Background styling
st.markdown("""
    <style>
        body {
            background-image: url("https://cdn.pixabay.com/photo/2018/03/30/11/48/kidney-3279900_960_720.jpg");
            background-size: cover;
            background-attachment: fixed;
        }
        .container {
            background-color: rgba(255,255,255,0.9);
            padding: 40px;
            border-radius: 15px;
            max-width: 900px;
            margin: auto;
            margin-top: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# Page Routing
if "page" not in st.session_state:
    st.session_state.page = "home"

# PDF Generation
def generate_pdf_report(data_dict, prediction, probability):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Chronic Kidney Disease Prediction Report", ln=1, align="C")
    pdf.ln(10)

    for key, val in data_dict.items():
        pdf.cell(200, 10, txt=f"{key}: {val}", ln=1)

    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Prediction: {'CKD Detected' if prediction == 1 else 'No CKD Detected'}", ln=1)
    pdf.cell(200, 10, txt=f"CKD Probability: {probability*100:.2f}%", ln=1)

    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    return BytesIO(pdf_bytes)

# HOME PAGE
if st.session_state.page == "home":
    st.markdown("""
        <div class="container">
            <h1 style='text-align:center;'>Chronic Kidney Disease Prediction</h1>
            <p style='text-align:center;'>
                This tool helps predict CKD using a trained machine learning model.<br>
                Fill in your medical test details to assess your risk and generate a report.
            </p>
        </div>
    """, unsafe_allow_html=True)

    if st.button("Start Prediction"):
        st.session_state.page = "predict"

# PREDICTION PAGE
elif st.session_state.page == "predict":
    st.markdown("<div class='container'>", unsafe_allow_html=True)
    st.header("Basic Information")
    age = st.number_input("Age", min_value=0)
    bp = st.number_input("Blood Pressure", min_value=0)
    sg = st.number_input("Specific Gravity", min_value=0.0, format="%.2f")
    al = st.number_input("Albumin", min_value=0)
    su = st.number_input("Sugar", min_value=0)

    st.header("Lab Test Results")
    rbc = st.selectbox("Red Blood Cells", ["normal", "abnormal"])
    pc = st.selectbox("Pus Cell", ["normal", "abnormal"])
    pcc = st.selectbox("Pus Cell Clumps", ["present", "notpresent"])
    ba = st.selectbox("Bacteria", ["present", "notpresent"])
    hemo = st.number_input("Hemoglobin")
    dm = st.selectbox("Diabetes Mellitus", ['yes', 'no'])
    cad = st.selectbox("Coronary Artery Disease", ['yes', 'no'])
    sod = st.number_input("Sodium")
    pot = st.number_input("Potassium")
    wc = st.number_input("White Blood Cell Count")
    rc = st.number_input("Red Blood Cell Count")
    appet = st.selectbox("Appetite", ['good', 'poor'])
    pcv = st.number_input("Packed Cell Volume")
    ane = st.selectbox("Anemia", ['yes', 'no'])
    htn = st.selectbox("Hypertension", ['yes', 'no'])
    bgr = st.number_input("Blood Glucose Random")
    pe = st.selectbox("Pedal Edema", ['yes', 'no'])
    sc = st.number_input("Serum Creatinine")
    bu = st.number_input("Blood Urea")

    input_data = pd.DataFrame([{
        'age': age, 'bp': bp, 'sg': sg, 'al': al, 'su': su,
        'rbc': rbc, 'pc': pc, 'pcc': pcc, 'ba': ba, 'hemo': hemo,
        'dm': dm, 'cad': cad, 'sod': sod, 'pot': pot, 'wc': wc,
        'rc': rc, 'appet': appet, 'pcv': pcv, 'ane': ane, 'htn': htn,
        'bgr': bgr, 'pe': pe, 'sc': sc, 'bu': bu
    }])

    # Encoding categorical features
    encode_map = {
        'rbc': {'normal': 0, 'abnormal': 1},
        'pc': {'normal': 0, 'abnormal': 1},
        'pcc': {'notpresent': 0, 'present': 1},
        'ba': {'notpresent': 0, 'present': 1},
        'dm': {'no': 0, 'yes': 1},
        'cad': {'no': 0, 'yes': 1},
        'appet': {'poor': 0, 'good': 1},
        'ane': {'no': 0, 'yes': 1},
        'htn': {'no': 0, 'yes': 1},
        'pe': {'no': 0, 'yes': 1}
    }

    for col, mapping in encode_map.items():
        input_data[col] = input_data[col].map(mapping)

    # Ensure input follows the selected features
    missing_features = set(selected_features) - set(input_data.columns)
    if missing_features:
        st.error(f"Missing features in input: {missing_features}")
        st.stop()

    # Reorder and scale input data
    input_data = input_data[selected_features]
    input_data = input_data.apply(pd.to_numeric, errors='coerce')
    input_scaled = scaler.transform(input_data)

    if st.button("🔍 Predict CKD"):
        prediction = model.predict(input_scaled)[0]
        proba_ckd = model.predict_proba(input_scaled)[0][1]

        if prediction == 1:
            st.error("🟥 Prediction: CKD Detected")
        else:
            st.success(f"🟩 Prediction: No CKD Detected\n🧪 CKD Risk Probability: {proba_ckd * 100:.2f}%")

        report_pdf = generate_pdf_report(input_data.iloc[0].to_dict(), prediction, proba_ckd)
        st.download_button(
            label="📥 Download PDF Report",
            data=report_pdf,
            file_name="ckd_report.pdf",
            mime='application/pdf'
        )

    if st.button("⬅ Back to Home"):
        st.session_state.page = "home"

    st.markdown("</div>", unsafe_allow_html=True)
