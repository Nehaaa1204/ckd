import streamlit as st
import pandas as pd
import joblib
from fpdf import FPDF
from io import BytesIO

# Load model
import joblib

model_path = r"C:\Users\adish\OneDrive\Desktop\chronic_kidney_prediction\ckd_model.pkl"
model = joblib.load(model_path)

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
        .centered {
            display: flex;
            justify-content: center;
            margin-top: 30px;
        }
        .btn-start {
            background-color: #2e8b57;
            color: white;
            padding: 15px 30px;
            font-size: 18px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
        }
    </style>
""", unsafe_allow_html=True)

# Session page routing
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

    if st.button("🔍 Predict CKD"):
        prediction = model.predict(input_data)[0]
        proba_ckd = model.predict_proba(input_data)[0][1]

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
