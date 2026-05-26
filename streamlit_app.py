import os
import joblib
import pandas as pd
import streamlit as st
import google.generativeai as genai

from monitor import log_prediction, check_drift

MODEL_VERSION = "v1.0"
MODEL_PATH = "xgboost_model.pkl"
DATA_PATH = "Lead Scoring.csv"

st.set_page_config(
    page_title="Lead Prioritisation CRM Tool",
    page_icon="📊",
    layout="wide"
)

# -------------------------------
# GEMINI SETUP
# -------------------------------
gemini_available = False

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel("models/gemini-flash-lite-latest")
    gemini_available = True
except Exception:
    gemini_model = None
    gemini_available = False

# -------------------------------
# CSS
# -------------------------------
st.markdown("""
<style>
.main { background-color: #f6f8fb; }

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    background: linear-gradient(135deg, #0f172a, #1e3a8a);
    padding: 35px;
    border-radius: 24px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0px 14px 35px rgba(15, 23, 42, 0.25);
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 8px;
}

.hero p {
    font-size: 17px;
    color: #dbeafe;
}

.card {
    background: white;
    padding: 24px;
    border-radius: 20px;
    box-shadow: 0px 8px 24px rgba(15, 23, 42, 0.08);
    margin-bottom: 18px;
}

.ai-card {
    background: linear-gradient(135deg, #eef2ff, #ffffff);
    padding: 24px;
    border-radius: 20px;
    border-left: 8px solid #4f46e5;
    box-shadow: 0px 8px 24px rgba(15, 23, 42, 0.08);
    margin-bottom: 18px;
}

.human-card {
    background: linear-gradient(135deg, #ecfdf5, #ffffff);
    padding: 24px;
    border-radius: 20px;
    border-left: 8px solid #16a34a;
    box-shadow: 0px 8px 24px rgba(15, 23, 42, 0.08);
    margin-bottom: 18px;
}

.high { border-left: 8px solid #16a34a; }
.medium { border-left: 8px solid #f59e0b; }
.low { border-left: 8px solid #dc2626; }

.badge {
    display: inline-block;
    padding: 7px 12px;
    border-radius: 999px;
    background-color: #e0f2fe;
    color: #075985;
    font-size: 13px;
    font-weight: 700;
    margin-right: 6px;
    margin-bottom: 6px;
}

.badge-green { background-color: #dcfce7; color: #166534; }
.badge-yellow { background-color: #fef3c7; color: #92400e; }
.badge-red { background-color: #fee2e2; color: #991b1b; }
.badge-purple { background-color: #ede9fe; color: #5b21b6; }

.section-title {
    font-size: 26px;
    font-weight: 800;
    margin-top: 10px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD MODEL + DATA
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "Prospect ID",
        "Lead Source",
        "Do Not Email",
        "Do Not Call",
        "TotalVisits",
        "Total
