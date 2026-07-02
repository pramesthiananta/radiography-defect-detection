import streamlit as st
from ultralytics import YOLO

st.set_page_config(
    page_title="Deteksi Cacat Radiografi",
    page_icon="🩻",
    layout="wide"
)

st.title("🩻 Deteksi Cacat Citra Radiografi Pengelasan")

st.write("Model berhasil dimuat.")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.success("✅ Model YOLO berhasil dimuat.")
