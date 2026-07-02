import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import pandas as pd

# ============================
# Konfigurasi halaman
# ============================

st.set_page_config(
    page_title="Deteksi Cacat Radiografi",
    page_icon="🩻",
    layout="wide"
)

# ============================
# Load Model
# ============================

@st.cache_resource
def load_model():
    return YOLO("best.pt")      

model = load_model()

# ============================
# Judul
# ============================

st.title("🩻 Deteksi Cacat Citra Radiografi Pengelasan")

st.markdown("""
Aplikasi ini menggunakan **YOLOv8 Segmentation** untuk mendeteksi cacat
pada citra radiografi hasil pengelasan.
""")

st.divider()

# ============================
# Upload
# ============================

uploaded_file = st.file_uploader(
    "Upload citra radiografi (.jpg/.png)",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Gambar Asli")
        st.image(image, use_container_width=True)

    # ============================
    # Simpan sementara
    # ============================

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    image.save(temp.name)

    # ============================
    # Prediksi
    # ============================

    with st.spinner("Sedang mendeteksi..."):

        results = model.predict(
            source=temp.name,
            conf=0.25,
            save=False
        )

    result = results[0]

    with col2:
        st.subheader("Hasil Deteksi")

        plotted = result.plot()

        st.image(plotted, use_container_width=True)

    st.divider()

    # ============================
    # Analisis hasil
    # ============================

    boxes = result.boxes

    if len(boxes)==0:

        st.success("🟢 HASIL INSPEKSI")

        st.markdown("""
### Tidak Ditemukan Cacat

Model tidak menemukan area yang terindikasi sebagai cacat
pada citra radiografi.
""")

    else:

        st.error("🔴 HASIL INSPEKSI")

        st.markdown(f"### Cacat Terdeteksi ({len(boxes)} area)")

        data=[]

        for box in boxes:

            cls=int(box.cls)

            nama=result.names[cls]

            conf=float(box.conf)

            data.append({
                "Jenis Cacat":nama,
                "Confidence (%)":round(conf*100,2)
            })

        df=pd.DataFrame(data)

        st.subheader("Detail Deteksi")

        st.dataframe(df,use_container_width=True)

        st.subheader("Ringkasan")

        jenis=df["Jenis Cacat"].unique()

        confidence=df["Confidence (%)"].max()

        st.info(
            f"""
Ditemukan **{len(df)} area cacat**.

Jenis cacat yang terdeteksi:

{', '.join(jenis)}

Confidence tertinggi:

**{confidence:.2f}%**
"""
        )

os.unlink(temp.name)
