import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import pandas as pd

# ===========================================
# KONFIGURASI HALAMAN
# ===========================================

st.set_page_config(
    page_title="Deteksi Cacat Radiografi",
    page_icon="🩻",
    layout="wide"
)

# ===========================================
# LOAD MODEL
# ===========================================

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ===========================================
# JUDUL
# ===========================================

st.title("🩻 Deteksi Cacat Citra Radiografi Pengelasan")

st.write("""
Aplikasi ini menggunakan **YOLOv8 Segmentation**
untuk mendeteksi cacat pada citra radiografi hasil pengelasan.
""")

st.divider()

# ===========================================
# UPLOAD GAMBAR
# ===========================================

uploaded_file = st.file_uploader(
    "Upload Citra Radiografi",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Gambar Asli")
        st.image(image, use_container_width=True)

    # Simpan gambar sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        image_path = tmp.name

    # Prediksi
    with st.spinner("Sedang melakukan deteksi..."):

        results = model.predict(
            source=image_path,
            conf=0.25,
            save=False
        )

    result = results[0]

    with col2:
        st.subheader("Hasil Deteksi")
        plotted = result.plot()
        st.image(plotted, use_container_width=True)

    st.divider()

    # ===========================================
    # TIDAK ADA CACAT
    # ===========================================

    if len(result.boxes) == 0:

        st.success("🟢 HASIL INSPEKSI")

        st.markdown("## Tidak Ditemukan Cacat")

        st.write(
            "Model tidak menemukan area yang terindikasi sebagai cacat."
        )

    # ===========================================
    # ADA CACAT
    # ===========================================

    else:

        st.error("🔴 HASIL INSPEKSI")

        jumlah = len(result.boxes)

        st.markdown(f"## Cacat Terdeteksi ({jumlah} area)")

        data = []

        for box in result.boxes:

            kelas = int(box.cls[0])

            nama = result.names[kelas]

            conf = float(box.conf[0]) * 100

            data.append({
                "Jenis Cacat": nama,
                "Confidence (%)": round(conf,2)
            })

        df = pd.DataFrame(data)

        st.subheader("Detail Deteksi")

        st.dataframe(df, use_container_width=True)

        st.subheader("Ringkasan")

        jenis = ", ".join(df["Jenis Cacat"].unique())

        confidence = df["Confidence (%)"].max()

        st.info(f"""
Jumlah cacat yang terdeteksi : **{jumlah}**

Jenis cacat :

**{jenis}**

Confidence tertinggi :

**{confidence:.2f}%**
""")
