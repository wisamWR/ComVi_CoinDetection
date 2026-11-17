"""Streamlit demo: upload a photo or use the camera, get detected denominations and the total."""
import cv2
import numpy as np
import streamlit as st

from money_detection import CURRENCIES, MoneyDetector, annotate

st.set_page_config(page_title="ComVi Money Detection", page_icon="💵", layout="wide")


@st.cache_resource
def load_detector(code):
    return MoneyDetector(code, device="cpu")


st.title("ComVi Money Detection")
st.caption("Deteksi uang kertas Rupiah dan koin US Dollar dengan YOLO26, lalu hitung totalnya.")

with st.sidebar:
    code = st.radio("Mata uang", list(CURRENCIES), format_func=lambda c: CURRENCIES[c].name)
    conf = st.slider("Confidence minimum", 0.1, 0.9, 0.5, 0.05)
    source = st.radio("Input", ["Upload foto", "Kamera"])

currency = CURRENCIES[code]
image_file = (st.file_uploader("Foto uang", type=["jpg", "jpeg", "png", "bmp", "webp"])
              if source == "Upload foto" else st.camera_input("Ambil foto"))

if image_file is None:
    st.info("Pilih mata uang di sidebar, lalu unggah foto atau ambil gambar dari kamera.")
    st.stop()

frame = cv2.imdecode(np.frombuffer(image_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
if frame is None:
    st.error("Gambar tidak bisa dibaca.")
    st.stop()

with st.spinner("Mendeteksi..."):
    dets = load_detector(code).detect(frame, conf=conf)

left, right = st.columns([3, 2])
left.image(annotate(frame, dets, currency)[:, :, ::-1])

classes = [d.cls for d in dets]
right.metric("Total", currency.format(currency.total(classes)), f"{len(dets)} item terdeteksi", delta_color="off")
if dets:
    right.table([{"Pecahan": label, "Jumlah": n, "Subtotal": currency.format(sub)}
                 for label, n, sub in currency.breakdown(classes)])
else:
    right.warning("Tidak ada uang terdeteksi. Coba turunkan confidence atau foto lebih dekat.")
