import streamlit as st
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
import platform
from streamlit_lottie import st_lottie
import requests

# -------- CONFIGURACIÓN GENERAL --------
st.set_page_config(
    page_title="🪐 Control de Nave Espacial",
    page_icon="🚀",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp {
        background-color: #020817;
        color: #00FFFF;
        font-family: 'Orbitron', sans-serif;
    }
    h1, h2, h3, .stMarkdown {
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# -------- CARGA DEL MODELO --------
st.title("🛰️ Sistema de Reconocimiento Gestual Espacial")
st.caption("Controla tu nave intergaláctica con gestos detectados por IA")

st.write("Versión de Python:", platform.python_version())

model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# -------- IMAGEN PRINCIPAL --------
st.image("https://i.ibb.co/3NtH0qG/space-pilot.jpg", width=350)

with st.sidebar:
    st.subheader("👽 Panel de Comando Galáctico")
    st.markdown("Usa gestos para controlar el sistema:")
    st.markdown("- ✋ Izquierda → Giro a babor")
    st.markdown("- ✋ Arriba → Ascenso orbital")

# -------- FUNCIÓN PARA CARGAR LOTTIES --------
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Animaciones Lottie (puedes cambiarlas si deseas)
lottie_left = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_6Yq7m9.json")   # giro
lottie_up = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_q5pk6p1k.json")   # ascenso
lottie_idle = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_qp1q7mct.json") # espera

# -------- CÁMARA Y PREDICCIÓN --------
img_file_buffer = st.camera_input("📸 Toma una foto para identificar tu gesto")

if img_file_buffer is not None:
    img = Image.open(img_file_buffer).resize((224, 224))
    img_array = np.array(img)
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)

    # -------- RESULTADOS --------
    if prediction[0][0] > 0.5:
        st.subheader("🪐 Maniobra espacial detectada: **Giro a la izquierda**")
        st_lottie(lottie_left, height=300, key="left")
        st.success(f"Probabilidad: {prediction[0][0]:.2f}")
        st.markdown("**Comando ejecutado:** Viraje estelar activado. Nave girando a babor.")
    elif prediction[0][1] > 0.5:
        st.subheader("🚀 Ascenso orbital detectado: **Movimiento hacia arriba**")
        st_lottie(lottie_up, height=300, key="up")
        st.success(f"Probabilidad: {prediction[0][1]:.2f}")
        st.markdown("**Comando ejecutado:** Motores de impulso encendidos. Iniciando ascenso.")
    else:
        st_lottie(lottie_idle, height=250, key="idle")
        st.info("🛰️ Esperando gesto... Sistema en modo de observación galáctica.")
else:
    st_lottie(lottie_idle, height=250, key="idle_idle")
    st.info("🛰️ Esperando señal de control...")

# -------- PIE DE PÁGINA --------
st.markdown("---")
st.caption("Desarrollado por la Agencia Espacial IA — Propulsado con Streamlit y Keras.")


