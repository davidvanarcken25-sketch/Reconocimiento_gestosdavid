import streamlit as st
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
import platform

# -------- CONFIGURACIÓN DE PÁGINA --------
st.set_page_config(
    page_title="🪐 Control de Nave Espacial",
    page_icon="🚀",
    layout="centered",
)

# -------- ESTILO VISUAL --------
st.markdown("""
    <style>
    .stApp {
        background-color: #030c1a;
        color: #00e5ff;
        font-family: 'Orbitron', sans-serif;
    }
    h1, h2, h3 {
        text-align: center;
        color: #00ffff;
    }
    .stButton>button {
        background-color: #00e5ff;
        color: black;
        border-radius: 12px;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #ff007f;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# -------- CARGA DEL MODELO --------
st.title("🛰️ Sistema de Reconocimiento Gestual Espacial")
st.caption("Controla tu nave intergaláctica con inteligencia artificial")
st.write("Versión de Python:", platform.python_version())

model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# -------- IMAGEN PRINCIPAL --------
st.image("nave.jpg", width=350)

with st.sidebar:
    st.subheader("👽 Panel de Comando Galáctico")
    st.markdown("Usa tus gestos para dirigir la nave:")
    st.markdown("- ✋ **Izquierda:** Giro a babor")
    st.markdown("- ✋ **Arriba:** Ascenso orbital")
    st.markdown("---")
    st.markdown("🧠 Modelo IA: *Teachable Machine (Keras)*")

# -------- CAPTURA DE CÁMARA --------
img_file_buffer = st.camera_input("📸 Toma una foto para identificar tu gesto")

# -------- PROCESAMIENTO Y RESULTADO --------
if img_file_buffer is not None:
    img = Image.open(img_file_buffer).resize((224, 224))
    img_array = np.array(img)
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)

    # -------- RESULTADOS --------
    if prediction[0][0] > 0.5:
        st.subheader("🪐 Maniobra detectada: **Giro a la izquierda**")
        st.image("ees.jpg", width=200)
        st.success(f"Probabilidad: {prediction[0][0]:.2f}")
        st.markdown("**Comando ejecutado:** La nave realiza un viraje estelar a babor.")
    elif prediction[0][1] > 0.5:
        st.subheader("🚀 Maniobra detectada: **Ascenso orbital**")
        st.image("nave.jpg", width=200)
        st.success(f"Probabilidad: {prediction[0][1]:.2f}")
        st.markdown("**Comando ejecutado:** Motores de impulso encendidos. Iniciando ascenso.")
    else:
        st.info("🛰️ Ningún gesto reconocido. Sistema en modo de observación galáctica.")
    
else:
    st.info("🛰️ Esperando señal de control... Toma una foto para continuar.")
    st.image("cet.jpg", width=200)




