import streamlit as st
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
import platform

# -------- CONFIGURACIÓN DE PÁGINA --------
st.set_page_config(
    page_title="🥋 Sensei IA - Entrenamiento de Artes Marciales",
    page_icon="🥷",
    layout="centered",
)

# -------- ESTILO VISUAL --------
st.markdown("""
    <style>
    .stApp {
        background-color: #0b0f19;
        color: #00ff9d;
        font-family: 'Rajdhani', sans-serif;
    }
    h1, h2, h3 {
        text-align: center;
        color: #00ffc8;
    }
    .stButton>button {
        background-color: #00ff9d;
        color: black;
        border-radius: 10px;
        font-weight: bold;
        border: none;
        padding: 0.5em 1em;
    }
    .stButton>button:hover {
        background-color: #ff3b3b;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# -------- CARGA DEL MODELO --------
st.title("🥋 Sensei IA — Entrenamiento de Artes Marciales")
st.caption("Aprende y mejora tus reflejos con inteligencia artificial.")
st.write("Versión de Python:", platform.python_version())

model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# -------- IMAGEN PRINCIPAL --------
st.image("cet.jpg", width=350, caption="Prepárate, estudiante...")

with st.sidebar:
    st.subheader("⚔️ Panel del Dojo")
    st.markdown("Imita los movimientos indicados:")
    st.markdown("- ✋ **Bloqueo:** Defensa rápida contra ataque enemigo.")
    st.markdown("- ✊ **Puñetazo:** Ataque frontal directo.")
    st.markdown("---")
    st.markdown("🧠 Modelo IA: *Teachable Machine (Keras)*")

# -------- CAPTURA DE CÁMARA --------
img_file_buffer = st.camera_input("📸 Captura tu movimiento marcial")

# -------- PROCESAMIENTO Y RESULTADO --------
if img_file_buffer is not None:
    img = Image.open(img_file_buffer).resize((224, 224))
    img_array = np.array(img)
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)

    # -------- RESULTADOS --------
    if prediction[0][0] > 0.5:
        st.subheader("🛡️ Movimiento detectado: **Bloqueo defensivo**")
        st.image("ees.jpg", width=200)
        st.success(f"Precisión: {prediction[0][0]:.2f}")
        st.markdown("**Resultado:** Defensa perfecta. Has desviado el ataque enemigo.")
    elif prediction[0][1] > 0.5:
        st.subheader("👊 Movimiento detectado: **Puñetazo directo**")
        st.image("nave.jpg", width=200)
        st.success(f"Precisión: {prediction[0][1]:.2f}")
        st.markdown("**Resultado:** ¡Impacto exitoso! Tu golpe ha sido poderoso.")
    else:
        st.info("🤔 Movimiento no reconocido. El Sensei te observa con paciencia...")
    
else:
    st.info("🥷 Esperando tu técnica... Captura un movimiento para continuar.")
    st.image("OIG5.jpg", width=200, caption="Mantén la calma y concéntrate.")

