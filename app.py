import streamlit as st
import cv2
import numpy as np
from PIL import Image as Image
from keras.models import load_model
from gtts import gTTS
import io
import platform

# -----------------------------
# CONFIGURACIÓN INICIAL
# -----------------------------
st.title("Análisis de Gestos – Piloto de Nave Espacial")
st.subheader("Sistema de reconocimiento de movimientos para el control de vuelo")
st.write("Versión de Python:", platform.python_version())

# Imagen decorativa (puede ser un logo de nave o radar)
image = Image.open("OIG5.jpg")
st.image(image, width=300, caption="Sistema de reconocimiento de comandos gestuales")

# Cargar el modelo entrenado
model = load_model("keras_model.h5")
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

with st.sidebar:
    st.header("Panel del piloto")
    st.write("Toma una foto con tu cámara y deja que la nave interprete tu gesto.")
    st.write("El sistema puede reconocer gestos para moverse a la izquierda o elevarse.")

# -----------------------------
# CAPTURA DE IMAGEN
# -----------------------------
img_file_buffer = st.camera_input("Toma una foto para analizar el gesto")

# -----------------------------
# PROCESAMIENTO Y PREDICCIÓN
# -----------------------------
if img_file_buffer is not None:
    img = Image.open(img_file_buffer)
    newsize = (224, 224)
    img = img.resize(newsize)
    img_array = np.array(img)
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)

    # -----------------------------
    # FUNCIÓN PARA REPRODUCIR AUDIO
    # -----------------------------
    def reproducir_audio(texto):
        tts = gTTS(text=texto, lang='es')
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        st.audio(audio_bytes.getvalue(), format='audio/mp3')

    # -----------------------------
    # RESULTADOS Y RESPUESTAS
    # -----------------------------
    if prediction[0][0] > 0.5:
        mensaje = "Comando detectado: giro a la izquierda. Activando propulsores laterales."
        st.success(mensaje)
        reproducir_audio(mensaje)
        st.image("https://cdn-icons-png.flaticon.com/512/744/744465.png", width=120)

    elif prediction[0][1] > 0.5:
        mensaje = "Comando detectado: ascenso. Aumentando potencia de los motores principales."
        st.success(mensaje)
        reproducir_audio(mensaje)
        st.image("https://cdn-icons-png.flaticon.com/512/744/744484.png", width=120)

    else:
        mensaje = "Gesto no reconocido. Intenta nuevamente o ajusta la iluminación."
        st.warning(mensaje)
        reproducir_audio(mensaje)
        st.image("https://cdn-icons-png.flaticon.com/512/565/565547.png", width=120)


