import streamlit as st
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
from gtts import gTTS
import platform
import time
import io

# ===============================
# CONFIGURACIÓN DE LA PÁGINA
# ===============================
st.set_page_config(
    page_title="Centro de Control Gestual – Misión A.R.G.O.S.",
    page_icon="🚀",
    layout="centered"
)

# ===============================
# CARGA DEL MODELO
# ===============================
model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# ===============================
# INTERFAZ PRINCIPAL
# ===============================
st.title("🚀 Centro de Control Gestual A.R.G.O.S.")
st.caption("Sistema de reconocimiento de gestos para navegación espacial")

# Imagen decorativa (futurista o de la nave)
st.image("OIG5.jpg", width=320, caption="Unidad de análisis visual A.R.G.O.S. en línea")

st.markdown("---")
st.text(f"Versión del sistema: Python {platform.python_version()} | Núcleo Visual A.R.G.O.S. v1.5")

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    st.subheader("📡 Módulo de entrenamiento activo")
    st.info("Este sistema usa un modelo de **Teachable Machine** para interpretar gestos como comandos espaciales.")
    st.write("Captura un gesto frente a la cámara para ejecutar una orden.")

# ===============================
# CAPTURA DE IMAGEN
# ===============================
st.markdown("### 🎥 Escáner de Comando Gestual")
img_file_buffer = st.camera_input("Activa la cámara y realiza tu gesto")

# ===============================
# PROCESAMIENTO DE LA IMAGEN
# ===============================
if img_file_buffer is not None:
    # Cargar imagen
    img = Image.open(img_file_buffer)
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized)

    # Normalizar
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    # Ejecutar predicción
    with st.spinner("🔍 Analizando gesto..."):
        time.sleep(1)
        prediction = model.predict(data)
    
    st.success("✅ Comando recibido con éxito")

    # ===============================
    # INTERPRETACIÓN Y RESULTADOS
    # ===============================
    mensaje = ""
    if prediction[0][0] > 0.5:
        st.subheader("🖐️ Gesto detectado: **IZQUIERDA**")
        st.write(f"Probabilidad: {prediction[0][0]:.2%}")
        mensaje = "Comando recibido: activando propulsores laterales izquierdos."
        st.caption(mensaje)

    elif prediction[0][1] > 0.5:
        st.subheader("✋ Gesto detectado: **ARRIBA**")
        st.write(f"Probabilidad: {prediction[0][1]:.2%}")
        mensaje = "Comando recibido: elevando la nave a coordenadas superiores."
        st.caption(mensaje)

    else:
        st.subheader("🤷‍♂️ Gesto no reconocido")
        mensaje = "Comando no válido. Por favor, intenta nuevamente."
        st.caption(mensaje)

    # ===============================
    # GENERAR AUDIO AUTOMÁTICAMENTE
    # ===============================
    if mensaje:
        tts = gTTS(text=mensaje, lang='es')
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        st.audio(audio_bytes.getvalue(), format="audio/mp3")

# ===============================
# PIE DE PÁGINA
# ===============================
st.markdown("---")
st.caption("""
🛰️ **Centro de Control Gestual A.R.G.O.S.**  
Proyecto experimental de reconocimiento de gestos.  
Desarrollado con **Streamlit + Keras + gTTS + Teachable Machine**.  
""")


