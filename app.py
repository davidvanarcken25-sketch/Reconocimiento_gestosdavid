import streamlit as st
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
import platform

# ===========================
# CONFIGURACIÓN DE LA PÁGINA
# ===========================
st.set_page_config(
    page_title="Mood Detector 3000 – Laboratorio de Gestos",
    page_icon="🎭",
    layout="centered"
)

# ===========================
# CARGA DEL MODELO
# ===========================
st.sidebar.title("⚙️ Panel de Control del Laboratorio")
st.sidebar.info("""
Bienvenido al **Mood Detector 3000**, un sistema experimental del  
*Laboratorio de Expresiones Humanas* encargado de analizar gestos faciales.

Sube tu modelo entrenado en **Teachable Machine (.h5)** y experimenta con tus gestos frente a la cámara.
""")

model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# ===========================
# INTERFAZ PRINCIPAL
# ===========================
st.title("🎭 Mood Detector 3000")
st.caption("Laboratorio de Expresiones Humanas | Proyecto EmotiCore")

st.image("OIG5.jpg", width=280, caption="Cámara biométrica de análisis emocional")

st.markdown("---")

# Mostrar versión del entorno
st.text(f"Versión del sistema: Python {platform.python_version()} • Núcleo EmotiCore v2.1")

# ===========================
# CAPTURA DE IMAGEN
# ===========================
st.markdown("### 📸 Escaneo facial")
st.write("Activa tu cámara y realiza un gesto. El sistema intentará identificar el estado emocional dominante.")

img_file_buffer = st.camera_input("Cámara de detección")

if img_file_buffer is not None:
    with st.spinner("🧠 Analizando microexpresiones..."):
        # Leer imagen y preparar datos
        img = Image.open(img_file_buffer)
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized)

        normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array

        # Predicción
        prediction = model.predict(data)

    st.success("✅ Escaneo completado")

    # ===========================
    # INTERPRETACIÓN DE RESULTADOS
    # ===========================
    st.markdown("### 🔬 Resultado del análisis de gesto")
    if prediction[0][0] > 0.5:
        st.subheader("😎 Modo Energético: Confianza Absoluta")
        st.write(f"**Intensidad del gesto:** {prediction[0][0]:.2%}")
        st.caption("Interpretación: Seguridad, poder y dominio de la situación.")
    elif prediction[0][1] > 0.5:
        st.subheader("🤔 Modo Energético: Curiosidad Activa")
        st.write(f"**Intensidad del gesto:** {prediction[0][1]:.2%}")
        st.caption("Interpretación: Interés, análisis y pensamiento crítico.")
    else:
        st.subheader("😐 Modo Energético: Neutro o no reconocido")
        st.caption("El sistema no pudo detectar un gesto dominante. Intenta otro movimiento facial más claro.")

    st.markdown("---")
    st.info("Vuelve a tomar una foto con otro gesto para probar diferentes resultados.")

# ===========================
# PIE DE PÁGINA
# ===========================
st.markdown("---")
st.caption("""
**Mood Detector 3000**  
Proyecto del Laboratorio de Expresiones Humanas.  
Sistema de detección de gestos faciales desarrollado con Streamlit y Keras.  
Versión experimental 1.2
""")


