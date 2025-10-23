import streamlit as st
from textblob import TextBlob
import pandas as pd

# Título principal
st.title("🏆 Análisis de Sentimientos Deportivos")

st.write("""
Esta aplicación analiza si los comentarios sobre **deportes** son positivos, negativos o neutros.  
Puedes escribir opiniones sobre jugadores, equipos o partidos recientes.
""")

# Entrada del usuario
user_input = st.text_area("✍️ Escribe aquí tu opinión deportiva:")

# Análisis de sentimiento
if st.button("Analizar"):
    if user_input.strip() != "":
        blob = TextBlob(use




