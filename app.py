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
        blob = TextBlob(user_input)
        sentiment = blob.sentiment.polarity

        if sentiment > 0:
            st.success("⚽ Resultado: ¡Sentimiento Positivo! 🎉")
        elif sentiment < 0:
            st.error("🥀 Resultado: Sentimiento Negativo 😞")
        else:
            st.info("😐 Resultado: Sentimiento Neutro")

        st.write("**Puntuación:**", sentiment)
    else:
        st.warning("Por favor escribe algo para analizar.")

# Ejemplos de uso
st.markdown("---")
st.subheader("Ejemplos de comentarios deportivos:")
examples = {
    "Comentario": [
        "El partido de anoche fue increíble, Messi jugó como un genio.",
        "Odio cuando el árbitro arruina el juego.",
        "La defensa del equipo estuvo bien, pero faltó precisión en el ataque."
    ]
}
df = pd.DataFrame(examples)
st.table(df)
