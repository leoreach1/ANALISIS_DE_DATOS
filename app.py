import streamlit as st
import pickle
import streamlit.components.v1 as components

st.set_page_config(page_title="COVID-19 USA", layout="wide")

# Mostrar el HTML como portada
with open("index.html", "r", encoding="utf-8") as f:
    html_string = f.read()

components.html(html_string, height=1600, scrolling=True)

# Separador visual
st.markdown("---")

# Cargar el modelo
try:
    with open("modelo_rf.pkl", "rb") as f:
        model = pickle.load(f)

    st.markdown("## 🧠 Predicción de Riesgo por Letalidad COVID-19")
    st.write("Ingresa los valores estimados para hacer una predicción:")

    # Formulario simple
    cases = st.number_input("Número de casos", min_value=1)
    deaths = st.number_input("Número de muertes", min_value=0)

    if st.button("Predecir"):
        fatality_rate = deaths / cases
        prediction = model.predict([[cases, deaths]])[0]

        st.metric("Tasa de Letalidad", f"{fatality_rate:.2%}")
        if prediction == 1:
            st.error("⚠️ Riesgo estimado: **ALTO**")
        else:
            st.success("✅ Riesgo estimado: **BAJO**")

except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
