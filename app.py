import streamlit as st
import pickle
import streamlit.components.v1 as components

# Incrustar HTML completo desde archivo
with open("index.html", "r", encoding="utf-8") as f:
    html_string = f.read()

components.html(html_string, height=600, scrolling=True)

# Cargar modelo entrenado
with open('modelo_rf.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Predicción de Riesgo COVID-19 por Condado")

st.write("Ingrese los datos del condado para predecir el nivel de riesgo basado en la tasa de letalidad.")

cases = st.number_input("Número de casos", min_value=1)
deaths = st.number_input("Número de muertes", min_value=0)

if st.button("Predecir"):
    fatality_rate = deaths / cases
    prediction = model.predict([[cases, deaths]])[0]
    st.write("Tasa de Letalidad estimada:", round(fatality_rate, 4))
    st.success("Riesgo: Alto" if prediction == 1 else "Riesgo: Bajo")
