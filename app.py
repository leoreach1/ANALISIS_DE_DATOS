import streamlit as st
import pickle
import streamlit.compobnents.v1 as components

# Leer archivo HTML
with open("index.html", "r", encoding="utf-8") as f:
    html_string = f.read()

<<<<<<< HEAD
# Leer archivo CSS
with open("assets/css/templatemo-chain-app-dev.css", "r", encoding="utf-8") as f:
    css_string = f.read()

# Inyectar CSS dentro del HTML
# Busca el <head> y lo reemplaza para insertar el <style> justo después
if "<head>" in html_string:
    html_string = html_string.replace(
        "<head>",
        f"<head>\n<style>\n{css_string}\n</style>\n"
    )
else:
    # Si no hay <head>, lo inserta al principio del body como emergencia
    html_string = html_string.replace(
        "<body>",
        f"<body>\n<style>\n{css_string}\n</style>\n"
    )

# Mostrar HTML con estilos aplicados
components.html(html_string, height=600, scrolling=True)

# Cargar modelo entrenado
with open('modelo_rf.pkl', 'rb') as f:
    model = pickle.load(f)

# Interfaz Streamlit
st.title("Predicción de Riesgo COVID-19 por Condado")

st.write("Ingrese los datos del condado para predecir el nivel de riesgo basado en la tasa de letalidad.")

cases = st.number_input("Número de casos", min_value=1)
deaths = st.number_input("Número de muertes", min_value=0)

if st.button("Predecir"):
    fatality_rate = deaths / cases
    prediction = model.predict([[cases, deaths]])[0]
    st.write("Tasa de Letalidad estimada:", round(fatality_rate, 4))
    st.success("Riesgo: Alto" if prediction == 1 else "Riesgo: Bajo")
=======

>>>>>>> a20d309ee0edcd63602099183a503edca7c125b0
