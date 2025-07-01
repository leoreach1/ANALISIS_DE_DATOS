import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="Predicción Multiclase COVID-19", layout="wide")
st.title("🧠 Análisis Epidemiológico y Predicción de Riesgo por COVID-19")

@st.cache_data
def cargar_datos():
    url = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv"
    df = pd.read_csv(url)
    df['date'] = pd.to_datetime(df['date'])
    return df

df = cargar_datos()
ultima_fecha = df['date'].max()
df = df[df['date'] == ultima_fecha].copy()
df = df.dropna(subset=['cases', 'deaths', 'state', 'county'])
df = df[df['cases'] > 0]
df['fatality_rate'] = df['deaths'] / df['cases']

def clasificar_riesgo(tasa):
    if tasa <= 0.02:
        return 0
    elif tasa <= 0.10:
        return 1
    else:
        return 2

df['risk_level'] = df['fatality_rate'].apply(clasificar_riesgo)
le = LabelEncoder()
df['state_encoded'] = le.fit_transform(df['state'])

st.markdown(f"📅 Fecha de análisis: **{ultima_fecha.date()}**")
st.markdown(f"🔢 Registros disponibles: **{len(df)}**")

st.subheader("📊 Casos y muertes por estado")
top_states = df.groupby('state')[['cases', 'deaths']].sum().sort_values(by='cases', ascending=False).head(10)
fig1, ax1 = plt.subplots()
top_states.plot(kind='bar', ax=ax1)
ax1.set_title("Top 10 Estados por Casos y Muertes")
st.pyplot(fig1)

st.subheader("🩺 Tasa de Letalidad por Estado (Top 10)")
letalidad_estado = df.groupby('state').apply(lambda x: x['deaths'].sum() / x['cases'].sum()).sort_values(ascending=False).head(10)
fig2, ax2 = plt.subplots()
letalidad_estado.plot(kind='bar', ax=ax2, color='tomato')
ax2.set_ylabel("Tasa de letalidad")
ax2.set_title("Top 10 Estados por Tasa de Letalidad")
st.pyplot(fig2)

st.subheader("🧭 Distribución del Nivel de Riesgo")
fig3, ax3 = plt.subplots()
sns.countplot(x='risk_level', data=df, palette='Set2', ax=ax3)
ax3.set_xticklabels(['Bajo', 'Medio', 'Extremo'])
ax3.set_title("Frecuencia de Niveles de Riesgo")
st.pyplot(fig3)

st.markdown("---")
st.header("⚙️ Entrenamiento del Modelo desde la App")

X = df[['cases', 'deaths', 'fatality_rate', 'state_encoded']]
y = df['risk_level']

if len(X) == 0 or len(y.unique()) < 2:
    st.error("❌ No hay suficientes datos o clases para entrenar el modelo.")
else:
    if st.button("📚 Entrenar modelo multiclase"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(random_state=42, n_estimators=150, max_depth=10)
        model.fit(X_train, y_train)
        with open("modelo_rf_multiclase.pkl", "wb") as f:
            pickle.dump(model, f)
        y_pred = model.predict(X_test)
        st.success("✅ Modelo entrenado y guardado como 'modelo_rf_multiclase.pkl'")
        st.markdown("### 📈 Reporte de Clasificación")
        st.text(classification_report(y_test, y_pred))
        st.markdown("### 🧩 Matriz de Confusión")
        st.text(confusion_matrix(y_test, y_pred))

st.markdown("---")
st.header("🔍 Predicción de Riesgo")
try:
    with open("modelo_rf_multiclase.pkl", "rb") as f:
        model = pickle.load(f)
    cases = st.number_input("Número de casos", min_value=1)
    deaths = st.number_input("Número de muertes", min_value=0)
    fatality_rate = deaths / cases
    state_encoded = st.number_input("Código del estado (state_encoded)", min_value=0)
    if st.button("Predecir Riesgo"):
        pred = model.predict([[cases, deaths, fatality_rate, state_encoded]])[0]
        labels = {0: "Bajo", 1: "Medio", 2: "Extremo"}
        colores = {0: "✅", 1: "⚠️", 2: "🚨"}
        st.metric("Tasa de letalidad", f"{fatality_rate:.2%}")
        st.success(f"{colores[pred]} Riesgo estimado: {labels[pred]}")
except Exception as e:
    st.warning("🔁 El modelo aún no ha sido entrenado.")
