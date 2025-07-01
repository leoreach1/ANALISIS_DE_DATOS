import streamlit as st
import pandas as pd
import pickle
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np



pdk.settings.mapbox_api_key = "TU_TOKEN_MAPBOX"

st.set_page_config(page_title="Predicción Multiclase COVID-19", layout="wide")
st.title("🧠 Predicción Multiclase del Riesgo por COVID-19 en EE.UU.")

# Coordenadas aproximadas de cada estado
state_coords = {
    'Alabama': [32.806671, -86.791130],
    'Alaska': [61.370716, -152.404419],
    'Arizona': [33.729759, -111.431221],
    'Arkansas': [34.969704, -92.373123],
    'California': [36.116203, -119.681564],
    'Colorado': [39.059811, -105.311104],
    'Connecticut': [41.597782, -72.755371],
    'Delaware': [39.318523, -75.507141],
    'Florida': [27.766279, -81.686783],
    'Georgia': [33.040619, -83.643074],
    'Hawaii': [21.094318, -157.498337],
    'Idaho': [44.240459, -114.478828],
    'Illinois': [40.349457, -88.986137],
    'Indiana': [39.849426, -86.258278],
    'Iowa': [42.011539, -93.210526],
    'Kansas': [38.526600, -96.726486],
    'Kentucky': [37.668140, -84.670067],
    'Louisiana': [31.169546, -91.867805],
    'Maine': [44.693947, -69.381927],
    'Maryland': [39.063946, -76.802101],
    'Massachusetts': [42.230171, -71.530106],
    'Michigan': [43.326618, -84.536095],
    'Minnesota': [45.694454, -93.900192],
    'Mississippi': [32.741646, -89.678696],
    'Missouri': [38.456085, -92.288368],
    'Montana': [46.921925, -110.454353],
    'Nebraska': [41.125370, -98.268082],
    'Nevada': [38.313515, -117.055374],
    'New Hampshire': [43.452492, -71.563896],
    'New Jersey': [40.298904, -74.521011],
    'New Mexico': [34.840515, -106.248482],
    'New York': [42.165726, -74.948051],
    'North Carolina': [35.630066, -79.806419],
    'North Dakota': [47.528912, -99.784012],
    'Ohio': [40.388783, -82.764915],
    'Oklahoma': [35.565342, -96.928917],
    'Oregon': [44.572021, -122.070938],
    'Pennsylvania': [40.590752, -77.209755],
    'Rhode Island': [41.680893, -71.511780],
    'South Carolina': [33.856892, -80.945007],
    'South Dakota': [44.299782, -99.438828],
    'Tennessee': [35.747845, -86.692345],
    'Texas': [31.054487, -97.563461],
    'Utah': [40.150032, -111.862434],
    'Vermont': [44.045876, -72.710686],
    'Virginia': [37.769337, -78.169968],
    'Washington': [47.400902, -121.490494],
    'West Virginia': [38.491226, -80.954570],
    'Wisconsin': [44.268543, -89.616508],
    'Wyoming': [42.755966, -107.302490]
}

def clasificar_riesgo(tasa):
    """Clasifica el riesgo basado en la tasa de letalidad"""
    if tasa <= 0.02:
        return 0  # Bajo
    elif tasa <= 0.10:
        return 1  # Medio
    else:
        return 2  # Extremo

def get_risk_color(risk_level):
    """Retorna el color basado en el nivel de riesgo"""
    colors = {0: '#00FF00', 1: '#FFA500', 2: '#FF0000'}
    return colors.get(risk_level, '#808080')

def load_and_process_data():
    """Carga y procesa los datos de COVID-19"""
    try:
        url = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv"
        df = pd.read_csv(url)
        df['date'] = pd.to_datetime(df['date'])
        
        # Usar datos más recientes
        ultima_fecha = df['date'].max()
        df = df[df['date'] == ultima_fecha].copy()
        
        # Limpiar datos
        df = df.dropna(subset=['cases', 'deaths', 'state', 'county'])
        df = df[df['cases'] > 0]
        df['fatality_rate'] = df['deaths'] / df['cases']
        df['risk_level'] = df['fatality_rate'].apply(clasificar_riesgo)
        
        return df, ultima_fecha
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return None, None

# -----------------------------------------------------
st.header("🔍 Predicción de Riesgo")

# Cargar datos para mostrar gráficos siempre
df_covid, fecha = load_and_process_data()

if df_covid is not None:
    # Mostrar gráficos de análisis de riesgo
    st.markdown("---")
    st.header("📊 Análisis de Riesgo por COVID-19")
    st.info(f"📆 Datos actualizados al: {fecha.date()}")
    
    # Crear columnas para métricas
    col1, col2, col3, col4 = st.columns(4)
    
    total_cases = df_covid['cases'].sum()
    total_deaths = df_covid['deaths'].sum()
    avg_fatality = df_covid['fatality_rate'].mean()
    total_counties = len(df_covid)
    
    with col1:
        st.metric("Total de Casos", f"{total_cases:,}")
    with col2:
        st.metric("Total de Muertes", f"{total_deaths:,}")
    with col3:
        st.metric("Tasa de Letalidad Promedio", f"{avg_fatality:.2%}")
    with col4:
        st.metric("Condados Analizados", f"{total_counties:,}")
    
    # Crear dos columnas para gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de distribución de riesgo
        risk_counts = df_covid['risk_level'].value_counts().sort_index()
        risk_labels = {0: 'Bajo', 1: 'Medio', 2: 'Extremo'}
        risk_colors = ['#00FF00', '#FFA500', '#FF0000']
        
        fig_risk = px.pie(
            values=risk_counts.values,
            names=[risk_labels[i] for i in risk_counts.index],
            title="📊 Distribución de Niveles de Riesgo",
            color_discrete_sequence=risk_colors
        )
        fig_risk.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with col2:
        # Gráfico de casos por nivel de riesgo
        df_grouped_risk = df_covid.groupby('risk_level').agg({
            'cases': 'sum',
            'deaths': 'sum'
        }).reset_index()
        df_grouped_risk['risk_label'] = df_grouped_risk['risk_level'].map(risk_labels)
        
        fig_cases_risk = px.bar(
            df_grouped_risk,
            x='risk_label',
            y='cases',
            title="📈 Casos Totales por Nivel de Riesgo",
            color='risk_level',
            color_discrete_sequence=risk_colors
        )
        fig_cases_risk.update_layout(showlegend=False)
        st.plotly_chart(fig_cases_risk, use_container_width=True)
    
    # Gráfico de casos por estado (top 15)
    st.markdown("### 🏛️ Estados con Mayor Número de Casos")
    df_by_state = df_covid.groupby('state').agg({
        'cases': 'sum',
        'deaths': 'sum',
        'fatality_rate': 'mean',
        'risk_level': 'mean'
    }).reset_index().sort_values('cases', ascending=False).head(15)
    
    fig_states = px.bar(
        df_by_state,
        x='cases',
        y='state',
        orientation='h',
        title="Top 15 Estados por Número de Casos",
        color='cases',
        color_continuous_scale='Reds'
    )
    fig_states.update_layout(height=600)
    st.plotly_chart(fig_states, use_container_width=True)
    
    # Mapa mejorado de Estados Unidos
    st.markdown("### 🗺️ Mapa de Riesgo por Estado")
    
    # Preparar datos para el mapa
    df_map = df_covid.groupby("state").agg({
        "cases": "sum",
        "deaths": "sum",
        "fatality_rate": "mean",
        "risk_level": "mean"
    }).reset_index()
    # Formato porcentaje para tasa de letalidad
    df_map["fatality_rate_str"] = (df_map["fatality_rate"] * 100).round(2).astype(str) + "%"

    # Texto para nivel de riesgo
    risk_labels = {0: "Bajo", 1: "Medio", 2: "Extremo"}
    df_map["risk_label_str"] = df_map["risk_level"].round().astype(int).map(risk_labels)
    df_map["risk_display"] = df_map["risk_level"].round(1).astype(str) + " (" + df_map["risk_label_str"] + ")"

    
    # Agregar coordenadas
    df_map["latitude"] = df_map["state"].apply(lambda x: state_coords.get(x, [None, None])[0])
    df_map["longitude"] = df_map["state"].apply(lambda x: state_coords.get(x, [None, None])[1])
    df_map = df_map.dropna(subset=["latitude", "longitude"])
    
    # Normalizar el tamaño de los círculos
    df_map['size'] = np.log(df_map['cases'] + 1) * 5000
    
    # Crear colores basados en el nivel de riesgo
    risk_color_map = {
        0: [0, 255, 0, 160],      # Verde (Bajo)
        1: [255, 165, 0, 160],    # Naranja (Medio)
        2: [255, 0, 0, 160]       # Rojo (Extremo)
    }
    
    df_map["color"] = df_map["risk_level"].apply(lambda x: risk_color_map[int(round(x))])
    
    # Crear el mapa con PyDeck
    view_state = pdk.ViewState(
        latitude=39.8283,
        longitude=-98.5795,
        zoom=3.5,
        pitch=0
    )
    
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_map,
        get_position='[longitude, latitude]',
        get_fill_color="color",
        get_radius="size",
        pickable=True,
        auto_highlight=True
    )
    
    tooltip = {
    "html": """
        <b>Estado:</b> {state}<br/>
        <b>Casos:</b> {cases}<br/>
        <b>Muertes:</b> {deaths}<br/>
        <b>Tasa de Letalidad:</b> {fatality_rate_str}<br/>
        <b>Nivel de Riesgo:</b> {risk_display}
    """,
    "style": {
        "backgroundColor": "steelblue",
        "color": "white"
        }
    }


    
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="light"
    )
    
    st.pydeck_chart(deck)
    
    # Leyenda del mapa
    st.markdown("""
    **Leyenda del Mapa:**
    - 🟢 **Verde**: Riesgo Bajo (≤ 2% letalidad)
    - 🟠 **Naranja**: Riesgo Medio (2-10% letalidad)  
    - 🔴 **Rojo**: Riesgo Extremo (> 10% letalidad)
    - **Tamaño del círculo**: Proporcional al número de casos
    """)

# Sección de predicción
try:
    with open("modelo_rf_multiclase.pkl", "rb") as f:
        model = pickle.load(f)

    st.success("✅ Modelo cargado exitosamente.")

    # Interfaz de predicción
    st.markdown("---")
    st.header("🎯 Realizar Predicción")
    
    col1, col2 = st.columns(2)
    
    with col1:
        estados = list(state_coords.keys())
        cases = st.number_input("Número de casos", min_value=1, value=100)
        deaths = st.number_input("Número de muertes", min_value=0, value=2)
        
    with col2:
        selected_state = st.selectbox("Seleccione el estado", estados)
        fatality_rate = deaths / cases if cases > 0 else 0
        st.metric("Tasa de letalidad calculada", f"{fatality_rate:.2%}")

    # Codificación del estado
    state_encoder = LabelEncoder()
    state_encoder.fit(estados)
    state_encoded = state_encoder.transform([selected_state])[0]

    if st.button("🔮 Predecir Riesgo", type="primary"):
        pred = model.predict([[cases, deaths, fatality_rate, state_encoded]])[0]
        pred_proba = model.predict_proba([[cases, deaths, fatality_rate, state_encoded]])[0]
        
        labels = {0: "Bajo", 1: "Medio", 2: "Extremo"}
        colores = {0: "✅", 1: "⚠️", 2: "🚨"}
        
        # Mostrar resultado
        st.markdown("### 📋 Resultado de la Predicción")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Estado", selected_state)
        with col2:
            st.metric("Tasa de letalidad", f"{fatality_rate:.2%}")
        with col3:
            st.success(f"{colores[pred]} **Riesgo estimado: {labels[pred]}**")
        
        # Mostrar probabilidades
        st.markdown("### 📊 Probabilidades por Nivel de Riesgo")
        prob_df = pd.DataFrame({
            'Nivel de Riesgo': ['Bajo', 'Medio', 'Extremo'],
            'Probabilidad': pred_proba,
            'Color': ['#00FF00', '#FFA500', '#FF0000']
        })
        
        fig_prob = px.bar(
            prob_df,
            x='Nivel de Riesgo',
            y='Probabilidad',
            color='Color',
            color_discrete_map={'#00FF00': '#00FF00', '#FFA500': '#FFA500', '#FF0000': '#FF0000'},
            title="Probabilidad de cada Nivel de Riesgo"
        )
        fig_prob.update_layout(showlegend=False)
        fig_prob.update_yaxes(tickformat='.1%')

        st.plotly_chart(fig_prob, use_container_width=True)

except FileNotFoundError:
    st.warning("🔁 El modelo aún no ha sido entrenado. Usa el botón de abajo para entrenarlo.")
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")

# -----------------------------------------------------
st.markdown("---")
st.header("⚙️ Entrenamiento del Modelo")

if st.button("🚀 Entrenar Modelo Ahora", type="primary"):
    with st.spinner("Cargando datos y entrenando el modelo..."):
        
        if df_covid is None:
            df_covid, fecha = load_and_process_data()
        
        if df_covid is not None:
            st.info(f"📆 Usando datos del: {fecha.date()}")
            
            # Preparar datos para entrenamiento
            le = LabelEncoder()
            df_covid['state_encoded'] = le.fit_transform(df_covid['state'])
            
            X = df_covid[['cases', 'deaths', 'fatality_rate', 'state_encoded']]
            y = df_covid['risk_level']
            
            # Validación del conjunto de datos
            st.markdown("### 🔍 Validación del conjunto de datos")
            
            if len(X) == 0 or len(y.unique()) < 2:
                st.error("❌ No hay datos suficientes para entrenar el modelo.")
                st.stop()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total de registros", f"{len(X):,}")
            with col2:
                st.metric("Clases detectadas", len(y.unique()))
            with col3:
                st.metric("Características", len(X.columns))
            
            # Entrenar modelo
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, stratify=y, test_size=0.2, random_state=42
            )
            
            model = RandomForestClassifier(
                random_state=42, 
                n_estimators=150, 
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            )
            
            model.fit(X_train, y_train)
            
            # Guardar modelo
            with open("modelo_rf_multiclase.pkl", "wb") as f:
                pickle.dump(model, f)
            
            # Evaluación
            y_pred = model.predict(X_test)
            accuracy = model.score(X_test, y_test)
            
            st.success(f"✅ Modelo entrenado exitosamente con {accuracy:.2%} de precisión")
            
            # Mostrar métricas
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📈 Reporte de Clasificación")
                st.text(classification_report(y_test, y_pred))
            
            with col2:
               labels_dict = {0: 'Bajo', 1: 'Medio', 2: 'Extremo'}

               unique_labels = sorted(np.unique(y_test))
               label_names = [labels_dict[i] for i in unique_labels]

                # Crear la matriz de confusión con etiquetas presentes
               cm = confusion_matrix(y_test, y_pred, labels=unique_labels)

                # Crear el gráfico con solo las etiquetas existentes
               fig_cm = px.imshow(
                    cm,
                    labels=dict(x="Predicción", y="Real", color="Cantidad"),
                    x=label_names,
                    y=label_names,
                    title="Matriz de Confusión"
                )
               fig_cm.update_xaxes(side="bottom")
               st.plotly_chart(fig_cm, use_container_width=True)
            
            # Importancia de características
            feature_importance = pd.DataFrame({
                'caracteristica': X.columns,
                'importancia': model.feature_importances_
            }).sort_values('importancia', ascending=False)
            
            fig_importance = px.bar(
                feature_importance,
                x='importancia',
                y='caracteristica',
                orientation='h',
                title="Importancia de las Características"
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        
        else:
            st.error("❌ No se pudieron cargar los datos para entrenamiento.")

# Información adicional
st.markdown("---")
st.markdown("""
### 📋 Información del Sistema

**Niveles de Riesgo:**
- **Bajo (Verde)**: Tasa de letalidad ≤ 2%
- **Medio (Naranja)**: Tasa de letalidad 2-10%
- **Extremo (Rojo)**: Tasa de letalidad > 10%

**Fuente de datos:** New York Times COVID-19 Data
**Modelo:** Random Forest Classifier
**Características:** Casos, Muertes, Tasa de Letalidad, Estado
""")