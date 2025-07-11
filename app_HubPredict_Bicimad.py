import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pydeck as pdk
from datetime import datetime, timedelta, date
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

import folium
from streamlit_folium import st_folium
import geopy.distance
import streamlit.components.v1 as components

# --- Importaciones de Modelos ---
import joblib
from category_encoders import TargetEncoder
import lightgbm as lgb
import requests
import base64 # Importación necesaria para _get_base64_image

# --- Constantes de ubicación por defecto (Madrid) ---
LATITUDE_MADRID = 40.4168
LONGITUDE_MADRID = -3.7038

# --- Power BI Embed Link ---
power_bi_embed_code = """
<iframe title="ProyectoBiciMad" width="100%" height="600" src="https://app.powerbi.com/reportEmbed?reportId=af0e6ac1-610d-477d-b86c-bde5f520e273&autoAuth=true&ctid=ae32eec6-145a-4167-8a63-02cea1138fe3&navContentPaneEnabled=false" frameborder="0" allowFullScreen="true"></iframe>
"""

# Configuración de página
st.set_page_config(
    page_title="HubPredict BiciMAD",
    layout="wide",
    page_icon="🚴‍♂️",
    initial_sidebar_state="expanded"
)

# CSS personalizado mejorado
st.markdown("""
<style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
        }

        .prediction-container {
            background: #f8f9fa;
            padding: 0.5rem 1.5rem; /* Reducido drásticamente el padding vertical */
            border-radius: 10px;
            border-left: 4px solid #667eea;
            margin: 0.2rem 0; /* Reducido el margen externo */
            /* Añadir overflow para asegurar que el contenido no desborde si se hace muy pequeño */
            overflow: hidden;
        }

        /* Estilos para el título h4 dentro de .prediction-container */
        .prediction-container h4 {
            margin-top: 0.3rem !important; /* Más agresivo con !important */
            margin-bottom: 0.3rem !important; /* Más agresivo con !important */
            font-size: 1.3rem; /* Ligeramente más pequeño */
            line-height: 1.2; /* Compacta la altura de línea */
        }

        /* Estilos para los párrafos p dentro de .prediction-container */
        .prediction-container p {
            margin-bottom: 0.1rem !important; /* Márgenes muy pequeños */
            margin-top: 0.1rem !important;
            font-size: 0.9rem; /* Opcional: un poco más pequeña la fuente del texto */
            line-height: 1.3; /* Compacta la altura de línea */
        }

        /* Intento más directo a los contenedores generados por Streamlit para el markdown */
        /* Puede que los divs directamente encima de nuestro HTML inyectado tengan padding */
        .stMarkdown, .stMarkdown > div, .stMarkdown > div > div {
            padding-top: 0 !important;
            padding-bottom: 0 !important;
            margin-top: 0 !important;
            margin-bottom: 0 !important;
        }

        .desenganche { border-left: 4px solid #dc3545; }
        .enganche { border-left: 4px solid #28a745; }
        .ambas { border-left: 4px solid #ffc107; }

        .station-search {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }

        .nearby-stations {
            background: #e7f3ff;
            padding: 0.8rem;
            border-radius: 6px;
            margin: 0.5rem 0;
            font-size: 0.9em;
        }

        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
            border-left: 4px solid #667eea;
        }

        .prediction-table {
            background: white;
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 1rem 0;
        }

        .demand-high { color: #dc3545; font-weight: bold; }
        .demand-medium { color: #ffc107; font-weight: bold; }
        .demand-low { color: #28a745; font-weight: bold; }

        .tab-content {
            padding: 1rem 0;
        }

        /* Nuevo estilo para el recuadro de Power BI */
        .powerbi-container {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
            border: 1px solid #e0e0e0; /* Un borde sutil para el recuadro */
        }

        /* ESTILOS AÑADIDOS PARA EL ICONO DEL CLIMA */
        .weather-icon {
            vertical-align: middle;
            margin-right: 5px;
            height: 24px; /* Ajusta el tamaño de los iconos si es necesario */
            width: 24px;
        }
    </style>
    """, unsafe_allow_html=True)

# Header principal
st.markdown("""
<div class="main-header">
    <h1>🚴‍♂️ HubPredict - BiciMAD </h1>
    <p>Sistema inteligente para predicción de enganches y desenganches en estaciones de Madrid</p>
</div>
""", unsafe_allow_html=True)

# --- Cargar modelos y encoders con caché ---
@st.cache_resource
def load_ml_assets():
    """Carga los modelos de ML y encoders."""
    try:
        # Asegúrate de que estos archivos .pkl existan en la carpeta 'modelo/'
        encoder_plug = joblib.load('modelo/encoder_plug.pkl')
        model_plug = joblib.load('modelo/model_plug_lgbm.pkl')
        encoder_unplug = joblib.load('modelo/encoder_unplug.pkl')
        model_unplug = joblib.load('modelo/model_unplug_lgbm.pkl')
        st.success("Modelos y encoders cargados exitosamente.")
        return encoder_plug, model_plug, encoder_unplug, model_unplug
    except FileNotFoundError as e:
        st.error(f"Error al cargar archivos del modelo. Asegúrate de que los archivos .pkl estén en la carpeta 'modelo/'. Detalle: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error inesperado al cargar los activos de ML: {e}")
        st.stop()

encoder_plug, model_plug, encoder_unplug, model_unplug = load_ml_assets()

# Simulación de datos de estaciones con coordenadas
@st.cache_data
def load_stations_data():
    """Cargar datos de estaciones desde estaciones_bicimad.csv"""
    try:
        df_stations = pd.read_csv("data/estaciones_bicimad.csv")
        
        stations_data = {}
        for index, row in df_stations.iterrows():
            stations_data[row['name']] = {
                'lat': row['latitude'],
                'lon': row['longitude'],
                'id': row['id'],
                'max_capacity': row.get('max_capacity', 24) # Asume 24 si no está la columna
            }
        return stations_data
    except FileNotFoundError:
        st.error("Error: El archivo 'estaciones_bicimad.csv' no se encontró en la carpeta 'data/'.")
        st.stop()
    except KeyError as e:
        st.error(f"Error: Una columna esperada no se encontró en 'estaciones_bicimad.csv'. Falta la columna: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Ocurrió un error al cargar los datos de las estaciones: {e}")
        st.stop()

# --- Función mejorada para obtener pronóstico meteorológico para varias horas ---
@st.cache_data(ttl=3600) # Caché por 1 hora
def get_future_weather_forecast_for_day(latitude, longitude, target_date):
    """Obtiene el pronóstico meteorológico por horas para una fecha específica."""
    date_str = target_date.strftime('%Y-%m-%d')
    url = (f"https://api.open-meteo.com/v1/forecast?"
           f"latitude={latitude}&longitude={longitude}&"
           f"hourly=temperature_2m,precipitation&"
           f"start_date={date_str}&end_date={date_str}&"
           f"timezone=Europe%2FBerlin") # Zona horaria de Madrid

    try:
        response = requests.get(url, timeout=10) # Añadir timeout
        response.raise_for_status() # Lanza excepción para códigos de error HTTP
        data = response.json()
        hourly_data = data.get('hourly', {})

        if not hourly_data or not hourly_data.get('time'):
            st.warning(f"No se encontraron datos horarios para la fecha {date_str}.")
            return pd.DataFrame()

        df = pd.DataFrame({
            'time': pd.to_datetime(hourly_data.get('time', [])),
            'temperature_2m': hourly_data.get('temperature_2m', []),
            'precipitation': hourly_data.get('precipitation', [])
        })
        df['hour'] = df['time'].dt.hour
        return df[['hour', 'temperature_2m', 'precipitation']]
    except requests.exceptions.Timeout:
        st.error("La solicitud a la API del clima ha tardado demasiado. Intenta de nuevo más tarde.")
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        st.error(f"Error al obtener datos de pronóstico del tiempo: {e}")
        return pd.DataFrame()

# --- Nueva función de predicción REAL con modelos ---
@st.cache_data # Caché de predicciones basado en inputs
def predict_demand_real(station_id, target_date, target_hour, prediction_type,
                        _encoder_plug_model, _model_plug_model,
                        _encoder_unplug_model, _model_unplug_model,
                        weather_data_df):
    """
    Realiza la predicción de demanda de enganche o desenganche para una estación
    y hora específicas, utilizando los modelos entrenados y datos meteorológicos.
    Retorna la predicción, temperatura y precipitación usadas.
    """
    
    # 1. Preparar datos de entrada base
    is_weekend = 1 if target_date.weekday() in [5, 6] else 0 # 5=Sábado, 6=Domingo

    # Obtener clima para la hora específica o la más cercana
    temp = 20.0 # Valor por defecto si no se encuentra nada
    prec = 0.0  # Valor por defecto si no se encuentra nada
    
    if not weather_data_df.empty:
        weather_row = weather_forecast_df[weather_forecast_df['hour'] == target_hour]
        
        if not weather_row.empty:
            temp = weather_row['temperature_2m'].iloc[0]
            prec = weather_row['precipitation'].iloc[0]
        else:
            # Si no se encuentra la hora exacta, buscar la más cercana
            weather_data_df['hour_diff'] = abs(weather_data_df['hour'] - target_hour)
            closest_weather_row = weather_data_df.loc[weather_data_df['hour_diff'].idxmin()]
            
            temp = closest_weather_row['temperature_2m']
            prec = closest_weather_row['precipitation']
            # st.warning(f"No se encontró pronóstico para la hora {target_hour:02d}:00. Usando el clima de la hora más cercana ({closest_weather_row['hour']:02d}:00).")
    else:
        st.warning("No se pudieron obtener datos del pronóstico del tiempo. Usando valores predeterminados para la predicción.")

    data_point = pd.DataFrame([{
        'station_id': station_id,
        'dia': target_date.day,
        'mes': target_date.month,
        'año': target_date.year,
        'dia_semana': target_date.weekday(), # 0=Lunes, 6=Domingo
        'hour': target_hour,
        'Valor_Temp': temp,
        'Valor_Prec': prec,
        'dia_especial': is_weekend # Asumiendo que dia_especial es 1 para fin de semana
    }])

    # 2. Definir columnas categóricas para el Target Encoding (6 columnas)
    categorical_cols_for_encoder = ['station_id', 'dia', 'mes', 'hour', 'año', 'dia_semana']

    # 3. Seleccionar encoder y modelo según el tipo de predicción
    if prediction_type == "Desenganche":
        encoder = _encoder_unplug_model
        model = _model_unplug_model
    elif prediction_type == "Enganche":
        encoder = _encoder_plug_model
        model = _model_plug_model
    else:
        raise ValueError("Tipo de predicción no válido. Debe ser 'Desenganche' o 'Enganche'.")

    # 4. Aplicar Target Encoding
    df_to_encode = data_point[categorical_cols_for_encoder].copy()
    encoded_features = encoder.transform(df_to_encode)
    
    df_encoded = pd.DataFrame(encoded_features, columns=categorical_cols_for_encoder, index=data_point.index)

    # 5. Combinar features codificadas con las no codificadas
    final_features_for_model = [
        'station_id', 'dia', 'mes', 'hour', 'año', 'dia_semana',
        'Valor_Temp', 'Valor_Prec', 'dia_especial'
    ]
    
    final_input_df = df_encoded.copy()
    final_input_df['Valor_Temp'] = data_point['Valor_Temp']
    final_input_df['Valor_Prec'] = data_point['Valor_Prec']
    final_input_df['dia_especial'] = data_point['dia_especial']

    final_input_df = final_input_df[final_features_for_model]

    # 6. Realizar la predicción
    predicted_count_raw = model.predict(final_input_df)
    predicted_count_rounded = np.round(predicted_count_raw[0])
    
    return max(0, int(predicted_count_rounded)), temp, prec # Retorna también temp y prec

def get_demand_level(demand):
    """Clasificar nivel de demanda (para el color, no para texto en gráfico)"""
    if demand >= 30:
        return "Alta", "demand-high"
    elif demand >= 15:
        return "Media", "demand-medium"
    else:
        return "Baja", "demand-low"

@st.cache_data # Cachear las imágenes para no cargarlas cada vez
def _get_base64_image(image_path):
    """Carga una imagen y la codifica en base64 para incrustarla en HTML."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo de imagen en {image_path}. Asegúrate de que la carpeta 'imagenes/' exista y contenga los iconos (ej. 'clima_generico.png').")
        return "" # Devuelve una cadena vacía o un icono de "no encontrado"

def get_weather_icon_and_value(precipitation, temperature):
    """Devuelve el HTML del icono de clima genérico junto a los valores numéricos."""
    icon_path = "imagenes/" # Asegúrate de que esta carpeta exista y contenga los iconos
    
    # Icono genérico de clima (puedes usar un sol, una nube, etc.)
    # Aquí usaré un sol como genérico, pero puedes cambiarlo por una nube si prefieres
    weather_icon_html = f"<img src='data:image/png;base64,{_get_base64_image(f'{icon_path}clima_generico.png')}' class='weather-icon' title='Clima'>"

    return f"<div>{weather_icon_html} {temperature:.1f}°C / {precipitation:.1f}mm</div>"


def find_nearby_stations(user_location, stations_data, radius_km=2):
    """Encontrar estaciones cercanas a la ubicación del usuario"""
    nearby = []
    for name, data in stations_data.items():
        distance = geopy.distance.distance(
            user_location,
            (data['lat'], data['lon'])
        ).kilometers

        if distance <= radius_km:
            nearby.append({
                'name': name,
                'distance': round(distance, 2),
                'lat': data['lat'],
                'lon': data['lon']
            })

    return sorted(nearby, key=lambda x: x['distance'])

def create_stations_map(stations_data, selected_stations=None):
    """Crear mapa con estaciones y ajustar el zoom para ver todas las estaciones."""

    lats = []
    lons = []
    for name, data in stations_data.items():
        lats.append(data['lat'])
        lons.append(data['lon'])

    if lats and lons:
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2
    else:
        center_lat, center_lon = 40.4168, -3.7038
        min_lat, max_lat, min_lon, max_lon = center_lat - 0.01, center_lat + 0.01, center_lon - 0.01, center_lon + 0.01


    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

    selected_stations_list = selected_stations if selected_stations is not None else []

    for name, data in stations_data.items():
        color = 'red' if name in selected_stations_list else 'blue'
        icon = 'star' if name in selected_stations_list else 'bicycle'

        folium.Marker(
            location=[data['lat'], data['lon']],
            popup=f"<b>{name}</b><br>ID: {data['id']}<br>Lat: {data['lat']}<br>Lon: {data['lon']}<br>Capacidad Max: {data['max_capacity']}",
            tooltip=name,
            icon=folium.Icon(color=color, icon=icon, prefix='fa', icon_size=(25, 25))
        ).add_to(m)

    if lats and lons:
        m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

    return m

# Cargar datos
stations_data = load_stations_data()
station_names = list(stations_data.keys())

# Layout principal
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 📋 Parámetros de Predicción")

    # Selector de tipo de predicción
    st.markdown("#### 🎯 Tipo de Predicción")
    prediction_type = st.radio(
        "Selecciona qué quieres predecir:",
        ["Desenganche", "Enganche", "Ambas"],
        help="• **Desenganche**: Bicicletas que salen de la estación\n• **Enganche**: Bicicletas que llegan a la estación\n• **Ambas**: Predicción completa"
    )

    # Selector de estación mejorado
    st.markdown("#### 🚉 Selección de Estación")

    search_option = st.selectbox(
        "Método de selección:",
        ["Lista desplegable", "Búsqueda por nombre", "Estaciones cercanas"]
    )

    selected_station_for_desenganche = None
    selected_station_for_enganche = None
    selected_station_single = None

    if prediction_type == "Ambas":
        st.markdown("#### Selecciona Estaciones Específicas para 'Ambas'")

        selected_station_for_desenganche = st.selectbox(
            "Estación para **Desenganche**:",
            station_names,
            key="desenganche_station_selector",
            help="Selecciona la estación para la cual predecir desenganches."
        )

        selected_station_for_enganche = st.selectbox(
            "Estación para **Enganche**:",
            station_names,
            key="enganche_station_selector",
            index=min(1, len(station_names) - 1),
            help="Selecciona la estación para la cual predecir enganches."
        )

        selected_stations_for_map = []
        if selected_station_for_desenganche:
            selected_stations_for_map.append(selected_station_for_desenganche)
        if selected_station_for_enganche and selected_station_for_enganche not in selected_stations_for_map:
            selected_stations_for_map.append(selected_station_for_enganche)

    else: # Si el tipo de predicción es "Desenganche" o "Enganche" individual
        if search_option == "Lista desplegable":
            selected_station_single = st.selectbox(
                "Estación:",
                station_names,
                key="single_station_list",
                help="Selecciona una estación de la lista."
            )

        elif search_option == "Búsqueda por nombre":
            search_text = st.text_input(
                "Buscar estación:",
                placeholder="Escribe el nombre de la estación...",
                key="single_station_search_text"
            )

            if search_text:
                filtered_stations = [s for s in station_names if search_text.lower() in s.lower()]

                if filtered_stations:
                    selected_station_single = st.selectbox(
                        "Estaciones encontradas:",
                        filtered_stations,
                        key="single_station_found"
                    )
                else:
                    st.warning("No se encontraron estaciones con ese nombre")

        elif search_option == "Estaciones cercanas":
            st.info("📍 Solicitando tu ubicación actual... Por favor, permite el acceso en tu navegador.")

            _component_func = components.declare_component(
                "geolocation_component",
                path="./components", # Asegúrate de que el archivo geolocation_component.py esté aquí
            )

            location_data = _component_func(key="user_geolocation")

            user_location = None

            if location_data:
                if "latitude" in location_data and "longitude" in location_data:
                    lat = location_data["latitude"]
                    lon = location_data["longitude"]
                    user_location = (lat, lon)
                    st.success(f"Ubicación obtenida: Latitud {lat:.4f}, Longitud {lon:.4f}")
                    st.write(f"Buscando estaciones cercanas a: ({lat:.4f}, {lon:.4f})")
                elif "error" in location_data:
                    st.error(f"Error al obtener la ubicación: {location_data['error']}")
                    st.warning("Usando ubicación predeterminada (centro de Madrid).")
                    user_location = (LATITUDE_MADRID, LONGITUDE_MADRID)
            else:
                st.warning("Esperando la ubicación... Asegúrate de permitir el acceso en tu navegador.")
                user_location = (LATITUDE_MADRID, LONGITUDE_MADRID)

            if user_location:
                nearby_stations = find_nearby_stations(user_location, stations_data, radius_km=1.5)

                if nearby_stations:
                    st.markdown("**Estaciones cercanas (dentro de 1.5 km):**")
                    station_options = [f"{s['name']} ({s['distance']} km)" for s in nearby_stations]
                    selected_option = st.selectbox("Estaciones cercanas:", station_options, key="single_station_nearby")
                    if selected_option:
                        selected_station_single = selected_option.split(" (")[0]
                else:
                    st.warning("No hay estaciones cercanas en el radio especificado con tu ubicación actual o la predeterminada.")
            else:
                st.warning("No se pudo determinar la ubicación del usuario para buscar estaciones cercanas.")
                selected_station_single = None


        selected_stations_for_map = [selected_station_single] if selected_station_single else []


    # Condición para mostrar fecha, hora, rango y botón
    can_predict = False
    if prediction_type == "Ambas":
        if selected_station_for_desenganche and selected_station_for_enganche:
            can_predict = True
    else: # Desenganche o Enganche individual
        if selected_station_single:
            can_predict = True

    predict_button = False
    if can_predict:
        st.markdown("#### 📅 Fecha y Hora")

        max_date = date.today() + timedelta(days=15)
        selected_date = st.date_input(
            "Fecha:",
            value=date.today() + timedelta(days=1),
            min_value=date.today(),
            max_value=max_date,
            key="prediction_date"
        )

        selected_hour = st.selectbox(
            "Hora:",
            range(24),
            format_func=lambda x: f"{x:02d}:00",
            index=9,
            key="prediction_hour"
        )

        # Selector de rango de horas
        st.markdown("#### ⏰ Rango de Predicción")
        hour_range = st.selectbox(
            "Número de horas a mostrar:",
            [3, 5, 7],
            index=1,
            help="Muestra predicciones para N horas centradas en la hora seleccionada",
            key="hour_range_selector"
        )
        
        # Botón de predicción
        predict_button = st.button(
            "🔮 Generar Predicción",
            type="primary",
            use_container_width=True
        )
        
        # --- Mover este bloque aquí, debajo del botón de predicción ---
        st.markdown("#### 🚲 Capacidad de la Estación")
        if prediction_type == "Ambas":
            if selected_station_for_desenganche:
                cap_des = stations_data[selected_station_for_desenganche]['max_capacity']
                st.info(f"**Desenganche:** {selected_station_for_desenganche} - Capacidad Máx: **{cap_des}** bicis")
            if selected_station_for_enganche:
                cap_eng = stations_data[selected_station_for_enganche]['max_capacity']
                st.info(f"**Enganche:** {selected_station_for_enganche} - Capacidad Máx: **{cap_eng}** bicis")
        else:
            if selected_station_single:
                cap_single = stations_data[selected_station_single]['max_capacity']
                st.info(f"Capacidad Máx. de {selected_station_single}: **{cap_single}** bicis")
        
        st.caption("*(Este es el número de bases que tiene la estación, no la cantidad de bicicletas disponibles en tiempo real.)*")
        # --- Fin del bloque a mover ---

    else: # Si no se puede predecir
        if prediction_type == "Ambas":
            st.warning("Selecciona una estación para desenganche y otra para enganche para generar la predicción.")
        else:
            st.warning("Selecciona una estación para continuar con la predicción.")


with col2:
    st.markdown("### 📊 Resultados y Visualización")

    # El mapa siempre se muestra
    st.markdown("### 🗺️ Ubicación de Estaciones")
    if selected_stations_for_map:
        stations_map = create_stations_map(stations_data, selected_stations=selected_stations_for_map)
        st_folium(stations_map, width=700, height=400)
    else:
        st.info("Selecciona una estación en la columna izquierda para verla en el mapa.")
        # Si no hay selección, mostrar un mapa general
        stations_map_general = create_stations_map(stations_data)
        st_folium(stations_map_general, width=700, height=400)


    # Resultados de predicción (solo se muestra si el botón de predicción fue presionado)
    if predict_button:
        with st.spinner("⏳ Predicción en curso... Esto puede tomar unos segundos mientras los modelos analizan los datos."):
            # Calcular rango de horas
            half_range = hour_range // 2
            hours_to_predict = []

            for i in range(-half_range, half_range + 1):
                hour = (selected_hour + i) % 24
                hours_to_predict.append(hour)
            
            # --- Obtener datos meteorológicos para TODAS las horas del día ---
            if prediction_type == "Ambas":
                weather_lat = stations_data.get(selected_station_for_desenganche, {'lat': LATITUDE_MADRID})['lat']
                weather_lon = stations_data.get(selected_station_for_desenganche, {'lon': LONGITUDE_MADRID})['lon']
            else:
                weather_lat = stations_data.get(selected_station_single, {'lat': LATITUDE_MADRID})['lat']
                weather_lon = stations_data.get(selected_station_single, {'lon': LONGITUDE_MADRID})['lon']

            weather_forecast_df = get_future_weather_forecast_for_day(weather_lat, weather_lon, selected_date)

            # Información de la consulta unificada y mejorada
            st.markdown(f"""
            <div class="prediction-container">
                <h4>🎯 Consulta Realizada ({prediction_type})</h4>
            """, unsafe_allow_html=True)

            station_info_text = "<strong>Estación(es):</strong> "
            if prediction_type == 'Ambas':
                if selected_station_for_desenganche:
                    station_info_text += f"Desenganche: {selected_station_for_desenganche}"
                if selected_station_for_enganche:
                    if selected_station_for_desenganche:
                        station_info_text += " - "
                    station_info_text += f"Enganche: {selected_station_for_enganche}"
            else:
                if selected_station_single:
                    station_info_text += selected_station_single

            st.markdown(f"<p>{station_info_text}</p>", unsafe_allow_html=True)

            st.markdown(f"""
                <p>
                    <strong>Fecha:</strong> {selected_date.strftime('%d/%m/%Y')} ({selected_date.strftime('%A')})
                    &nbsp;&nbsp;-&nbsp;&nbsp;
                    <strong>Hora objetivo:</strong> {selected_hour:02d}:00
                </p>
            </div>
            """, unsafe_allow_html=True)

            if prediction_type == "Ambas":
                tab_desenganche, tab_enganche = st.tabs(["🔴 Desenganche", "🟢 Enganche"])

                with tab_desenganche:
                    if selected_station_for_desenganche:
                        st.markdown(f"#### Predicciones de Desenganche para: {selected_station_for_desenganche}")
                        predictions_des = []
                        station_id_des = stations_data[selected_station_for_desenganche]['id']

                        for hour in hours_to_predict:
                            # predict_demand_real ahora retorna también temperatura y precipitación
                            demand, temp, prec = predict_demand_real(
                                station_id_des, selected_date, hour, "Desenganche",
                                encoder_plug, model_plug, encoder_unplug, model_unplug,
                                weather_forecast_df
                            )
                            
                            predictions_des.append({
                                'Hora': f"{hour:02d}:00",
                                'Predicción Desenganches': demand,
                                'Clima': get_weather_icon_and_value(prec, temp), # Usamos la nueva función
                                'Tipo': '🎯 Objetivo' if hour == selected_hour else '📍 Referencia'
                            })

                        df_des = pd.DataFrame(predictions_des)
                        
                        # Columnas a mostrar en la tabla
                        display_cols_des = ['Hora', 'Predicción Desenganches', 'Clima', 'Tipo']
                        st.markdown(df_des[display_cols_des].to_html(escape=False, index=False), unsafe_allow_html=True)


                        # --- Gráfico (sin cambios significativos aquí en la lógica de datos, solo el título puede cambiar) ---
                        fig_des = go.Figure()
                        color_target_des = '#0047AB' # Azul más oscuro para objetivo
                        color_reference_des = '#ADD8E6' # Azul claro para referencia

                        # Predicción
                        fig_des.add_trace(go.Bar(
                            x=df_des['Hora'],
                            y=df_des['Predicción Desenganches'],
                            marker_color=[color_target_des if t == '🎯 Objetivo' else color_reference_des for t in df_des['Tipo']],
                            name='Predicción',
                            text=df_des['Predicción Desenganches'],
                            textposition='auto',
                            textfont=dict(size=14)
                        ))
                        
                        fig_des.update_layout(
                            title=f"Desenganches - {selected_station_for_desenganche}",
                            xaxis_title="Hora",
                            yaxis_title="Cantidad de Desenganches",
                            height=400
                        )
                        st.plotly_chart(fig_des, use_container_width=True)
                    else:
                        st.info("No se seleccionó estación para Desenganche.")

                with tab_enganche:
                    if selected_station_for_enganche:
                        st.markdown(f"#### Predicciones de Enganche para: {selected_station_for_enganche}")
                        predictions_eng = []
                        station_id_eng = stations_data[selected_station_for_enganche]['id']

                        for hour in hours_to_predict:
                            # predict_demand_real ahora retorna también temperatura y precipitación
                            demand, temp, prec = predict_demand_real(
                                station_id_eng, selected_date, hour, "Enganche",
                                encoder_plug, model_plug, encoder_unplug, model_unplug,
                                weather_forecast_df
                            )
                            
                            predictions_eng.append({
                                'Hora': f"{hour:02d}:00",
                                'Predicción Enganches': demand,
                                'Clima': get_weather_icon_and_value(prec, temp), # Usamos la nueva función
                                'Tipo': '🎯 Objetivo' if hour == selected_hour else '📍 Referencia'
                            })

                        df_eng = pd.DataFrame(predictions_eng)

                        # Columnas a mostrar en la tabla
                        display_cols_eng = ['Hora', 'Predicción Enganches', 'Clima', 'Tipo']
                        st.markdown(df_eng[display_cols_eng].to_html(escape=False, index=False), unsafe_allow_html=True)

                        # --- Gráfico (sin cambios significativos aquí) ---
                        fig_eng = go.Figure()
                        color_target_eng = '#0047AB' # Azul más oscuro para objetivo
                        color_reference_eng = '#87CEEB' # Azul cielo para referencia (o un verde si prefieres)

                        # Predicción
                        fig_eng.add_trace(go.Bar(
                            x=[p['Hora'] for p in predictions_eng],
                            y=[p['Predicción Enganches'] for p in predictions_eng],
                            marker_color=[color_target_eng if p['Tipo'] == '🎯 Objetivo' else color_reference_eng for p in predictions_eng],
                            name='Predicción',
                            text=[f"{p['Predicción Enganches']}" for p in predictions_eng],
                            textposition='auto',
                            textfont=dict(size=14)
                        ))

                        fig_eng.update_layout(
                            title=f"Enganches - {selected_station_for_enganche}",
                            xaxis_title="Hora",
                            yaxis_title="Cantidad de Enganches",
                            height=400
                        )
                        st.plotly_chart(fig_eng, use_container_width=True)
                    else:
                        st.info("No se seleccionó estación para Enganche.")
            else: # Si el tipo de predicción es "Desenganche" o "Enganche" individual
                if selected_station_single:
                    st.markdown(f"#### Predicciones para: {selected_station_single}")
                    predictions_single = []
                    station_id_single = stations_data[selected_station_single]['id']

                    for hour in hours_to_predict:
                        # predict_demand_real ahora retorna también temperatura y precipitación
                        demand, temp, prec = predict_demand_real(
                            station_id_single, selected_date, hour, prediction_type,
                            encoder_plug, model_plug, encoder_unplug, model_unplug,
                            weather_forecast_df
                        )
                        
                        predictions_single.append({
                            'Hora': f"{hour:02d}:00",
                            f'Predicción {prediction_type}': demand,
                            'Clima': get_weather_icon_and_value(prec, temp), # Usamos la nueva función
                            'Tipo': '🎯 Objetivo' if hour == selected_hour else '📍 Referencia'
                        })

                    df_single = pd.DataFrame(predictions_single)
                    
                    # Columnas a mostrar en la tabla
                    display_cols_single = ['Hora', f'Predicción {prediction_type}', 'Clima', 'Tipo']
                    st.markdown(df_single[display_cols_single].to_html(escape=False, index=False), unsafe_allow_html=True)


                    # --- Gráfico (sin cambios significativos aquí) ---
                    fig_single = go.Figure()
                    # Colores
                    color_target_single = '#0047AB'
                    color_reference_single = '#ADD8E6' if prediction_type == "Desenganche" else '#87CEEB'

                    # Predicción
                    fig_single.add_trace(go.Bar(
                        x=df_single['Hora'],
                        y=df_single[f'Predicción {prediction_type}'],
                        marker_color=[color_target_single if t == '🎯 Objetivo' else color_reference_single for t in df_single['Tipo']],
                        name='Predicción',
                        text=df_single[f'Predicción {prediction_type}'],
                        textposition='auto',
                        textfont=dict(size=14)
                    ))

                    fig_single.update_layout(
                        title=f"{prediction_type} - {selected_station_single}",
                        xaxis_title="Hora",
                        yaxis_title=f"Cantidad de {prediction_type}s",
                        height=400
                    )
                    st.plotly_chart(fig_single, use_container_width=True)
                else:
                    st.info("No se seleccionó estación para la predicción.")

# --- Sección de Power BI (fuera del condicional predict_button para que siempre esté visible) ---
st.markdown("---")
st.markdown("### 📊 Dashboard Interactivo (Power BI)")
st.markdown("""
<div class="powerbi-container">
    <p>Explora métricas adicionales y tendencias históricas en nuestro dashboard de Power BI.</p>
</div>
""", unsafe_allow_html=True)
components.html(power_bi_embed_code, height=600)