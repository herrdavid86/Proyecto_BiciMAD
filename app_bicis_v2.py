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
import streamlit.components.v1 as components # Importación necesaria para componentes

# --- Power BI Embed Link ---
power_bi_embed_code = """
<iframe title="ProyectoBiciMad" width="100%" height="600" src="https://app.powerbi.com/reportEmbed?reportId=af0e6ac1-610d-477d-b86c-bde5f520e273&autoAuth=true&ctid=ae32eec6-145a-4167-8a63-02cea1138fe3" frameborder="0" allowFullScreen="true"></iframe>
"""

# Configuración de página
st.set_page_config(
    page_title="BiciMAD Predictor",
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
    </style>
    """, unsafe_allow_html=True)

# Header principal
st.markdown("""
<div class="main-header">
    <h1>🚴‍♂️ BiciMAD - Predictor Avanzado de Demanda</h1>
    <p>Sistema inteligente para predicción de enganches y desenganches en estaciones de Madrid</p>
</div>
""", unsafe_allow_html=True)

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
                'id': row['id']
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

@st.cache_data
def predict_demand(station, target_date, target_hour, prediction_type):
    """Predicción mejorada con tipos de operación"""
    base_demand = np.random.randint(10, 40)

    weekday = target_date.weekday()
    weekday_factor = 1.2 if weekday < 5 else 0.8

    if prediction_type == "Desenganche":
        if 7 <= target_hour <= 9:
            hour_factor = 1.8
        elif 12 <= target_hour <= 14:
            hour_factor = 1.3
        elif 17 <= target_hour <= 19:
            hour_factor = 1.4
        else:
            hour_factor = 0.6
    elif prediction_type == "Enganche":
        if 8 <= target_hour <= 10:
            hour_factor = 1.6
        elif 13 <= target_hour <= 15:
            hour_factor = 1.2
        elif 18 <= target_hour <= 20:
            hour_factor = 1.7
        else:
            hour_factor = 0.7
    else:
        if 7 <= target_hour <= 20:
            hour_factor = 1.0
        else:
            hour_factor = 0.5

    month = target_date.month
    if month in [6, 7, 8]:
        season_factor = 1.3
    elif month in [12, 1, 2]:
        season_factor = 0.7
    else:
        season_factor = 1.0

    demand = int(base_demand * weekday_factor * hour_factor * season_factor)
    demand += np.random.randint(-3, 4)

    return max(0, demand)

def get_demand_level(demand):
    """Clasificar nivel de demanda"""
    if demand >= 30:
        return "Alta", "demand-high"
    elif demand >= 15:
        return "Media", "demand-medium"
    else:
        return "Baja", "demand-low"

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
            popup=f"<b>{name}</b><br>ID: {data['id']}<br>Lat: {data['lat']}<br>Lon: {data['lon']}",
            tooltip=name,
            icon=folium.Icon(color=color, icon=icon, prefix='fa', icon_size=(25, 25)) # Icono de 25x25
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

            # Define la ruta al componente JavaScript
            _component_func = components.declare_component(
                "geolocation_component",
                path="./components", # Ruta a la carpeta que contiene geolocation.js
            )

            # Llama al componente y guarda el resultado
            # Esto ejecutará el JavaScript y pedirá la ubicación al navegador
            location_data = _component_func(key="user_geolocation")

            user_location = None # Inicializa user_location

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
                    user_location = (40.4168, -3.7038) # Ubicación predeterminada si falla el GPS
            else:
                # Esto ocurre la primera vez que se carga la página antes de que el componente devuelva algo
                st.warning("Esperando la ubicación... Asegúrate de permitir el acceso en tu navegador.")
                user_location = (40.4168, -3.7038) # Usar predeterminada mientras se espera o si el usuario no interactúa

            # Ahora, usa user_location para encontrar estaciones cercanas
            # Asegúrate de que user_location no sea None antes de llamar a find_nearby_stations
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
                # Si user_location es None aquí, significa que el componente no devolvió nada
                # y no hubo un error explícito, lo cual es raro pero posible si el navegador no se carga correctamente.
                # En este caso, no hay estaciones cercanas que mostrar.
                selected_station_single = None # Asegurarse de que no haya una estación seleccionada


        selected_stations_for_map = [selected_station_single] if selected_station_single else []


    # Condición para mostrar fecha, hora, rango y botón
    can_predict = False
    if prediction_type == "Ambas":
        if selected_station_for_desenganche and selected_station_for_enganche:
            can_predict = True
    else: # Desenganche o Enganche individual
        if selected_station_single:
            can_predict = True

    predict_button = False # Inicializa predict_button fuera del if
    if can_predict:
        st.markdown("#### 📅 Fecha y Hora")

        max_date = date.today() + timedelta(days=15)
        selected_date = st.date_input(
            "Fecha:",
            value=date.today() + timedelta(days=1), # Sugerir mañana por defecto
            min_value=date.today(),
            max_value=max_date,
            key="prediction_date"
        )

        selected_hour = st.selectbox(
            "Hora:",
            range(24),
            format_func=lambda x: f"{x:02d}:00",
            index=9, # Por defecto 09:00
            key="prediction_hour"
        )

        # Selector de rango de horas
        st.markdown("#### ⏰ Rango de Predicción")
        hour_range = st.selectbox(
            "Número de horas a mostrar:",
            [3, 5, 7],
            index=1,  # Default 5
            help="Muestra predicciones para N horas centradas en la hora seleccionada",
            key="hour_range_selector"
        )

        # Botón de predicción
        predict_button = st.button(
            "🔮 Generar Predicción",
            type="primary",
            use_container_width=True
        )
    else:
        if prediction_type == "Ambas":
            st.warning("Selecciona una estación para desenganche y otra para enganche para generar la predicción.")
        else:
            st.warning("Selecciona una estación para continuar con la predicción.")


with col2:
    st.markdown("### 📊 Resultados y Visualización")

    # Mostrar mapa de estaciones (solo si no se ha presionado el botón de predicción)
    if not predict_button:
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
        # Calcular rango de horas
        half_range = hour_range // 2
        hours_to_predict = []

        for i in range(-half_range, half_range + 1):
            hour = (selected_hour + i) % 24
            hours_to_predict.append(hour)

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
                    for hour in hours_to_predict:
                        demand = predict_demand(selected_station_for_desenganche, selected_date, hour, "Desenganche")
                        level, css_class = get_demand_level(demand)
                        predictions_des.append({
                            'Hora': f"{hour:02d}:00",
                            'Demanda': demand,
                            'Nivel': level,
                            'Tipo': '🎯 Objetivo' if hour == selected_hour else '📍 Referencia'
                        })

                    df_des = pd.DataFrame(predictions_des)
                    st.dataframe(df_des, use_container_width=True, hide_index=True)

                    fig_des = go.Figure()
                    colors_des = ['#007bff' if p['Tipo'] == '🎯 Objetivo' else '#ffc107' for p in predictions_des]

                    fig_des.add_trace(go.Bar(
                        x=[p['Hora'] for p in predictions_des],
                        y=[p['Demanda'] for p in predictions_des],
                        marker_color=colors_des,
                        name='Desenganche',
                        text=[f"{p['Demanda']}<br>{p['Nivel']}" for p in predictions_des],
                        textposition='auto'
                    ))

                    fig_des.update_layout(
                        title=f"Desenganches - {selected_station_for_desenganche}",
                        xaxis_title="Hora",
                        yaxis_title="Desenganches Estimados",
                        height=400
                    )

                    st.plotly_chart(fig_des, use_container_width=True)
                else:
                    st.info("No se seleccionó estación para Desenganche.")

            with tab_enganche:
                if selected_station_for_enganche:
                    st.markdown(f"#### Predicciones de Enganche para: {selected_station_for_enganche}")
                    predictions_eng = []
                    for hour in hours_to_predict:
                        demand = predict_demand(selected_station_for_enganche, selected_date, hour, "Enganche")
                        level, css_class = get_demand_level(demand)
                        predictions_eng.append({
                            'Hora': f"{hour:02d}:00",
                            'Demanda': demand,
                            'Nivel': level,
                            'Tipo': '🎯 Objetivo' if hour == selected_hour else '📍 Referencia'
                        })

                    df_eng = pd.DataFrame(predictions_eng)
                    st.dataframe(df_eng, use_container_width=True, hide_index=True)

                    fig_eng = go.Figure()
                    colors_eng = ['#007bff' if p['Tipo'] == '🎯 Objetivo' else '#28a745' for p in predictions_eng]

                    fig_eng.add_trace(go.Bar(
                        x=[p['Hora'] for p in predictions_eng],
                        y=[p['Demanda'] for p in predictions_eng],
                        marker_color=colors_eng,
                        name='Enganche',
                        text=[f"{p['Demanda']}<br>{p['Nivel']}" for p in predictions_eng],
                        textposition='auto'
                    ))

                    fig_eng.update_layout(
                        title=f"Enganches - {selected_station_for_enganche}",
                        xaxis_title="Hora",
                        yaxis_title="Enganches Estimados",
                        height=400
                    )

                    st.plotly_chart(fig_eng, use_container_width=True)
                else:
                    st.info("No se seleccionó estación para Enganche.")

        else: # Lógica para "Desenganche" o "Enganche" individual
            if selected_station_single:
                st.markdown(f"#### Predicciones de {prediction_type} para: {selected_station_single}")
                predictions = []
                for hour in hours_to_predict:
                    demand = predict_demand(selected_station_single, selected_date, hour, prediction_type)
                    level, css_class = get_demand_level(demand)
                    predictions.append({
                        'Hora': f"{hour:02d}:00",
                        'Demanda': demand,
                        'Nivel': level,
                        'Tipo': '🎯 Objetivo' if hour == selected_hour else '📍 Referencia'
                    })

                df_results = pd.DataFrame(predictions)
                st.dataframe(df_results, use_container_width=True, hide_index=True)

                fig = go.Figure()
                colors = ['#007bff' if p['Tipo'] == '🎯 Objetivo' else '#ffc107' for p in predictions]

                fig.add_trace(go.Bar(
                    x=[p['Hora'] for p in predictions],
                    y=[p['Demanda'] for p in predictions],
                    marker_color=colors,
                    text=[f"{p['Demanda']}<br>{p['Nivel']}" for p in predictions],
                    textposition='auto'
                ))

                fig.update_layout(
                    title=f"Predicción de {prediction_type} - {selected_station_single}",
                    xaxis_title="Hora",
                    yaxis_title=f"{prediction_type}s Estimados",
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Selecciona una estación y presiona 'Generar Predicción'.")

    else: # Si no se ha presionado el botón o no se cumplen las condiciones iniciales
        # Este bloque ahora solo se ejecutará si NO se ha presionado el botón de predicción
        # y no hay un mapa de estaciones general o específico que se deba mostrar por defecto.
        # Aquí es donde estaba el mensaje de bienvenida.
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: #666;">
            <h3>🔮 Predictor Avanzado</h3>
            <p>Selecciona una estación y completa los parámetros para generar predicciones.</p>
            <p><strong>Nuevas funcionalidades:</strong></p>
            <ul style="text-align: left; max-width: 400px; margin: 0 auto;">
                <li>Predicción de enganches y desenganches</li>
                <li>Búsqueda avanzada de estaciones</li>
                <li>Rango configurable de horas (3, 5 o 7)</li>
                <li>Mapa interactivo de estaciones</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # --- BLOQUE DE POWER BI MOVido AQUÍ ---
    # Este bloque siempre se mostrará en col2, debajo de los resultados de la predicción (si hay)
    # o debajo del mensaje de bienvenida/mapa inicial.
    st.markdown("---") # Separador para que se vea claro el inicio del Power BI
    st.markdown("""
    <div class="powerbi-container">
        <h4>📊 Informe Interactivo de Power BI</h4>
        <p>Explora datos adicionales y métricas de rendimiento de BiciMAD.</p>
    </div>
    """, unsafe_allow_html=True)
    
    components.html(power_bi_embed_code, height=600, scrolling=True)
    # --- FIN DEL BLOQUE DE POWER BI MOVido ---


# Métricas del sistema
st.markdown("---")
st.markdown("### 📈 Estado del Sistema")

col_metrics1, col_metrics2, col_metrics3, col_metrics4, col_metrics5 = st.columns(5) # Cambié nombres para evitar conflicto con col1, col2

with col_metrics1:
    st.metric("Estaciones", f"{len(stations_data)}", "2")

with col_metrics2:
    st.metric("Bicis Disponibles", "2,156", "-45")

with col_metrics3:
    st.metric("Desenganches Hoy", "1,890", "123")

with col_metrics4:
    st.metric("Enganches Hoy", "1,845", "98")

with col_metrics5:
    st.metric("Precisión Modelo", "91.2%", "2.1%")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🚴‍♂️ Sistema Avanzado de Predicción BiciMAD | Desarrollado con Streamlit</p>
    <p><em>Predicciones de enganches y desenganches con IA | Datos simulados para demostración</em></p>
</div>
""", unsafe_allow_html=True)