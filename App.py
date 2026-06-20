"""
app.py
------
Aplicación web de monitoreo de UHI.

Mejoras de esta versión:
  1. Análisis de país completo corregido: la referencia "rural" ya no
     usa un anillo externo sin sentido a esa escala, sino la vegetación
     interna de la propia región (ver uhi_core.compute_rural_reference).
  2. Mapa interactivo: un slider de año cambia la capa del mapa sin
     tener que volver a darle "Analizar" — cada año se cachea individual
     y perezosamente (solo se calcula cuando lo visitas por primera vez).
  3. Predicción numérica: tabla con los valores proyectados año por año
     para las 3 métricas, no solo la línea punteada del gráfico.
  4. Gráfica más legible: colores fijos, puntos con tooltip, línea
     divisoria entre histórico/proyección, títulos más claros.
  5. Procesamiento más rápido: la serie histórica se calcula con
     build_timeseries_batch (1 llamada de red para todos los años) y la
     escala de los píxeles se ajusta según el tamaño del área analizada.
"""

import time

import altair as alt
import ee
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

import uhi_core

st.set_page_config(page_title="UHI Monitor", page_icon="🌡️", layout="wide")

CURRENT_YEAR = 2026
METRIC_LABELS = {
    "mean_lst_c": "LST media de la zona (°C)",
    "max_uhi_c": "Pico máximo de UHI (°C)",
    "risk_area_pct": "% de territorio en riesgo",
}

# Escala de píxel (m) y modo de referencia rural según el tamaño del área.
# Un país completo con scale=100 puede tardar mucho o fallar por exceso
# de píxeles; con 500 m sigue siendo representativo y es mucho más rápido.
ESCALA_CONFIG = {
    "Ciudad / Municipio": {"scale": 100, "external_buffer": True, "zoom": 11},
    "Departamento / Estado": {"scale": 200, "external_buffer": False, "zoom": 7},
    "País Completo": {"scale": 500, "external_buffer": False, "zoom": 5},
}

GEE_PROJECT_ID = "uhi-global-monitor"


@st.cache_resource
def get_ee_session(project_id: str) -> bool:
    """Inicializa la sesión de Earth Engine de forma segura en la nube o local."""
    import json
    
    # 1. Intentamos leer las credenciales secretas de la nube (Streamlit Secrets)
    if "GEE_JSON" in st.secrets:
        try:
            # Parseamos el texto plano TOML/JSON que guardaste a un diccionario de Python
            creds_dict = json.loads(st.secrets["GEE_JSON"])
            credentials = ee.ServiceAccountCredentials(creds_dict["client_email"], key_data=creds_dict["private_key"])
            ee.Initialize(credentials=credentials, project=project_id)
            return True
        except Exception as e:
            st.error(f"Error de autenticación con la cuenta de servicio en la nube: {e}")
            
    # 2. Respaldo por si estás corriendo el archivo de forma local en tu Windows
    try:
        uhi_core.init_earth_engine(project=project_id)
        return True
    except Exception as e:
        st.error(f"No se pudo inicializar Earth Engine localmente: {e}")
        st.stop()

get_ee_session(GEE_PROJECT_ID)


@st.cache_data(show_spinner=False)
def obtener_geometria_ajustada(target_name, country_name, tipo_escala):
    if tipo_escala == "País Completo":
        fc = ee.FeatureCollection("FAO/GAUL/2015/level0")
        query = fc.filter(ee.Filter.stringContains("ADM0_NAME", country_name))
        if query.size().getInfo() == 0:
            raise ValueError(f"No se encontró el país '{country_name}' en el catálogo GAUL.")
        return query.first().geometry().getInfo()

    if tipo_escala == "Departamento / Estado":
        fc = ee.FeatureCollection("FAO/GAUL/2015/level1")
        query = fc.filter(ee.Filter.stringContains("ADM1_NAME", target_name))
        if country_name:
            query = query.filter(ee.Filter.stringContains("ADM0_NAME", country_name))
        if query.size().getInfo() > 0:
            return query.first().geometry().getInfo()
    else:
        fc = ee.FeatureCollection("FAO/GAUL/2015/level2")
        query = fc.filter(ee.Filter.stringContains("ADM2_NAME", target_name))
        if country_name:
            query = query.filter(ee.Filter.stringContains("ADM0_NAME", country_name))
        if query.size().getInfo() > 0:
            return query.first().geometry().getInfo()

    return uhi_core.geocode_city(target_name, country_name).getInfo()


@st.cache_data(show_spinner=False)
def obtener_centro_mapa(roi_geojson):
    roi = ee.Geometry(roi_geojson)
    lon, lat = roi.centroid().coordinates().getInfo()
    return lat, lon


@st.cache_data(show_spinner=False)
def calcular_serie_historica(roi_geojson, years_list, threshold, external_buffer, scale):
    """Serie completa en una sola llamada de red (ver build_timeseries_batch)."""
    roi = ee.Geometry(roi_geojson)
    return uhi_core.build_timeseries_batch(
        roi, years_list, threshold, external_buffer=external_buffer, scale=scale
    )


@st.cache_data(show_spinner=False)
def obtener_capa_de_year(roi_geojson, year, threshold, external_buffer, scale):
    """Calcula (y cachea) la capa del mapa SOLO para un año puntual.
    Esto es lo que hace posible el slider: mover el slider a un año ya
    visitado es instantáneo (no vuelve a tocar Earth Engine), y un año
    nuevo solo paga el costo de ese año, no de toda la serie."""
    roi = ee.Geometry(roi_geojson)
    resultado = uhi_core.analyze_year(
        roi, year, threshold, external_buffer=external_buffer, scale=scale
    )
    vis_lst = {"min": 20, "max": 42, "palette": ["blue", "cyan", "green", "yellow", "orange", "red"]}
    vis_uhi = {"min": -2, "max": 6, "palette": ["blue", "yellow", "orange", "red"]}

    url_lst = resultado["lst_image"].getMapId(vis_lst)["tile_fetcher"].url_format
    url_uhi = resultado["uhi_image"].getMapId(vis_uhi)["tile_fetcher"].url_format

    return {"url_lst": url_lst, "url_uhi": url_uhi, "stats": resultado["stats"]}


@st.cache_data(show_spinner=False)
def obtener_capa_proyectada(
    roi_geojson, years_list, threshold, external_buffer, scale, future_year, slope_factor, manual_shift
):
    """Genera (y cachea) una capa de mapa PROYECTADA a futuro usando años normalizados."""
    roi = ee.Geometry(roi_geojson)
    
    # Construimos la colección indexada de Earth Engine
    coleccion = uhi_core.build_pixel_trend_collection(
        roi, years_list, threshold, external_buffer=external_buffer, scale=scale
    )
    
    # Extraemos el año base (el primero del rango elegido, ej: 2022) para pasarlo como parámetro
    anio_base_historico = int(years_list[0])
    
    # CAMBIO CRÍTICO: Llamamos a la nueva función ajustada pasando el año base
    lst_proj = uhi_core.project_future_layer_ajustada(
        coleccion, future_year, anio_base_historico, "LST", slope_factor=slope_factor, manual_shift=manual_shift
    )
    uhi_proj = uhi_core.project_future_layer_ajustada(
        coleccion, future_year, anio_base_historico, "UHI", slope_factor=slope_factor, manual_shift=manual_shift
    )
    risk_proj = uhi_proj.gt(threshold).rename("RISK_projected")

    vis_lst = {"min": 20, "max": 42, "palette": ["blue", "cyan", "green", "yellow", "orange", "red"]}
    vis_uhi = {"min": -2, "max": 6, "palette": ["blue", "yellow", "orange", "red"]}

    url_lst = lst_proj.getMapId(vis_lst)["tile_fetcher"].url_format
    url_uhi = uhi_proj.getMapId(vis_uhi)["tile_fetcher"].url_format

    stack = lst_proj.addBands(uhi_proj).addBands(risk_proj)
    
    combined_reducer = (
        ee.Reducer.mean().combine(ee.Reducer.percentile([95]), sharedInputs=True)
    )
    stats = stack.reduceRegion(
        reducer=combined_reducer,
        geometry=roi,
        scale=scale,
        maxPixels=1e9,
        bestEffort=True,
        tileScale=2,
    ).getInfo()

    return {
        "url_lst": url_lst,
        "url_uhi": url_uhi,
        "mean_lst_c": stats.get("LST_projected_mean"),
        "max_uhi_c": stats.get("UHI_projected_p95"),
        "risk_area_pct": (stats.get("RISK_projected_mean") or 0) * 100,
    }


# ---------------------------------------------------------------------------
# BARRA LATERAL
# ---------------------------------------------------------------------------

st.sidebar.title("🌡️ UHI Monitor")
country = st.sidebar.text_input(
    "País (Obligatorio)",
    value="Colombia",
    help="Nombre del país donde está la zona a analizar. Se usa para "
    "desambiguar nombres repetidos (ej. 'Santiago' existe en varios países).",
).strip()
escala_seleccionada = st.sidebar.radio(
    "Escala del análisis:",
    ["Ciudad / Municipio", "Departamento / Estado", "País Completo"],
    index=1,
    help="A qué nivel quieres ver el fenómeno. 'Ciudad' compara el centro "
    "urbano contra el campo cercano. 'Departamento' y 'País' comparan las "
    "zonas más calientes de la región contra su propia vegetación interna, "
    "porque a esa escala no existe un 'afuera' cercano comparable.",
)

city = ""
if escala_seleccionada != "País Completo":
    city = st.sidebar.text_input(
        "Nombre de la zona (Ciudad o Departamento):",
        value="Bolivar",
        help="Nombre administrativo de la ciudad o departamento/estado, "
        "tal como aparece oficialmente (ej. 'Cartagena', 'Bolivar').",
    ).strip()

year_range = st.sidebar.slider(
    "Rango de años a comparar",
    min_value=2014,
    max_value=CURRENT_YEAR - 1,
    value=(2022, CURRENT_YEAR - 1),
    help="Define el periodo histórico que se analizará año por año con "
    "imágenes satelitales reales (Landsat). Mientras más años incluyas, "
    "más confiable es la tendencia para la proyección a futuro.",
)
years_selected = tuple(range(year_range[0], year_range[1] + 1))
risk_threshold = st.sidebar.slider(
    "Umbral de riesgo UHI (°C)",
    1.0,
    8.0,
    3.0,
    step=0.5,
    help="Diferencia de temperatura (en °C) por encima de la cual un punto "
    "se considera 'en riesgo' por isla de calor. Por ejemplo, con umbral 3°C, "
    "solo se cuenta como riesgo el territorio que está al menos 3°C más "
    "caliente que la referencia natural/rural de la zona.",
)
forecast_years = st.sidebar.slider(
    "Años a proyectar a futuro",
    1,
    10,
    5,
    help="Cuántos años hacia adelante quieres extrapolar la tendencia "
    "histórica (ej. 5 años proyecta hasta el año siguiente al último "
    "año analizado + 5). Es una proyección estadística simple, no un "
    "modelo climático.",
)

if "analisis_ejecutado" not in st.session_state:
    st.session_state.analisis_ejecutado = False

if st.sidebar.button("Analizar Territorio", type="primary"):
    if escala_seleccionada != "País Completo" and not city:
        st.sidebar.error("⚠️ Especifica el nombre de la ciudad o departamento.")
    elif not country:
        st.sidebar.error("⚠️ El campo 'País' es obligatorio.")
    else:
        st.session_state.analisis_ejecutado = True

nombre_pantalla = city if city else country
config = ESCALA_CONFIG[escala_seleccionada]
st.title(f"Análisis Térmico Superficial — {nombre_pantalla} ({escala_seleccionada})")

if not st.session_state.analisis_ejecutado:
    st.info("Configura los parámetros en la barra lateral y presiona **Analizar Territorio**.")
    st.stop()

try:
    with st.spinner(f"Localizando límites exactos de {nombre_pantalla}..."):
        roi_geojson = obtener_geometria_ajustada(city, country, escala_seleccionada)
    lat_c, lon_c = obtener_centro_mapa(roi_geojson)

    t0 = time.perf_counter()
    with st.spinner(f"Procesando {len(years_selected)} años en GEE..."):
        df_historico = calcular_serie_historica(
            roi_geojson, list(years_selected), risk_threshold, config["external_buffer"], config["scale"]
        )
    tiempo_calculo = time.perf_counter() - t0
except Exception as e:
    st.error(f"Error en el pipeline geográfico: {e}")
    st.stop()

if df_historico.empty:
    st.error("No se obtuvieron datos. Verifica que la zona tenga cobertura Landsat en el rango de años elegido.")
    st.stop()

st.caption(f"⏱️ Serie histórica de {len(years_selected)} años calculada en {tiempo_calculo:.2f} s.")

# ---------------------------------------------------------------------------
# PREDICCIÓN (se calcula aquí porque los indicadores y el mapa la usan)
# ---------------------------------------------------------------------------

tabla_prediccion = None
meta_prediccion = {}
try:
    tabla_prediccion, meta_prediccion = uhi_core.predict_future_table(
        df_historico, list(METRIC_LABELS.keys()), n_years=forecast_years
    )
except ValueError:
    pass  # se avisa más abajo, en la sección de proyección numérica

# ---------------------------------------------------------------------------
# MODO: histórico o proyección a futuro (afecta indicadores y mapa)
# ---------------------------------------------------------------------------

modo_mapa = st.radio(
    "¿Qué quieres ver?",
    ["Histórico (datos satelitales reales)", "Proyección a futuro"],
    horizontal=True,
    help="**Histórico** usa imágenes reales del satélite Landsat. **Proyección** "
    "ajusta la tendencia de los años analizados y la extiende a futuro — es una "
    "extrapolación estadística, no una observación satelital real. Afecta tanto "
    "los indicadores de abajo como el mapa.",
)

# ---------------------------------------------------------------------------
# INDICADORES CLAVE
# ---------------------------------------------------------------------------

col1, col2, col3 = st.columns(3)

if modo_mapa.startswith("Histórico") or tabla_prediccion is None:
    ultimo = df_historico.iloc[-1]
    col1.metric(
        f"LST media ({int(ultimo['year'])})",
        f"{ultimo['mean_lst_c']:.1f} °C",
        help="**LST (Land Surface Temperature)**: temperatura de la superficie "
        "del suelo medida por el satélite — no es la temperatura del aire que "
        "se siente, sino la del asfalto, techos, tierra o vegetación. Este "
        "valor es el promedio de toda la zona analizada en ese año.",
    )
    col2.metric(
        "Pico UHI máximo",
        f"{ultimo['max_uhi_c']:.1f} °C",
        help="**UHI (Urban Heat Island / Isla de Calor Urbana)**: cuánto más "
        "caliente está un punto comparado con la referencia 'natural' de la "
        "zona (vegetación densa). El 'pico máximo' es el punto más extremo "
        "encontrado — el lugar donde el efecto de isla de calor es más fuerte.",
    )
    col3.metric(
        "% territorio en riesgo",
        f"{ultimo['risk_area_pct']:.1f} %",
        help="Porcentaje del área total cuya diferencia de temperatura supera "
        "el **umbral de riesgo** que configuraste en la barra lateral. Si subes "
        "el umbral, este porcentaje baja, porque exige una diferencia más "
        "extrema para contar como 'en riesgo'.",
    )
else:
    promedio_proyectado = tabla_prediccion[["mean_lst_c", "max_uhi_c", "risk_area_pct"]].mean()
    rango_txt = f"{int(tabla_prediccion['year'].min())}–{int(tabla_prediccion['year'].max())}"
    col1.metric(
        f"LST media proyectada ({rango_txt})",
        f"{promedio_proyectado['mean_lst_c']:.1f} °C",
        help="Promedio de la LST media proyectada para TODOS los años futuros "
        "que estás extrapolando (no solo el año que ves en el mapa), siguiendo "
        "la tendencia histórica calculada.",
    )
    col2.metric(
        "Pico UHI medio proyectado",
        f"{promedio_proyectado['max_uhi_c']:.1f} °C",
        help="Promedio del pico UHI proyectado a lo largo de todos los años "
        "futuros del rango de proyección. Es una extrapolación estadística, "
        "no una medición satelital.",
    )
    col3.metric(
        "% territorio en riesgo (proyectado)",
        f"{promedio_proyectado['risk_area_pct']:.1f} %",
        help="Promedio del % de territorio en riesgo proyectado a lo largo de "
        "todos los años futuros del rango de proyección, usando el umbral de "
        "riesgo configurado en la barra lateral.",
    )

# ---------------------------------------------------------------------------
# MAPA INTERACTIVO con slider de año
# ---------------------------------------------------------------------------

st.subheader("🗺️ Mapa de calor superficial")

layer_choice = st.radio(
    "Capa a visualizar",
    ["LST (temperatura superficial)", "Intensidad UHI"],
    horizontal=True,
    help="**LST**: temperatura de superficie en °C, tal cual la mide el "
    "satélite. **Intensidad UHI**: cuánto más caliente está cada punto "
    "comparado con la referencia natural de la zona — entre más rojo, "
    "mayor efecto de isla de calor en ese lugar.",
)

if modo_mapa.startswith("Histórico"):
    año_mapa = st.slider(
        "Año a visualizar en el mapa",
        min_value=years_selected[0],
        max_value=years_selected[-1],
        value=years_selected[-1],
        help="Mueve el slider para ver cómo cambió la temperatura/UHI año por año. "
        "Los años ya visitados se cargan al instante (quedan en caché).",
    )
    with st.spinner(f"Generando capa del mapa para {año_mapa}..."):
        capa = obtener_capa_de_year(
            roi_geojson, año_mapa, risk_threshold, config["external_buffer"], config["scale"]
        )
    mapa_key = f"hist_{año_mapa}"
    pie_mapa = (
        f"📅 Año {año_mapa} (datos satelitales reales) — "
        f"LST media: {capa['stats']['mean_lst_c']:.1f} °C | "
        f"Pico UHI: {capa['stats']['max_uhi_c']:.1f} °C | "
        f"Territorio en riesgo: {capa['stats']['risk_area_pct']:.1f} %"
    )
elif tabla_prediccion is None:
    st.warning(
        "⚠️ No hay suficientes años históricos para proyectar el mapa a futuro "
        "(amplía el rango de años en la barra lateral)."
    )
    st.stop()
else:
    año_proyectado = st.select_slider(
        "Año futuro a proyectar en el mapa",
        options=tabla_prediccion["year"].tolist(),
        help="Año futuro extrapolado a partir de la tendencia de cada píxel. "
        "No corresponde a una imagen satelital real, ya que esos años aún no han ocurrido.",
    )

    st.markdown("**🎛️ Escenario hipotético** _(opcional — mueve estos sliders para ver 'qué pasaría si...')_")
    esc_col1, esc_col2 = st.columns(2)
    slope_factor = esc_col1.slider(
        "Factor de la tendencia",
        min_value=-1.5,
        max_value=3.0,
        value=1.0,
        step=0.1,
        help="Multiplica la tendencia histórica observada. **1.0× = se respeta "
        "la tendencia tal cual** (sin supuestos extra). Valores mayores la "
        "aceleran (ej. simula más urbanización o pérdida de áreas verdes). "
        "Valores menores la frenan. **0× = se queda plano**, sin cambio futuro. "
        "**Negativo = invierte la tendencia**, simulando una mitigación fuerte "
        "(ej. reforestación masiva, techos verdes a gran escala).",
    )
    manual_shift = esc_col2.slider(
        "Ajuste manual por intervención (°C)",
        min_value=-5.0,
        max_value=5.0,
        value=0.0,
        step=0.5,
        help="Suma (o resta) un valor fijo, parejo en toda la zona, además de "
        "la tendencia. Sirve para simular una intervención puntual de magnitud "
        "conocida — ej. -2°C por pintar techos blancos en toda la ciudad, o "
        "+1°C por perder la cobertura vegetal que queda. Se aplica por igual a "
        "la LST y al UHI proyectados.",
    )
    if slope_factor == 1.0 and manual_shift == 0.0:
        st.caption("Mostrando la proyección **neutra** (la tendencia histórica, sin ajustes de escenario).")
    else:
        st.caption(
            f"⚙️ Escenario modificado: tendencia × **{slope_factor:.1f}**, "
            f"ajuste manual de **{manual_shift:+.1f} °C**. Esto YA NO es la "
            "tendencia histórica observada, es una simulación que tú definiste."
        )

    with st.spinner(f"Ajustando tendencia por píxel y proyectando {año_proyectado}..."):
        capa = obtener_capa_proyectada(
            roi_geojson,
            list(years_selected),
            risk_threshold,
            config["external_buffer"],
            config["scale"],
            año_proyectado,
            slope_factor,
            manual_shift,
        )
    st.info(
        f"🔮 Mostrando una **proyección** para {año_proyectado}: Earth Engine ajustó "
        "una tendencia lineal en cada píxel usando los años analizados y la extendió "
        "hasta este año. No es una imagen satelital real, es una extrapolación estadística."
    )
    pie_mapa = (
        f"📅 Año {año_proyectado} (proyectado) — "
        f"LST media proyectada: {capa['mean_lst_c']:.1f} °C | "
        f"Pico UHI proyectado: {capa['max_uhi_c']:.1f} °C | "
        f"Territorio en riesgo proyectado: {capa['risk_area_pct']:.1f} %"
    )
    mapa_key = f"proy_{año_proyectado}_{slope_factor}_{manual_shift}"

m = folium.Map(location=[lat_c, lon_c], zoom_start=config["zoom"])
url_capa = capa["url_lst"] if layer_choice.startswith("LST") else capa["url_uhi"]
folium.TileLayer(
    tiles=url_capa, attr="Google Earth Engine - Landsat", overlay=True, opacity=0.7, name=layer_choice
).add_to(m)
st_folium(m, width=900, height=520, key=f"mapa_{nombre_pantalla}_{layer_choice}_{mapa_key}", returned_objects=[])

st.caption(pie_mapa)

# ---------------------------------------------------------------------------
# DATOS ESTADÍSTICOS CONCRETOS
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("📋 Datos Estadísticos Concretos")
df_informe = df_historico.copy().rename(
    columns={
        "year": "Año",
        "mean_lst_c": "Temperatura Media (°C)",
        "max_uhi_c": "Pico Máximo UHI (°C)",
        "risk_area_pct": "Territorio en Riesgo (%)",
        "t_rural_c": "Referencia Base (°C)",
    }
)
columnas_ordenadas = ["Año", "Temperatura Media (°C)", "Pico Máximo UHI (°C)", "Territorio en Riesgo (%)", "Referencia Base (°C)"]
df_informe = df_informe[columnas_ordenadas]

st.dataframe(
    df_informe,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Año": st.column_config.NumberColumn(
            "Año", format="%d", help="Año al que corresponde la fila (verano de ese año)."
        ),
        "Temperatura Media (°C)": st.column_config.NumberColumn(
            "Temperatura Media (°C)",
            format="%.2f °C",
            help="LST (Land Surface Temperature) promedio de toda la zona en ese año: "
            "la temperatura de la superficie medida por satélite, no la del aire.",
        ),
        "Pico Máximo UHI (°C)": st.column_config.NumberColumn(
            "Pico Máximo UHI (°C)",
            format="%.2f °C",
            help="La mayor diferencia de temperatura encontrada entre un punto de la "
            "zona y la referencia natural/rural — el punto más caliente respecto a su entorno.",
        ),
        "Territorio en Riesgo (%)": st.column_config.NumberColumn(
            "Territorio en Riesgo (%)",
            format="%.2f %%",
            help="Porcentaje del área cuya diferencia de temperatura superó el "
            "umbral de riesgo configurado en la barra lateral.",
        ),
        "Referencia Base (°C)": st.column_config.NumberColumn(
            "Referencia Base (°C)",
            format="%.2f °C",
            help="Temperatura 'natural' de referencia (vegetación densa) usada para "
            "calcular la anomalía UHI de ese año — el punto de comparación de todo lo demás.",
        ),
    },
)

st.download_button(
    label="📥 Descargar datos concretos (CSV)",
    data=df_informe.to_csv(index=False).encode("utf-8"),
    file_name=f"reporte_termico_{nombre_pantalla}.csv",
    mime="text/csv",
)

# ---------------------------------------------------------------------------
# PREDICCIÓN NUMÉRICA A FUTURO
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("🔮 Proyección Numérica a Futuro")

metric_choice = st.selectbox(
    "Métrica a graficar / proyectar", options=list(METRIC_LABELS.keys()), format_func=lambda c: METRIC_LABELS[c]
)

if tabla_prediccion is not None:
    tabla_mostrar = tabla_prediccion.rename(columns={"year": "Año", **METRIC_LABELS})
    st.dataframe(
        tabla_mostrar,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Año": st.column_config.NumberColumn(
                "Año", format="%d", help="Año futuro proyectado (todavía no ha ocurrido)."
            ),
            METRIC_LABELS["mean_lst_c"]: st.column_config.NumberColumn(
                METRIC_LABELS["mean_lst_c"],
                format="%.2f °C",
                help="Valor proyectado de la temperatura de superficie (LST) promedio "
                "para ese año futuro, siguiendo la tendencia histórica.",
            ),
            METRIC_LABELS["max_uhi_c"]: st.column_config.NumberColumn(
                METRIC_LABELS["max_uhi_c"],
                format="%.2f °C",
                help="Valor proyectado del punto más caliente respecto a la referencia "
                "natural, siguiendo la tendencia histórica.",
            ),
            METRIC_LABELS["risk_area_pct"]: st.column_config.NumberColumn(
                METRIC_LABELS["risk_area_pct"],
                format="%.2f %%",
                help="Porcentaje proyectado de territorio que superaría el umbral de "
                "riesgo, siguiendo la tendencia histórica.",
            ),
        },
    )

    info = meta_prediccion[metric_choice]
    trend_word = "subiendo" if info["slope_per_year"] > 0 else "bajando"
    st.caption(
        f"Para **{METRIC_LABELS[metric_choice]}**: tendencia {trend_word} "
        f"≈ {abs(info['slope_per_year']):.2f} unidades/año (R² = {info['r_squared']:.2f}). "
        "Es una proyección estadística simple sobre el histórico, no un modelo climático físico."
    )
else:
    st.warning(
        "⚠️ Se necesitan al menos 2 años históricos para proyectar (amplía el "
        "rango de años en la barra lateral)."
    )

# ---------------------------------------------------------------------------
# GRÁFICA — histórico + proyección, más legible
# ---------------------------------------------------------------------------

st.subheader("📈 Evolución Visual")

chart_df = df_historico[["year", metric_choice]].copy()
chart_df["tipo"] = "histórico"

if tabla_prediccion is not None:
    forecast_chart_df = tabla_prediccion[["year", metric_choice]].copy()
    forecast_chart_df["tipo"] = "proyección"
    combined = pd.concat([chart_df, forecast_chart_df], ignore_index=True)
else:
    combined = chart_df

color_scale = alt.Scale(domain=["histórico", "proyección"], range=["#1f77b4", "#ff7f0e"])

base = alt.Chart(combined).encode(
    x=alt.X("year:O", title="Año", axis=alt.Axis(labelAngle=0)),
    y=alt.Y(f"{metric_choice}:Q", title=METRIC_LABELS[metric_choice], scale=alt.Scale(zero=False)),
    color=alt.Color("tipo:N", title="", scale=color_scale),
    tooltip=[
        alt.Tooltip("year:O", title="Año"),
        alt.Tooltip(f"{metric_choice}:Q", title=METRIC_LABELS[metric_choice], format=".2f"),
        alt.Tooltip("tipo:N", title="Tipo de dato"),
    ],
)

linea = base.mark_line(strokeWidth=3).encode(
    strokeDash=alt.condition(alt.datum.tipo == "proyección", alt.value([6, 4]), alt.value([0]))
)
puntos = base.mark_point(size=80, filled=True)

ultimo_historico = int(chart_df["year"].max())
regla_division = (
    alt.Chart(pd.DataFrame({"year": [ultimo_historico]}))
    .mark_rule(strokeDash=[2, 2], color="gray", opacity=0.6)
    .encode(x="year:O")
)

chart = (
    (linea + puntos + regla_division)
    .properties(height=380, title=f"{METRIC_LABELS[metric_choice]} — histórico y proyección")
    .configure_axis(labelFontSize=12, titleFontSize=13, gridOpacity=0.3)
    .configure_legend(orient="top", labelFontSize=12)
    .configure_title(fontSize=16)
)

st.altair_chart(chart, use_container_width=True)