# 🌡️ UHI Advanced Monitor

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Version](https://img.shields.io/badge/version-1.1.0-green.svg)
![Stack](https://img.shields.io/badge/stack-Python%20%7C%20Streamlit%20%7C%20GEE-orange)
![Data](https://img.shields.io/badge/satellites-NASA%20Landsat%208%20%26%209-blue)

**Plataforma interactiva de monitoreo y simulación de Islas de Calor Urbanas (UHI).**

Este monitor nació para descentralizar el análisis geoespacial climático avanzado, permitiendo mapear la Temperatura Superficial del Suelo (LST) y predecir anomalías térmicas críticas sin necesidad de usar un software GIS pesado, directamente desde el navegador.

---

## El ecosistema del pipeline

La aplicación opera bajo una arquitectura desacoplada que separa el cómputo geoespacial masivo en la nube de la interfaz interactiva de usuario:

### ⚙️ uhi_core (El Motor Lógico)
Es el núcleo físico y computacional que se conecta directamente con los servidores de la NASA y el USGS mediante Google Earth Engine. 
- **Física de Emisividad:** Implementa el método de umbral de NDVI (Modelo de Sobrino) para aislar la emisividad de los materiales (concreto, asfalto, agua) de la banda térmica `ST_B10`.
- **Cómputo en Lote (*Server-Side*):** Optimización avanzada mediante funciones mapeadas en la nube (`build_timeseries_batch`), reduciendo drásticamente la latencia de red al consolidar toda la serie de años en una única consulta.
- **Modelado de Referencia Dinámico:** Adapta matemáticamente la línea base rural según la escala elegida (anillo *buffer* externo de 15 km para ciudades, o filtros de vegetación interna densa con NDVI > 0.6 para departamentos enteros).

### 🖊️ App (La Interfaz Web)
El frontend interactivo diseñado con Streamlit que procesa los datos y los expone de forma amigable para la toma de decisiones.
- **Renderizado Dinámico:** Renderiza mapas interactivos en tiempo real con Folium empleando *caching* perezoso (*lazy loading*) por año.
- **Simulador de Escenarios Hipotéticos:** Dispone de controles para modificar la pendiente lineal por píxel (`ee.Reducer.linearFit`) y proyectar mapas a futuro bajo supuestos controlados (como reforestación urbana o pérdida de áreas verdes).
- **Exportación Abierta:** Tablas de datos concretos sin índices innecesarios y descarga directa de reportes consolidados en formato CSV.

---

## Features y Roadmap Técnico

### Gestión de Capas y Escalas Administrativas

| Escala del Análisis | Filtro Catálogo FAO | Resolución Malla (`scale`) | Comportamiento Referencia Rural |
|---|---|---|---|
| **Ciudad / Municipio** | `level2` | 100 metros | Anillo amortiguador externo de 15 km |
| **Departamento / Estado** | `level1` | 200 metros | Aislamiento de vegetación interna |
| **País Completo** | `level0` | 500 metros | Aislamiento de vegetación interna |

### Estado de las Capas del Mapa

| Capa Base Ráster | Tipo de Dato | Origen / Algoritmo | Estado |
|---|---|---|---|
| LST (Temperatura Superficial) | Histórico Real | Landsat 8/9 Thermal Infrared | ✅ |
| Intensidad UHI (Anomalía) | Histórico Real | Matriz de diferencia vs Línea Base | ✅ |
| LST Proyectada a Futuro | Proyección Estadística | `linearFit` Normalizado por píxel | ✅ |
| Intensidad UHI Proyectada | Proyección Estadística | Extrapolación de Delta Temporal | ✅ |
| Escenarios Modificados (Simulación) | Escenario Hipotético | Multiplicador de tendencia + Ajuste manual | ✅ |

---

## Stack Tecnológico

| Capa | Tecnología | Función |
|---|---|---|
| **Geo-Cómputo** | Google Earth Engine API (`ee`) | Procesamiento masivo de píxeles satelitales en la nube. |
| **Datos Base** | NASA / USGS Landsat (C02/T1_L2) | Imágenes multiespectrales e infrarrojas térmicas de verano. |
| **UI Framework** | Streamlit | Construcción del frontend web dinámico y reactivo. |
| **Mapas** | Folium + `streamlit_folium` | Renderizado de mosaicos e interactividad espacial. |
| **Análisis Tabular**| Pandas + NumPy | Estructuración, ordenamiento de datos y regresión lineal agregada. |
| **Gráficas** | Altair | Gráficos vectoriales de evolución temporal con tooltips interactivos. |

---

## Getting Started

### Prerrequisitos

- Python 3.10 o 3.11 instalado (Asegúrate de marcar la casilla *"Add Python to PATH"*).
- Una terminal de Windows (PowerShell o CMD).

### Instalación Local

```powershell
# 1. Crear el directorio de trabajo y moverte a él
mkdir uhi_app_local
cd uhi_app_local

# 2. Configurar y activar el entorno virtual aislado de Windows
python -m venv venv
.\venv\Scripts\activate

# 3. Instalar las dependencias del proyecto
pip install earthengine-api streamlit streamlit-folium folium pandas altair numpy scipy

# 4. Autenticar tu máquina local con Google Earth Engine
earthengine authenticate