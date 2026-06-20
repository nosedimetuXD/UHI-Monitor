"""
uhi_core.py
-----------
Núcleo de procesamiento para el Monitor de Islas de Calor Urbanas (UHI).
Mapeo matricial usando colecciones Landsat de la NASA/USGS y Google Earth Engine.
"""

from __future__ import annotations
import ee
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Inicialización y Geocodificación
# ---------------------------------------------------------------------------

def init_earth_engine(project: str | None = None) -> None:
    """Inicializa la API de Google Earth Engine con el proyecto especificado."""
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)


def geocode_city(city_name: str, country_name: str | None = None) -> ee.Geometry:
    """Geocodificación flexible con coincidencia parcial (stringContains)
    recorriendo los tres niveles del catálogo político-administrativo GAUL de la FAO.
    """
    levels = [
        ("FAO/GAUL/2015/level2", "ADM2_NAME"),  # Municipios / Ciudades
        ("FAO/GAUL/2015/level1", "ADM1_NAME"),  # Departamentos / Estados
        ("FAO/GAUL/2015/level0", "ADM0_NAME"),  # Países Completos
    ]
    for dataset_id, name_field in levels:
        fc = ee.FeatureCollection(dataset_id)
        query = fc.filter(ee.Filter.stringContains(name_field, city_name))
        if country_name and dataset_id != "FAO/GAUL/2015/level0":
            query = query.filter(ee.Filter.stringContains("ADM0_NAME", country_name))
        if query.size().getInfo() > 0:
            return query.first().geometry()
            
    raise ValueError(
        f"No se encontró '{city_name}' en el catálogo GAUL. "
        "Prueba con el nombre oficial completo o revisa la ortografía."
    )


# ---------------------------------------------------------------------------
# 2. Pipeline Satelital (Filtros de Nubes y Ventanas Temporales)
# ---------------------------------------------------------------------------

def mask_clouds(image: ee.Image) -> ee.Image:
    """Aplica una máscara de bits basada en la banda QA_PIXEL para remover nubes y sombras."""
    qa = image.select("QA_PIXEL")
    cloud_bit = 1 << 3
    shadow_bit = 1 << 4
    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(shadow_bit).eq(0))
    return image.updateMask(mask)


def _summer_window(roi: ee.Geometry, year: int) -> tuple[str, str]:
    """Determina la ventana de meses óptima según el hemisferio (Verano local)."""
    centroid_lat = roi.centroid().coordinates().get(1).getInfo()
    if centroid_lat >= 0:
        return f"{year}-06-01", f"{year}-08-31"  # Verano Hemisferio Norte
    return f"{year}-12-01", f"{year + 1}-02-28"  # Verano Hemisferio Sur


def get_summer_composite(roi: ee.Geometry, year: int) -> ee.Image:
    """Une las colecciones de Landsat 8 y 9, filtra por fecha/geometría y calcula la mediana."""
    start, end = _summer_window(roi, year)
    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
    collection = l8.merge(l9).filterBounds(roi).filterDate(start, end).map(mask_clouds)
    return collection.median().clip(roi)


# ---------------------------------------------------------------------------
# 3. Ecuaciones Físicas (NDVI, Emisividad de Sobrino y LST)
# ---------------------------------------------------------------------------

def compute_ndvi(image: ee.Image) -> ee.Image:
    """Calcula el Índice de Vegetación de Diferencia Normalizada usando las escalas de reflectancia."""
    nir = image.select("SR_B5").multiply(0.0000275).add(-0.2)
    red = image.select("SR_B4").multiply(0.0000275).add(-0.2)
    return nir.subtract(red).divide(nir.add(red)).rename("NDVI")


def compute_brightness_temp(image: ee.Image) -> ee.Image:
    """Convierte los datos de la banda térmica B10 a Temperatura de Brillo (BT) en Kelvin."""
    return image.select("ST_B10").multiply(0.00341802).add(149.0).rename("BT")


def compute_emissivity(ndvi: ee.Image) -> ee.Image:
    """Calcula la emisividad de la superficie (EM) mediante el método de umbral de NDVI de Sobrino."""
    pv = ndvi.subtract(0.2).divide(0.3).pow(2)
    emissivity_mixed = pv.multiply(0.02).add(0.97)
    emissivity = (
        ee.Image(0.97)
        .where(ndvi.gt(0.2).And(ndvi.lte(0.5)), emissivity_mixed)
        .where(ndvi.gt(0.5), 0.99)
    )
    return emissivity.rename("EMISSIVITY")


def compute_lst(image: ee.Image) -> ee.Image:
    """Ecuación matemática de transferencia radiativa para obtener la temperatura en Celsius (°C)."""
    ndvi = compute_ndvi(image)
    bt = compute_brightness_temp(image)
    emissivity = compute_emissivity(ndvi)
    wavelength = 10.895e-6
    rho = 1.438e-2
    lst_k = bt.expression(
        "BT / (1 + (lambda_ * BT / rho) * log(em))",
        {"BT": bt, "lambda_": wavelength, "rho": rho, "em": emissivity},
    )
    return lst_k.subtract(273.15).rename("LST")


# ---------------------------------------------------------------------------
# 4. Modelado y Reducción Estadística de la Anomalía UHI
# ---------------------------------------------------------------------------

def compute_rural_reference(
    lst: ee.Image,
    roi: ee.Geometry,
    ndvi: ee.Image,
    buffer_m: int = 15000,
    external_buffer: bool = True,
    scale: int = 100,
) -> ee.Number:
    """Calcula el valor térmico de referencia base (T_rural).
    Soporta anillo externo para ciudades o máscara interna vegetal para departamentos/países.
    """
    reference_region = roi if not external_buffer else roi.buffer(buffer_m).difference(roi)
    rural_lst = lst.updateMask(ndvi.gt(0.6))

    stats = rural_lst.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=reference_region,
        scale=scale,
        maxPixels=1e9,
        bestEffort=True,
        tileScale=2,
    )
    return ee.Number(stats.get("LST"))


def compute_uhi_matrix(lst: ee.Image, t_rural: ee.Number) -> ee.Image:
    """Resta la constante de referencia rural a cada píxel de LST para aislar la anomalía urbana."""
    return lst.subtract(ee.Image.constant(t_rural)).rename("UHI")


def compute_uhi_stats(
    uhi: ee.Image,
    lst: ee.Image,
    roi: ee.Geometry,
    risk_threshold_c: float = 3.0,
    scale: int = 100,
) -> ee.Dictionary:
    """Agrupa las bandas y calcula la media, el pico y el área en riesgo en una sola operación."""
    risk_mask = uhi.gt(risk_threshold_c).rename("RISK")
    stack = lst.rename("LST").addBands(uhi.rename("UHI")).addBands(risk_mask)

    combined_reducer = ee.Reducer.mean().combine(ee.Reducer.max(), sharedInputs=True)
    result = stack.reduceRegion(
        reducer=combined_reducer,
        geometry=roi,
        scale=scale,
        maxPixels=1e9,
        bestEffort=True,
        tileScale=2,
    )

    return ee.Dictionary(
        {
            "mean_lst_c": result.get("LST_mean"),
            "max_uhi_c": result.get("UHI_max"),
            "risk_area_pct": ee.Number(result.get("RISK_mean")).multiply(100),
        }
    )


# ---------------------------------------------------------------------------
# 5. Pipeline en Lote (Procesamiento Server-Side)
# ---------------------------------------------------------------------------

def analyze_year(
    roi: ee.Geometry,
    year: int,
    risk_threshold_c: float = 3.0,
    external_buffer: bool = True,
    scale: int = 100,
) -> dict:
    """Procesamiento aislado de un único año para alimentar el renderizado del mapa."""
    composite = get_summer_composite(roi, year)
    ndvi = compute_ndvi(composite)
    lst = compute_lst(composite)
    t_rural = compute_rural_reference(
        lst, roi, ndvi, external_buffer=external_buffer, scale=scale
    )
    uhi = compute_uhi_matrix(lst, t_rural)
    stats = compute_uhi_stats(uhi, lst, roi, risk_threshold_c, scale=scale).getInfo()
    stats["year"] = year
    stats["t_rural_c"] = t_rural.getInfo()
    return {"stats": stats, "lst_image": lst, "uhi_image": uhi, "ndvi_image": ndvi}


def build_timeseries_batch(
    roi: ee.Geometry,
    years: list[int],
    risk_threshold_c: float = 3.0,
    external_buffer: bool = True,
    scale: int = 100,
    buffer_m: int = 15000,
) -> pd.DataFrame:
    """Ejecuta toda la serie temporal de forma iterativa dentro del servidor de Earth Engine
    y descarga la matriz final consolidada en una única petición de red (.getInfo()).
    """
    centroid_lat = roi.centroid().coordinates().get(1).getInfo()
    boreal = centroid_lat >= 0
    reference_region = roi if not external_buffer else roi.buffer(buffer_m).difference(roi)

    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
    landsat = l8.merge(l9)

    combined_reducer = ee.Reducer.mean().combine(ee.Reducer.max(), sharedInputs=True)

    def year_to_feature(y):
        y = ee.Number(y)
        if boreal:
            start = ee.Date.fromYMD(y, 6, 1)
            end = ee.Date.fromYMD(y, 8, 31)
        else:
            start = ee.Date.fromYMD(y, 12, 1)
            end = ee.Date.fromYMD(y.add(1), 2, 28)

        composite = (
            landsat.filterBounds(roi)
            .filterDate(start, end)
            .map(mask_clouds)
            .median()
            .clip(roi)
        )

        ndvi = compute_ndvi(composite)
        lst = compute_lst(composite)

        rural_lst = lst.updateMask(ndvi.gt(0.6))
        t_rural = ee.Number(
            rural_lst.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=reference_region,
                scale=scale,
                maxPixels=1e9,
                bestEffort=True,
                tileScale=2,
            ).get("LST")
        )

        uhi = lst.subtract(t_rural)
        risk_mask = uhi.gt(risk_threshold_c).rename("RISK")
        stack = lst.rename("LST").addBands(uhi.rename("UHI")).addBands(risk_mask)

        result = stack.reduceRegion(
            reducer=combined_reducer,
            geometry=roi,
            scale=scale,
            maxPixels=1e9,
            bestEffort=True,
            tileScale=2,
        )

        return ee.Feature(
            None,
            {
                "year": y,
                "mean_lst_c": result.get("LST_mean"),
                "max_uhi_c": result.get("UHI_max"),
                "risk_area_pct": ee.Number(result.get("RISK_mean")).multiply(100),
                "t_rural_c": t_rural,
            },
        )

    fc = ee.FeatureCollection(ee.List(years).map(year_to_feature))
    features = fc.getInfo()["features"]
    rows = [f["properties"] for f in features]

    if not rows:
        return pd.DataFrame(
            columns=["year", "mean_lst_c", "max_uhi_c", "risk_area_pct", "t_rural_c"]
        )
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. Módulos de Regresión Espacial y Proyección Rástes a Futuro
# ---------------------------------------------------------------------------

def build_pixel_trend_collection(
    roi: ee.Geometry,
    years: list[int],
    risk_threshold_c: float = 3.0,
    external_buffer: bool = True,
    scale: int = 100,
    buffer_m: int = 15000,
) -> ee.ImageCollection:
    """Construye una colección de imágenes indexadas temporalmente de forma relativa (Año 0, 1, 2...)
    para estabilizar los cálculos de regresión lineal por píxel en Earth Engine.
    """
    centroid_lat = roi.centroid().coordinates().get(1).getInfo()
    boreal = centroid_lat >= 0
    reference_region = roi if not external_buffer else roi.buffer(buffer_m).difference(roi)

    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
    landsat = l8.merge(l9)
    
    anio_base = float(years[0])  # Origen temporal para evitar desbordes decimales

    def year_to_image(y):
        y = ee.Number(y)
        if boreal:
            start = ee.Date.fromYMD(y, 6, 1)
            end = ee.Date.fromYMD(y, 8, 31)
        else:
            start = ee.Date.fromYMD(y, 12, 1)
            end = ee.Date.fromYMD(y.add(1), 2, 28)

        composite = (
            landsat.filterBounds(roi)
            .filterDate(start, end)
            .map(mask_clouds)
            .median()
            .clip(roi)
        )
        ndvi = compute_ndvi(composite)
        lst = compute_lst(composite)

        rural_lst = lst.updateMask(ndvi.gt(0.6))
        t_rural = ee.Number(
            rural_lst.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=reference_region,
                scale=scale,
                maxPixels=1e9,
                bestEffort=True,
                tileScale=2,
            ).get("LST")
        )
        uhi = lst.subtract(t_rural)
        
        # Guardamos la distancia en años respecto al año base (Normalización)
        year_normalizado = y.subtract(anio_base)
        year_band = ee.Image.constant(year_normalizado).toFloat().rename("year")
        
        return year_band.addBands(lst.rename("LST")).addBands(uhi.rename("UHI")).clip(roi)

    return ee.ImageCollection(ee.List(years).map(year_to_image))


def project_future_layer_ajustada(
    collection: ee.ImageCollection,
    future_year: int,
    anio_base: int,
    band: str,
    slope_factor: float = 1.0,
    manual_shift: float = 0.0,
) -> ee.Image:
    """Ajusta una regresión por mínimos cuadrados píxel por píxel (linearFit) y calcula
    la extrapolación espacial limpia basándose en el delta del año relativo.
    """
    fit = collection.select(["year", band]).reduce(ee.Reducer.linearFit())
    
    # Calculamos la distancia de tiempo relativa limpia (Ej: 2026 - 2022 = 4)
    tiempo_relativo = float(future_year - anio_base)
    
    projected = (
        fit.select("scale")
        .multiply(slope_factor)
        .multiply(tiempo_relativo)
        .add(fit.select("offset"))
        .add(manual_shift)
        .rename(f"{band}_projected")
    )
    return projected


# ---------------------------------------------------------------------------
# 7. Modelos Estadísticos y Predicción Tabular (Pandas/Numpy)
# ---------------------------------------------------------------------------

def predict_future(df: pd.DataFrame, column: str, n_years: int = 5) -> pd.DataFrame:
    """Ajusta una regresión lineal clásica sobre el DataFrame histórico agregado."""
    if len(df) < 2:
        raise ValueError("Se necesitan al menos 2 años históricos para proyectar.")

    x = df["year"].to_numpy(dtype=float)
    y = df[column].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)

    last_year = int(x.max())
    future_years = np.arange(last_year + 1, last_year + 1 + n_years)
    predicted = slope * future_years + intercept

    y_fit = slope * x + intercept
    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    residual_std = float(np.std(y - y_fit, ddof=1)) if len(y) > 2 else 0.0

    future_df = pd.DataFrame({"year": future_years, column: predicted, "tipo": "proyección"})
    future_df.attrs["slope_per_year"] = float(slope)
    future_df.attrs["r_squared"] = r_squared
    future_df.attrs["residual_std"] = residual_std
    return future_df


def predict_future_table(
    df: pd.DataFrame, columns: list[str], n_years: int = 5
) -> tuple[pd.DataFrame, dict]:
    """Genera una matriz de predicción numérica multivariable mapeada para el reporte."""
    if len(df) < 2:
        raise ValueError("Se necesitan al menos 2 años históricos para proyectar.")

    last_year = int(df["year"].max())
    future_years = list(range(last_year + 1, last_year + 1 + n_years))
    table = {"year": future_years}
    meta = {}

    for col in columns:
        fdf = predict_future(df, col, n_years=n_years)
        table[col] = fdf[col].round(2).tolist()
        meta[col] = {
            "slope_per_year": round(fdf.attrs["slope_per_year"], 3),
            "r_squared": round(fdf.attrs["r_squared"], 3),
            "residual_std": round(fdf.attrs.get("residual_std", 0.0), 3),
        }

    return pd.DataFrame(table), meta