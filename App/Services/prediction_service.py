
import pandas as pd
from App.Ml.model import load_model
import requests
import numpy as np
import json
from pathlib import Path


# Predicción recursiva mediante lags de 2, 24, 48 y 72 horas
async def predict_from_excel(file, lat, long):

    MODELS_DIR = Path("Models")
    TARGET = "electricity_demand_values[kw]"

    model = load_model()

    # ==========================================
    # 1. CARGAR CARACTERÍSTICAS DEL MODELO
    # ==========================================

    with open(MODELS_DIR / "features.json", "r", encoding="utf-8") as f:
        features = json.load(f)

    # ==========================================
    # 2. LEER Y LIMPIAR EXCEL
    # ==========================================

    df = pd.read_excel(file.file)

    df["Time"] = pd.to_datetime(
        df["Time"],
        format="%m/%d/%Y %H:%M",
        errors="coerce"
    )

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce"
    ).astype("float64")

    # Eliminar filas sin fecha válida
    df = df.dropna(subset=["Time"]).copy()

    # Ordenar cronológicamente
    df = df.sort_values("Time").reset_index(drop=True)

    # Eliminar fechas duplicadas, conservando la última
    df = df.drop_duplicates(
        subset=["Time"],
        keep="last"
    ).reset_index(drop=True)

    if len(df) < 168:
        raise ValueError(
            "Se necesitan al menos 168 registros históricos "
            "para calcular los lags y medias móviles."
        )

    # Datos históricos para comparar en el frontend
    df_previous = df.tail(720).copy()

    # Datos utilizados para construir los lags
    df24h = df.tail(170).copy()

    # No eliminar la última fila: necesitamos la fecha más reciente
    df24h = df24h.reset_index(drop=True)

    # Comprobar fecha inicial
    ultima_fecha = df24h["Time"].iloc[-1]

    if pd.isna(ultima_fecha):
        raise ValueError("La última fecha del Excel no es válida.")

    print("Última fecha histórica:", ultima_fecha)

    # ==========================================
    # 3. CONSULTAR API METEOROLÓGICA
    # ==========================================

    start_datetime = ultima_fecha + pd.Timedelta(hours=1)
    end_datetime = start_datetime + pd.Timedelta(days=30)

    start_date = start_datetime.strftime("%Y-%m-%d")
    end_date30dias = end_datetime.strftime("%Y-%m-%d")

    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={long}"
        f"&start_date={start_date}"
        f"&end_date={end_date30dias}"
        "&hourly="
        "surface_pressure,"
        "temperature_2m,"
        "relative_humidity_2m,"
        "wind_speed_10m,"
        "shortwave_radiation,"
        "cloud_cover"
        "&wind_speed_unit=ms"
        "&timezone=auto"
    )

    print(
        f"Consultando clima por hora: "
        f"{lat}, {long} ({start_date} al {end_date30dias})"
    )

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    data = response.json()

    if "hourly" not in data:
        raise ValueError(
            f"La API meteorológica no devolvió datos horarios: {data}"
        )

    hourly = data["hourly"]

    # Crear un diccionario para encontrar rápidamente cada hora
    weather_index = {
        fecha: i
        for i, fecha in enumerate(hourly["time"])
    }

    predictions_rows = []

    # ==========================================
    # 4. PREDICCIÓN RECURSIVA DE 720 HORAS
    # ==========================================

    for i in range(720):

        # Obtener siempre la última fecha válida
        ultima_fecha = pd.to_datetime(
            df24h["Time"].iloc[-1],
            errors="coerce"
        )

        if pd.isna(ultima_fecha):
            raise ValueError(
                f"Fecha inválida en la iteración {i}. "
                f"Últimas filas:\n{df24h[['Time']].tail()}"
            )

        fechaprediccion = (
            ultima_fecha + pd.Timedelta(hours=1)
        )

        next_fecha_format = fechaprediccion.strftime(
            "%Y-%m-%dT%H:%M"
        )

        print(
            f"Iteración {i + 1}/720 - "
            f"Fecha: {next_fecha_format}"
        )

        # ------------------------------------------
        # Obtener datos meteorológicos de esa hora
        # ------------------------------------------

        if next_fecha_format not in weather_index:
            raise ValueError(
                f"No existen datos meteorológicos para "
                f"{next_fecha_format}. "
                f"Revisa el rango de fechas de Open-Meteo."
            )

        index = weather_index[next_fecha_format]

        # ------------------------------------------
        # Construir nueva fila
        # IMPORTANTE: utilizar Time, no Fecha_Hora
        # ------------------------------------------

        nueva_fila = pd.DataFrame({

            "Time": [fechaprediccion],

            "air_pressure_mmHg": [
                hourly["surface_pressure"][index] / 1.33322
                if hourly["surface_pressure"][index] is not None
                else np.nan
            ],

            "air_temperature_degree_celcius": [
                hourly["temperature_2m"][index]
            ],

            "relative_humidity": [
                hourly["relative_humidity_2m"][index]
            ],

            "wind_speed_M_S": [
                hourly["wind_speed_10m"][index]
            ],

            "solar_irridiation_W_m": [
                hourly["shortwave_radiation"][index]
            ],

            "total_cloud_cover_from_ten_10_10": [0],

            "total_cloud_cover_from_ten_2_10_3_10": [0],

            "total_cloud_cover_from_ten_4_10": [0],

            "total_cloud_cover_from_ten_5_10": [0],

            "total_cloud_cover_from_ten_7_10_8_10": [0],

            "total_cloud_cover_from_ten_Sky_obscured_by_fog_and_or_other_meteorological_phenomena": [0],

            "total_cloud_cover_from_ten_no_clouds": [0],

            "total_cloud_cover_from_ten_unknown": [0],

            TARGET: np.array([0.0], dtype=np.float64),

            "hora": [fechaprediccion.hour],

            "dia_semana": [fechaprediccion.dayofweek],

            "mes": [fechaprediccion.month],

            "dia_mes": [fechaprediccion.day],

            "es_fin_semana": [
                int(fechaprediccion.dayofweek >= 5)
            ],

            "hora_sin": [
                np.sin(2 * np.pi * fechaprediccion.hour / 24)
            ],

            "hora_cos": [
                np.cos(2 * np.pi * fechaprediccion.hour / 24)
            ],

            "dia_sin": [
                np.sin(2 * np.pi * fechaprediccion.dayofweek / 7)
            ],

            "dia_cos": [
                np.cos(2 * np.pi * fechaprediccion.dayofweek / 7)
            ],

            "consumo_lag_1": [
                df24h[TARGET].iloc[-1]
            ],

            "consumo_lag_2": [
                df24h[TARGET].iloc[-2]
            ],

            "consumo_lag_3": [
                df24h[TARGET].iloc[-3]
            ],

            "consumo_lag_24": [
                df24h[TARGET].iloc[-24]
            ],

            "consumo_lag_48": [
                df24h[TARGET].iloc[-48]
            ],

            "consumo_lag_72": [
                df24h[TARGET].iloc[-72]
            ],

            "consumo_lag_168": [
                df24h[TARGET].iloc[-168]
            ],

            "consumo_mean_3": [
                df24h[TARGET].iloc[-3:].mean()
            ],

            "consumo_mean_24": [
                df24h[TARGET].iloc[-24:].mean()
            ],

            "consumo_mean_168": [
                df24h[TARGET].iloc[-168:].mean()
            ],

        })

        # ------------------------------------------
        # Clasificar nubosidad
        # ------------------------------------------

        cloud = hourly["cloud_cover"][index]

        if cloud is None:
            cloud_category = "unknown"
        elif cloud == 0:
            cloud_category = "no_clouds"
        elif cloud >= 90:
            cloud_category = "10_10"
        elif cloud >= 65:
            cloud_category = "7_10_8_10"
        elif cloud >= 45:
            cloud_category = "5_10"
        elif cloud >= 35:
            cloud_category = "4_10"
        else:
            cloud_category = "2_10_3_10"

        cloud_column = (
            f"total_cloud_cover_from_ten_{cloud_category}"
        )

        nueva_fila.loc[0, cloud_column] = 1

        # ------------------------------------------
        # Preparar características de XGBoost
        # ------------------------------------------

        dfXPredict = nueva_fila[features].copy()

        dfXPredict = dfXPredict.apply(
            pd.to_numeric,
            errors="coerce"
        )

        # Revisar valores nulos
        if dfXPredict.isnull().any().any():

            columnas_nulas = dfXPredict.columns[
                dfXPredict.isnull().any()
            ].tolist()

            print("\nERROR: CARACTERÍSTICAS VACÍAS")

            for columna in columnas_nulas:
                print(
                    columna,
                    "=>",
                    dfXPredict[columna].values
                )

            raise ValueError(
                f"Valores nulos en iteración {i + 1}: "
                f"{columnas_nulas}"
            )

        # ------------------------------------------
        # Realizar predicción
        # ------------------------------------------

        prediccion = float(model.predict(dfXPredict)[0])

        print("Consumo predicho:", prediccion)

        # Guardar el consumo predicho en la nueva fila
        nueva_fila.loc[0, TARGET] = prediccion

        # Guardar resultado para el frontend
        predictions_rows.append({
            "Fecha_Hora": fechaprediccion,
            "Consumo": prediccion
        })

        # ------------------------------------------
        # Actualizar historial para siguiente iteración
        # ------------------------------------------

        df24h = pd.concat(
            [df24h, nueva_fila],
            ignore_index=True,
            sort=False
        ).tail(400).reset_index(drop=True)

        # Comprobación de integridad temporal
        if pd.isna(df24h["Time"].iloc[-1]):
            raise ValueError(
                f"La fecha se perdió en la iteración {i + 1}"
            )

    # ==========================================
    # 5. EXPORTAR RESULTADOS
    # ==========================================

    df24h.to_excel("excelAPI333.xlsx", index=False)

    # ==========================================
    # 6. PREPARAR DATOS PARA EL FRONTEND
    # ==========================================

    df_current = pd.DataFrame(predictions_rows)

    df_current["Fecha_Hora"] = pd.to_datetime(
        df_current["Fecha_Hora"]
    )

    chart_data = []

    for i in range(min(len(df_current), len(df_previous))):

        chart_data.append({

            "month": df_current.iloc[i][
                "Fecha_Hora"
            ].strftime("%Y-%m-%dT%H:%M:%S"),

            "current": float(
                df_current.iloc[i]["Consumo"]
            ),

            "previous": float(
                df_previous.iloc[i][TARGET]
            )

        })

    print("Predicción finalizada correctamente.")

    return {
        "predictions": chart_data
    }
