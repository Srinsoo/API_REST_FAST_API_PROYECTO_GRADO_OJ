import pandas as pd
import requests


def preprocess_data(df,latitude,longitude,etapa):

    df = df.dropna()
    # Creamos una columna solo con la fecha (YYYY-MM-DD) sin la hora para buscar el clima

    if 'Consumo' not in df.columns:
        return "No existe la columna Consumo en el registro"

    if 'Fecha_Hora' not in df.columns:
        return "No existe la columna Fecha_Hora en el registro"

    df['Fecha_Hora'] = pd.to_datetime(df['Fecha_Hora'])
    df['Fecha'] = df['Fecha_Hora'].dt.normalize()

    #Fecha de inicio y finalización del registro
    start_date = df['Fecha'].min().strftime('%Y-%m-%d')
    end_date = df['Fecha'].max().strftime('%Y-%m-%d')
   
    # Parametros coordenadas Sincelejo
    # 3. Coordenadas de Sincelejo, Sucre
    # lat = 9.3045
    # lon = -75.3905

    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={latitude}&longitude={longitude}&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation"
    )

    print(f"Consultando clima por hora para la ubicación({latitude} y {longitude}) ({start_date} al {end_date})...")
    response = requests.get(url)
    data = response.json()

    target = 'Consumo'

    if response.status_code == 200 and 'hourly' in data:
    # 4. Organizar la respuesta horaria de la API en una tabla temporal
        hourly_data = data['hourly']
        df_api = pd.DataFrame({
            'Fecha_Hora': pd.to_datetime(hourly_data['time']), # La API devuelve la hora exacta
            'Temperatura (°C)': hourly_data['temperature_2m'],
            'Humedad Relativa (%)': hourly_data['relative_humidity_2m'],
            'Sensación Térmica (°C)': hourly_data['apparent_temperature'],
            # 'Precipitación (mm)': hourly_data['precipitation']
        })

        # 5. Crear la columna del Día de la Semana en español basada en la fecha/hora
        dias_espanol = {
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
            'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6
        }

        # --- Definimos columnas adicionales -----

        # Hora
        df_api['hora'] = df['Fecha_Hora'].dt.hour
        #dia de la semana
        df_api['dia_semana'] = df_api['Fecha_Hora'].dt.day_name().map(dias_espanol)
        # Mes
        df_api['mes'] = df['Fecha_Hora'].dt.month
        # Es fin de semana binario (1 si es Sábado o Domingo, 0 en caso contrario)
        df_api['es_fin_de_semana'] = (df_api['dia_semana'] >= 5).astype(int)


        # --- Variables calculadas adicionales para una mejor precisión del modelo (más correlaciones que de donde aprender) -------

        # 1. Crear variables clave para mejorar la predicción

        # 2. Crear las nuevas variables de retraso (lags) y medias móviles
        # Nota: Usamos shifts para evitar usar datos futuros de forma accidental (data leakage)
        df_api['Consumo'] = df[target]

        df_api['active_power_lag_1'] = df[target].shift(1)
        # 3. Manejo de valores nulos (originados por el desfase temporal)
        df_api['active_power_lag_1'] = df_api['active_power_lag_1'].bfill()

        df_api['active_power_lag_2'] = df[target].shift(2)
        df_api['active_power_lag_2'] = df_api['active_power_lag_2'].bfill()

   
        df_api['active_power_lag_24'] = df[target].shift(24)
        df_api['active_power_lag_24'] = df_api['active_power_lag_24'].bfill()

        # Hace 48 y 72 horas
        df_api['active_power_lag_48'] = df[target].shift(48)
        df_api['active_power_lag_48'] = df_api['active_power_lag_48'].bfill()

        df_api['active_power_lag_72'] = df[target].shift(72)
        df_api['active_power_lag_72'] = df_api['active_power_lag_72'].bfill()

        # Misma hora de la semana anterior
        df_api['active_power_lag_168'] = df[target].shift(168)
        df_api['active_power_lag_168'] = df_api['active_power_lag_168'].bfill()

        # Media móvil de las últimas 3 horas (usando shift(1) para no incluir la hora actual en el cálculo)
        df_api['active_power_roll_mean_3'] = df[target].shift(1).rolling(window=3).mean()
        df_api['active_power_roll_mean_3'] = df_api['active_power_roll_mean_3'].bfill()

        df_api['rolling_mean_24'] = df[target].shift(1).rolling(24).mean()
        df_api['rolling_mean_24'] = df_api['rolling_mean_24'].bfill()

        df_api['rolling_mean_168'] = df[target].shift(1).rolling(168).mean()
        df_api['rolling_mean_168'] = df_api['rolling_mean_168'].bfill()



        # -----------------------------------------

        

        # 6. Unir los datos meteorológicos con tu Excel original haciendo match exacto por Fecha_Hora
        df_final = pd.merge(df, df_api, on='Fecha_Hora', how='left')

        # 7. Guardar el resultado en un nuevo archivo Excel
        # archivo_salida = 'excel_registro_preprocesado.xlsx'
        archivo_salida2 = 'excel_registro_preprocesadoAPI.xlsx'
        # df_final.to_excel(archivo_salida, index=False)
        df_api.to_excel(archivo_salida2, index=False)
        
        # print(f"¡Listo! Archivo horario guardado como: '{archivo_salida}'")
        print(f"¡Listo! Archivo horario guardado como: '{archivo_salida2}'")

        # model = load_model()

        df_modelo = df_api.drop(
            columns=['Fecha'],
            errors='ignore'
        )

        df_modelo = df_modelo.dropna().reset_index(drop=True)


        
        # prediccion = model.predict(df_modelo)

        # print(prediccion[0])


    else:
        print("Ocurrió un error al consultar la API:", data)

    return df_modelo