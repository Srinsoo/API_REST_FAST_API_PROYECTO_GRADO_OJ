import pandas as pd
import requests

def preprocess_data(df):

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
    lat = 9.3045
    lon = -75.3905

    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation"
    )

    print(f"Consultando clima por hora para Sincelejo ({start_date} al {end_date})...")
    response = requests.get(url)
    data = response.json()

    if response.status_code == 200 and 'hourly' in data:
    # 4. Organizar la respuesta horaria de la API en una tabla temporal
        hourly_data = data['hourly']
        df_api = pd.DataFrame({
            'Fecha_Hora': pd.to_datetime(hourly_data['time']), # La API devuelve la hora exacta
            'Temperatura (°C)': hourly_data['temperature_2m'],
            'Humedad Relativa (%)': hourly_data['relative_humidity_2m'],
            'Sensación Térmica (°C)': hourly_data['apparent_temperature'],
            'Precipitación (mm)': hourly_data['precipitation']
        })

        # 5. Crear la columna del Día de la Semana en español basada en la fecha/hora
        dias_espanol = {
            'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miércoles',
            'Thursday': 'Jueves', 'Friday': 'Viernes', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
        }
        df_api['Día de la Semana'] = df_api['Fecha_Hora'].dt.day_name().map(dias_espanol)

        # 6. Unir los datos meteorológicos con tu Excel original haciendo match exacto por Fecha_Hora
        df_final = pd.merge(df, df_api, on='Fecha_Hora', how='left')

        # 7. Guardar el resultado en un nuevo archivo Excel
        archivo_salida = 'excel_registro_preprocesado.xlsx'
        df_final.to_excel(archivo_salida, index=False)
    
        print(f"¡Listo! Archivo horario guardado como: '{archivo_salida}'")

    else:
        print("Ocurrió un error al consultar la API:", data)

    return df