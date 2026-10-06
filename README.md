# API_REST_FAST_API_PROYECTO_GRADO_OJ
Este es el repositorio principal del Back-End del proyecto de grado.
<<<<<<< HEAD



#Documentación de la estructura

1. Main.py llama a las rutas, actualmente el archivo de rutas "prediction.py" tiene una sola ruta "/" y se encarga de recibir un archivo excel.
2. La ruta "/" recibe el excel y lo envía al servicio "prediction services" y aquí se crea un dataframe con pandas para luego ser envíado  a preprocessing.py para hacer los ajustes de creación de la columna de valores ambientales y dia de la semana.
3. Luego este dataframe finalizado se le envía al modelo por medio del archivo "model.py" y se esperan los valores de la predicción.
=======
>>>>>>> 1717858a46a84533b3fac4662a72ede48b45131b
