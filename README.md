# Trabajo Practico 3 Visión por Computadora: Reconstrucción 3D y Estimación de Pose
FACULTAD DE CS. EXACTAS, INGENIERÍA Y AGRIMENSURA
LICENCIATURA EN CIENCIAS DE LA COMPUTACIÓN
ROBÓTICA MÓVIL

Autores: 
Franco Gozzerino
Jordi Solá

# Requerimientos
Tener instalado ROS2 y RViz2

Para instalar las librerias necesarias para correr este proyecto uno puede instalar manualmente las librerias listadas en `requirements.txt` o puede generar un virtual enviroment de la siguiente forma:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Y una vez hecho se podra correr

# Uso
Para utilizar este proyecto se debera hacer:
```
python3 recAndEstScript.py
```
Donde ademas tendremos las siguientes opciones a la hora de correrlo
    `--ransac`: Se utilira el metodo RANSAC para filtrar matches espureos
    `--gt`: Se utilizaran las poses dadas por el ground-truth en el mapa denso generado por el disparity map
    `--calibration-dir <DIR>`: Los archivos de calibracion se encuentran en el directorio `<DIR>`
    `--bag-dir <DIR>`: La bag que se debe reproducir se encuentra en el directorio `<DIR>`