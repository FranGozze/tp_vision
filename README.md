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

Donde ademas tendremos las siguientes opciones a la hora de correrlo:

- `--ransac`: Se utilira el metodo RANSAC para filtrar matches espureos
- `--gt`: Se utilizaran las poses dadas por el ground-truth en el mapa denso generado por el disparity map
- `--calibration-dir <DIR>`: Los archivos de calibracion se encuentran en el directorio `<DIR>`
- `--bag-dir <DIR>`: La bag que se debe reproducir se encuentra en el directorio `<DIR>`
- `--ground-truth <FILE>` El archivo ground-truth correspondiente a la bag que se proveyo en el directorio
- `--calibr-file <FILE>` El achivo de calibracion que contiene las matrices necesarias para cambiar de IMU a la camara izquierda

# Decisiones tomadas

A lo largo del proyecto se tomaron ciertas deciciones que afectaron su comportamiento y aqui explicaremos cuales fueron y sus motivos:

- Primero cabe destacar que el proyecto esta pensado para que tome los datos de una bag y se publiquen. Al mismo tiempo, habra otros nodos que procesen las imagenes resultantes en paralelo.
- Para poder identificar cuando 2 imagenes corresponden al mismo frame, se decidio usar el stamp como determinante, donde si 2 o mas imagenes tienen el mismo stamp, entonces dichas imagenes se relacionan entre si. Esto es util para poder trabajar las imagenes por separado
- La mayor parte de las comunicaciones entre nodos se dan a traves de publicaciones, no obstante algunos nodos requieren que otros les pasen informacion a traves del codigo. Esto se hizo asi para no perder atributos dados por las librerias en python.
- Asumimos que el ground truth que se lee de un csv esta dado en coordenadas de la imu, por lo que a cada pose se la pasara al sistema de coordenadas de la camara izquierda.
- Debido a que la cantidad de poses que hay en el ground truth es muy similar a la cantidad de imagenes tomadas en el recorrido que se uso para probar el proyecto, se decidio que cada imagen se correspondera con una pose.
- En el nodo que se encarga del mapeo denso persistente, se decidio que no se usaran todos los frames, y de los que si se utilizaran se tomaran una cantidad delimitada debido al gran volumen de puntos que se generaba, lo que tenia como consecuencia que RViz funcionara incorrectamente y fuera imposible navegar en el mismo.
