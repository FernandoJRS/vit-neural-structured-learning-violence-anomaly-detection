# ViT Neural Structured Learning Violence Anomaly Detection

## Vision Transformer Featured Model For Training.

El modelo que usamos para realizar es un Vision Transformer (ViT) el modelo preentrenado que usamos en los scripts de este repositorio puede descargarse desde el siguiente enlace:

  Link: https://www.dropbox.com/sh/yt9xdtwqymrwd7i/AAAx5s7FDSX1V4EiPCxMVJpRa?dl=0

Para la correcta ejecución de los scripts, debe descargarse el modelo ViT y cargarlo mediante la ruta relativa en el script que se desea ejecutar. El proceso se describe en detalle al final de este documento.

## Datasets

Los datasets utilizados en este proyecto son NTU CCTV-Fights, UBI_Fights, XD-Violence y UCF_Crime. <br/>

Las rutas de los datasets usados son las siguientes:

  NTU CCTV-Fights: https://rose1.ntu.edu.sg/dataset/cctvFights/ <br/>
  UBI_Fights: http://socia-lab.di.ubi.pt/EventDetection/ <br/>
  XD-Violence: https://roc-ng.github.io/XD-Violence/ <br/>
  UCF_Crime: https://www.crcv.ucf.edu/projects/real-world/ <br/>

## Requirements 

En esta sección se presentan los requisitos necesarios para ejecutar los scripts asi como un archivo de instalación de requisitos que se indica a continuación.
  
  File: requirements.py <br/>
  
  pip install tensorflow==2.7.0 <br/>
  pip install scikit-learn <br/>
  pip install neural_structured_learning <br/>
  pip install opencv-python==4.5.5.62 <br/>

## Files Scheme

El esquema de archivos de este repositorio va asociado por pares y a su vez cada par de scripts asociado a un conjunto de de datos. Todos ellos trabajan con el mismo modelo ViT. El esquema es el siguiente:

1. DATA_CCTV.py - NSL_2D_CCTV.py - NTU CCTV-Fights.

- El archivo DATA_CCTV carga y procesa el conjunto de datos NTU CCTV-Fights. El archivo NSL_2D_CCTV carga el modelo ViT, aplica el aprendizaje neuronal estructurado y realiza el entrenamiento y la evaluación para el conjunto de datos NTU CCTV-Fights.
    
2. DATA_UBI.py - NSL_2D_UBI.py - UBI_Fights.

- El archivo DATA_UBI carga y procesa el conjunto de datos UBI_Fights. El archivo NSL_2D_UBI carga el modelo ViT, aplica el aprendizaje neuronal estructurado y realiza el entrenamiento y la evaluación para el conjunto de datos UBI_Fights.

3. DATA_XD.py - NSL_2D_XD.py - XD-Violence.

- El archivo DATA_XD carga y procesa el conjunto de datos XD-Violence. El archivo NSL_2D_XD carga el modelo ViT, aplica el aprendizaje neuronal estructurado y realiza el    entrenamiento y la evaluación para el conjunto de datos XD-Violence.
    
4. DATA_UCF.py - NSL_2D_UCF.py - UCF_Crime.

- El archivo DATA_UCF carga y procesa el conjunto de datos UCF_Crime. El archivo NSL_2D_UCF carga el modelo ViT, aplica el aprendizaje neuronal estructurado y realiza el entrenamiento y la evaluación para el conjunto de datos UCF_Crime.

## Run Scripts

Para ejecutar los scripts correctamente hay que seguir los siguientes pasos:

1. Descargar el modelo ViT a través del enlace que se proporciona en este documento.

2. Seleccionar y descargar los pares de scripts que se quieren ejecutar y descargar el dataset correspondiente, ejemplo:

- Si queremos ejecutar el script NSL_2D_CCTV, debemos descargar dicho script y su asociado DATA_CCTV, además también debemos descargar su conjunto de datos correspondiente UCF_Crime a través del enlace que se proporciona en este documento. De igual forma para el resto de scripts (NSL_2D_CCTV, NSL_2D_UBI, NSL_2D_XD).

3. El tercer paso es configurar los siguientes parametros de los scripts NSL_2D y DATA_:

- El primer parámetro a configurar es la ruta del modelo ViT, una vez descargado se debe indicar su ruta relativa en la variable load_model del script NSL_2D usando la función hub.load().

```
loaded_model = hub.load("HubModels/vit_s16_fe_1")
```

- Los siguientes parámetros a configurar es la ruta del conjunto de datos, una vez descargado se debe indicar una serie de rutas relativas en los scripts DATA_.

  - Para el script DATA_CCTV las rutas a modificar en función de donde se haya descargado el conjunto de datos son:
  
    - path_base = 'CCTV-Fights/'
    - path_videos = 'CCTV-Fights/Videos/'<br/>
   
    ```
    path_base = 'CCTV-Fights/'
    path_videos = 'CCTV-Fights/Videos/'
    ```

  - Para el script DATA_UBI las rutas a modificar en función de donde se haya descargado el conjunto de datos son:
  
    - path_base = 'UBI_Fights/annotation/'
    - path_videos = 'UBI_Fights/videos/'<br/><br/>
    
    ```
    path_base = 'UBI_Fights/annotation/'
    path_videos = 'UBI_Fights/videos/'
    ```

  - Para el script DATA_XD las rutas a modificar en función de donde se haya descargado el conjunto de datos son:
  
    - path_base = 'XD-Violence/'
    - path_videos_train = 'XD-Violence/Train/'
    - path_videos_test = 'XD-Violence/Test/'<br/><br/>
    
    ```
    path_base = 'XD-Violence/'
    path_videos_train = 'XD-Violence/Train/'
    path_videos_test = 'XD-Violence/Test/'<br/><br/>
    ```

  - Para el script DATA_UCF las rutas a modificar en función de donde se haya descargado el conjunto de datos son:
  
    - path_videos = 'UCF_Crimes/Videos/'
    - path_splits = 'UCF_Crimes/Action_Regnition_splits/'<br/><br/>

    ```
    path_videos = 'UCF_Crimes/Videos/'
    path_splits = 'UCF_Crimes/Action_Regnition_splits/'
    ```

- Los últimos parámetros a configurar son la ruta de los checkpoints para almacenar el modelo pre-entrenado y la ruta donde se almacenará un TensorBoard con la información del entrenamiento del modelo y los archivos .log. Los resultados se almacenará dentro de un directorio llamado 'Results' que se generara automaticamente dentro de la ruta en la cual se haya descargado cada uno de los conjuntos de datos.

  - Para el conjunto de datos CCTV-Fights las rutas son las siguientes:
  
    ```
    log_dir = "CCTV-Fights/Results/logs/fit/"
    checkpoint_path = "CCTV-Fights/Results/logs/checkpoint/"
    ```
    
  - Para el conjunto de datos UBI_Fights las rutas son las siguientes:
  
    ```
    log_dir = "UBI_Fights/Results/logs/fit/"
    checkpoint_path = "UBI_Fights/Results/logs/checkpoint/"
    ```
    
  - Para el conjunto de datos XD-Violence las rutas son las siguientes:

    ```
    log_dir = "XD-Violence/Results/logs/fit/"
    checkpoint_path = "XD-Violence/Results/logs/checkpoint/"
    ```
    
  - Para el conjunto de datos UCF_Crimes las rutas son las siguientes:

    ```
    log_dir = "UCF_Crimes/Results/logs/fit/"
    checkpoint_path = "UCF_Crimes/Results/logs/checkpoint/"
    ```
