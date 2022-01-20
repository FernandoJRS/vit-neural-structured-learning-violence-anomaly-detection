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

1º DATA_CCTV.py - NSL_2D_CCTV - NTU CCTV-Fights.
  
- El archivo DATA_CCTV carga y procesa el conjunto de datos NTU CCTV-Fights. El archivo NSL_2D_CCTV carga el modelo ViT, aplica el aprendizaje neuronal estructurado y realiza     el entrenamiento y la evaluación para el conjunto de datos NTU CCTV-Fights.
    
2º DATA_UBI.py - NSL_2D_UBI - UBI_Fights.

- El archivo DATA_UBI carga y procesa el conjunto de datos UBI_Fights. El archivo NSL_2D_UBI carga el modelo ViT, aplica el aprendizaje neuronal estructurado y realiza             el entrenamiento y la evaluación para el conjunto de datos UBI_Fights.

3º DATA_XD.py - NSL_2D_XD - XD-Violence.
  
- El archivo DATA_XD carga y procesa el conjunto de datos XD-Violence. El archivo NSL_2D_XD carga el modelo ViT, aplica el aprendizaje neuronal estructurado y realiza el    entrenamiento y la evaluación para el conjunto de datos XD-Violence.
    
4º DATA_UCF.py - NSL_2D_UCF - UCF_Crime.
  
- El archivo DATA_UCF carga y procesa el conjunto de datos UCF_Crime. El archivo NSL_2D_UCF carga el modelo ViT, aplica el aprendizaje neuronal estructurado y realiza el entrenamiento y la evaluación para el conjunto de datos UCF_Crime.

## Run Scripts

Para ejecutar los scripts correctamente hay que seguir los siguientes pasos:

1º Descargar el modelo ViT a través del enlace que se proporciona en este documento.

2º Seleccionar los pares de scripts se quiere ejecutar y descargar el dataset correspondiente, ejemplo:

- Si queremos ejecutar el script NSL_2D_CCTV, debemos descargar dicho script y su asociado DATA_CCTV, además también debemos descargar su conjunto de datos correspondiente UCF_Crime a través del enlace que se proporciona en este documento. De igual forma para el resto de scripts (NSL_2D_CCTV, NSL_2D_UBI, NSL_2D_XD).

3º El tercer paso es configurar los siguientes parametros de los scripts NSL_2D:

- El primer parámetro a configurar es la ruta del modelo ViT, una vez descargado se debe indicar su ruta relativa en la variable load_model en la función hub.load(). <br/>
  loaded_model = hub.load("HubModels/vit_s16_fe_1")
