# ViT Neural Structured Learning Violence Anomaly Detection

## Vision Transformer Featured Model For Training.

  Link: https://www.dropbox.com/sh/yt9xdtwqymrwd7i/AAAx5s7FDSX1V4EiPCxMVJpRa?dl=0


## Datasets

Los datasets utilizados en este proyecto son NTU CCTV-Fights, UBI_Fights, XD-Violence y UCF_Crime. <br/>

Las rutas de los datasets usados son las siguientes:

  NTU CCTV-Fights: https://rose1.ntu.edu.sg/dataset/cctvFights/ <br/>
  UBI_Fights: http://socia-lab.di.ubi.pt/EventDetection/ <br/>
  XD-Violence: https://roc-ng.github.io/XD-Violence/ <br/>
  UCF_Crime: https://www.crcv.ucf.edu/projects/real-world/ <br/>

## Requirements 
  
  File: requirements.py <br/>
  
  pip install tensorflow==2.7.0 <br/>
  pip install scikit-learn <br/>
  pip install neural_structured_learning <br/>
  pip install opencv-python==4.5.5.62 <br/>

## Files Scheme

El esquema de archivos de este repositorio va asociado por pares y a su vez cada par de scripts asociado a un conjunto de de datos. Todos ellos trabajan con el mismo modelo ViT. El esquema es el siguiente:

1º DATA_CCTV.py - NSL_2D_CCTV - NTU CCTV-Fights.
  
    El archivo DATA_CCTV carga y procesa el conjunto de datos NTU CCTV-Fights. El archivo NSL_2D_CCTV carga el modelo ViT, aplica el aprendizaje neuronal estructurado y realiza     el entrenamiento y la evaluación para el conjunto de datos NTU CCTV-Fights.
    
2º DATA_UBI.py - NSL_2D_CCTV - NTU CCTV-Fights.
  
    El archivo DATA_CCTV carga y procesa el conjunto de datos NTU CCTV-Fights. El archivo NSL_2D_CCTV carga el modelo ViT, aplica el aprendizaje neuronal estructurado y realiza     el entrenamiento y la evaluación para el conjunto de datos NTU CCTV-Fights.

3º DATA_XD.py - NSL_2D_CCTV - NTU CCTV-Fights.
  
    El archivo DATA_CCTV carga y procesa el conjunto de datos NTU CCTV-Fights. El archivo NSL_2D_CCTV carga el modelo ViT, aplica el aprendizaje neuronal estructurado y realiza     el entrenamiento y la evaluación para el conjunto de datos NTU CCTV-Fights.
    
4º DATA_UCF.py - NSL_2D_UCF - UCF_Crime.
  
    El archivo DATA_UCF carga y procesa el conjunto de datos UCF_Crime. El archivo NSL_2D_UCF carga el modelo ViT, aplica el aprendizaje neuronal estructurado y realiza     el       entrenamiento y la evaluación para el conjunto de datos UCF_Crime.


