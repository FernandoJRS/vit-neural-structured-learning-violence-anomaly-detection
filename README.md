# ViT Neural Structured Learning Violence Anomaly Detection

## Vision Transformer Featured Model For Training.

The model we use to make is a Vision Transformer (ViT). The pre-trained model that we use in the scripts in this repository can be downloaded from the following link:

  Link: https://tfhub.dev/sayakpaul/vit_s16_fe/1

For the correct run of the scripts, the ViT model must be downloaded and loaded using the relative path in the script to be run. The process is described in detail at the end of this document.

## Datasets

The datasets used in this project are NTU CCTV-Fights, UBI_Fights, XD-Violence, and UCF_Crime. <br/>

The paths of the datasets used are the following:

  NTU CCTV-Fights: https://rose1.ntu.edu.sg/dataset/cctvFights/ <br/>
  UBI_Fights: http://socia-lab.di.ubi.pt/EventDetection/ <br/>
  XD-Violence: https://roc-ng.github.io/XD-Violence/ <br/>
  UCF_Crime: https://www.crcv.ucf.edu/projects/real-world/ <br/>

## Requirements 

This section presents the requirements needed to run the scripts as well as an installation file requirements listed below.

  File: requirements.py <br/>
  
  !pip install --upgrade tensorflow_hub <br/>
  !pip install tensorflow==2.7.0 <br/>
  !pip install scikit-learn <br/>
  !pip install neural_structured_learning <br/>
  !pip install opencv-python==4.5.5.62 <br/>

## Files Scheme

The file scheme of this repository is associated by pairs and in turn each pair of scripts is associated with a data set. They all work with the same ViT model. The scheme is as follows:

1. DATA_CCTV.py - NSL_2D_CCTV.py - NTU CCTV-Fights.

- The DATA_CCTV file loads and processes the NTU CCTV-Fights dataset. The NSL_2D_CCTV file loads the ViT model, applies structured neural learning, and performs training and evaluation for the NTU CCTV-Fights dataset.
    
2. DATA_UBI.py - NSL_2D_UBI.py - UBI_Fights.

- The DATA_UBI file loads and processes the UBI_Fights dataset. The NSL_2D_UBI file loads the ViT model, applies structured neural learning, and performs training and evaluation for the UBI_Fights dataset.

3. DATA_XD.py - NSL_2D_XD.py - XD-Violence.

- The DATA_XD file loads and processes the XD-Violence dataset. The NSL_2D_XD file loads the ViT model, applies structured neural learning, and performs training and evaluation for the XD-Violence dataset.
    
4. DATA_UCF.py - NSL_2D_UCF.py - UCF_Crime.

- The DATA_UCF.py file loads and processes the UCF_Crime data set. The NSL_2D_UCF.py file loads the ViT model, applies structured neural learning, and performs training and evaluation for the UCF_Crime dataset.

## Run Scripts

To run the scripts correctly, follow these steps:

1. Download the ViT model through the link provided in this document.

2. Select and download the pairs of scripts that you want to run and download the corresponding dataset, example:

- If we want to run the NSL_2D_CCTV.py script, we must download that script and its associated DATA_CCTV.py script, in addition we must also download its corresponding NTU CCTV-Fights dataset through the link provided in this document. Similarly for the rest of the scripts (NSL_2D_CCTV.py, NSL_2D_UBI.py, NSL_2D_XD.py).

3. The third step is to configure the following parameters of the NSL_2D_DatasetName.py and DATA_DatasetName.py scripts:

- The first parameter to configure is the path of the ViT model, once downloaded its relative path must be indicated in the load_model variable of the NSL_2D_DatasetName script using the hub.load() function.

```
loaded_model = hub.load("HubModels/vit_b8_fe_1")
```

- The next parameters to configure is the path of the data set, once downloaded, a series of relative paths must be indicated in the DATA_DatasetName scripts.

  - For the DATA_CCTV.py script, the routes to modify depending on where the dataset has been downloaded are:
  
    - path_base = 'CCTV-Fights/'
    - path_videos = 'CCTV-Fights/Videos/'<br/><br/>
   
    ```
    path_base = 'CCTV-Fights/'
    path_videos = 'CCTV-Fights/Videos/'
    ```

  - For the DATA_UBI.py script, the paths to modify depending on where the dataset has been downloaded are:
  
    - path_base = 'UBI_Fights/annotation/'
    - path_videos = 'UBI_Fights/videos/'<br/><br/>
    
    ```
    path_base = 'UBI_Fights/annotation/'
    path_videos = 'UBI_Fights/videos/'
    ```

  - For the DATA_XD.py script, the paths to modify depending on where the dataset has been downloaded are:
  
    - path_base = 'XD-Violence/'
    - path_videos_train = 'XD-Violence/Train/'
    - path_videos_test = 'XD-Violence/Test/'
    - path_videos_annotations = 'XD-Violence/annotations.txt'<br/><br/>
    
    ```
    path_base = 'XD-Violence/'
    path_videos_train = 'XD-Violence/Train/'
    path_videos_test = 'XD-Violence/Test/'
    path_videos_annotations = 'XD-Violence/annotations.txt'
    ```

  - For the DATA_UCF.py script, the paths to modify depending on where the dataset has been downloaded are:
  
    - path_videos = 'UCF_Crimes/Videos/'
    - path_splits = 'UCF_Crimes/Action_Regnition_splits/'<br/><br/>

    ```
    path_videos = 'UCF_Crimes/Videos/'
    path_splits = 'UCF_Crimes/Action_Regnition_splits/'
    ```


- The last parameters to configure are the path of the checkpoints to store the pre-trained model and the path where a TensorBoard will be stored with the model training information and the .log files. These paths are modified in the NSL_2D_DatasetName.py scripts. The results will be stored in a directory called 'Results' that will be generated automatically within the path in which each of the data sets has been downloaded.

  - For the NTU CCTV-Fights dataset, in the NSL_2D_CCTV.py script, the paths are as follows:
  
    ```
    log_dir = "CCTV-Fights/Results/logs/fit/"
    checkpoint_path = "CCTV-Fights/Results/logs/checkpoint/"
    ```
    
  - For the UBI_Fights dataset, in the NSL_2D_UBI.py script, the paths are as follows:
  
    ```
    log_dir = "UBI_Fights/Results/logs/fit/"
    checkpoint_path = "UBI_Fights/Results/logs/checkpoint/"
    ```
    
  - For the XD-Violence dataset, in the NSL_2D_XD.py script, the paths are as follows:

    ```
    log_dir = "XD-Violence/Results/logs/fit/"
    checkpoint_path = "XD-Violence/Results/logs/checkpoint/"
    ```
    
  - For the UCF_Crime.py dataset, in the NSL_2D_UCF.py script, the paths are as follows:

    ```
    log_dir = "UCF_Crimes/Results/logs/fit/"
    checkpoint_path = "UCF_Crimes/Results/logs/checkpoint/"
    ```
    
4. The fourth and last step is to execute the process of training and evaluating the model with the selected data set, for each data set the following scripts must be executed.

- For the NTU CCTV-Fights dataset, the NSL_2D_CCTV.py script must be executed.
- For the UBI_Fights dataset, the NSL_2D_UBI.py script must be executed.
- For the XD-Violence dataset, run the NSL_2D_XD.py script.
- For the UCF_Crime dataset, run the NSL_2D_UCF.py script.

## Results

This section shows the results obtained for each of the data sets using two metrics, Sparse Categorical Accuracy and Categorical Accuracy.

The first table shows the results of an iteration for the partition between training, validation and testing that is predefined for each of the data sets. The UBI_Fights dataset does not have a predefined partition, so in the DATA_UBI.py file, we perform the partition in training, validation and test.


|     Dataset     |     Input    | Sparse Categorical Accuracy | Categorical Accuracy | Inference Time |
|-----------------|--------------|-----------------------------|----------------------|----------------|
| NTU CCTV-Fights | Optical Flow |         100.00 %	           |        100.00 %      |    0,040258 s  |
| UBI_Fights      | Optical Flow |         100.00 %	           |        100.00 %      |    0,042453 s  |
| XD-Violence     | Optical Flow |         99.98 %             |        99.98 %       |    0,040882 s  |
| UCF_Crime	      | Optical Flow |         99.99 %	           |        99.99 %       |    0,040956 s  |

El conjunto de datos UCF_Crime viene con cuatro particiones predefinidas en la segunda tabla se muestran los resultados para cada partición. En la primera tabla los resultados corresponden a la partición Train_001/Test_001.


| UCF_Crime	Iteration | Train Split | Test Split | Sparse Categorical Accuracy | Categorical Accuracy | Inference Time |
|---------------------|-------------|------------|-----------------------------|----------------------|----------------|
| 01                  | Train_001   | Test_001   | 99.99 %                     | 99.99 %              |    0,040956 s  |
| 02                  | Train_002   | Test_002   | 99.99 %                     | 99.99 %              |    0,041358 s  |  
| 03                  | Train_003   | Test_003   | 99.99 %                     | 99.99 %              |    0,041226 s  |  
| 04	                | Train_004   | Test_004   | 99.99 %                     | 99.99 %              |    0,040693 s  |


The following table shows the confusion matrices for each of the datasets.


CCTV Confusion Matrix      | UBI Confusion Matrix      | XD Confusion Matrix       | UCF Confusion Matrix
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![CCTV_CM](figures/CCTV_CM.png?raw=True "CCTV_CM") | ![UBI_CM](figures/UBI_CM.png?raw=True "UBI_CM") | ![XD_CM](figures/XD_CM.png?raw=True "XD_CM") | ![UCF_CM](figures/UCF_CM.png?raw=True "UCF_CM")

The following table shows the confusion matrices for a cross experimentation between the CCTV-Fights and UBI-Fights single-class datasets in which the pre-trained models with each of the datasets are used to test the other.

CCTV (Train) - UBI (Test) Confusion Matrix  | UBI (Train) - CCTV (Test) Confusion Matrix      |
:------------------------------------------:|:-----------------------------------------------:|
![CCTV_UBI_CM](figures/CCTV_UBI_CM.png?raw=True "CCTV_UBI_CM") | ![UBI_CCTV_CM](figures/UBI_CCTV_CM.png?raw=True "UBI_CCTV_CM")

The following table shows the confusion matrices for cross-experimentation between the UCF-Crime and XD-Violence multiclass datasets in which the trained models are used only with the matching classes in both datasets.

UCF (Train) - XD (Test) Confusion Matrix  | XD (Train) - UCF (Test) Confusion Matrix      |
:------------------------------------------:|:-----------------------------------------------:|
![UCF_XD_CMM](figures/UCF_XD_CMM.png?raw=True "UCF_XD_CMM") | ![XD_UCF_CMM](figures/XD_UCF_CMM.png?raw=True "XD_UCF_CMM")

## Ablation Study

In this section, an ablation study is presented in which ViT is used without applying structured neural learning, simply by applying supervised learning. The results obtained are shown in the following table:

|     Dataset     |     Input    | Sparse Categorical Accuracy | Categorical Accuracy | Inference Time |
|-----------------|--------------|-----------------------------|----------------------|----------------|
| NTU CCTV-Fights | Optical Flow |         90.45 % 	           |        90.45%        |    0,040374 s  |
| UBI_Fights      | Optical Flow |         89.76 %	           |        89.76 %       |    0,041883 s  |
| XD-Violence     | Optical Flow |         87.23 %             |        87.23 %       |    0,041159 s  |
| UCF_Crime	      | Optical Flow |         87.52 %	           |        87.52 %       |    0,040981 s  |

## Performance

An example of how the model works on several videos can be found in the following links.

Link 1: https://www.dropbox.com/s/b0ila0rlr5f9a2k/fight0005.avi?dl=0 <br/>
Link 2: https://www.dropbox.com/s/sfit3lrco9rbwe2/fight0033.avi?dl=0 <br/>
Link 3: https://www.dropbox.com/s/hfo63facdg4kbnu/fight0128.avi?dl=0

