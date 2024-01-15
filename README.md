# RAPTA
RAPTA: A Hierarchical Representation Learning Architecture to Predict Path-Based Timing Slacks at Design Time. This repository presents RAPTA, a customized Representation-learning Architecture for automation of feature engineering and predicting the result of Path-based Timing-Analysis early in the physical design cycle. To the best of our knowledge, this is the first work, in which Bidirectional Long Short-Term Memory (Bi-LSTM) representation learning is used to digest raw information for feature engineering, where generation of latent features and Multilayer Perceptron (MLP) based regression for timing prediction can be trained end-to-end.

## Links
1. Paper: [Chowdhury, Tanmoy, Ashkan Vakil, Banafsheh Saber Latibari, Seyed Aresh Beheshti Shirazi, Ali Mirzaeian, Xiaojie Guo, Sai Manoj PD et al. "RAPTA: A hierarchical representation learning solution for real-time prediction of path-based static timing analysis." In Proceedings of the Great Lakes Symposium on VLSI 2022, pp. 493-500. 2022](https://dl.acm.org/doi/pdf/10.1145/3526241.3530831)
2. Presentation [Link](https://dl.acm.org/doi/10.1145/3526241.3530831)

## Instructions:
There are several ways to run the code. 

### Google Colab: 
1. In "RAPTA" folder, there is folder named [Google_CoLab](/Google_CoLab/) which has the code to run on Google Colab.
2. Open [Google Colab](https://colab.research.google.com/).
3. File -> Upload notebook
4. Choose the file [RAPTA\Google_CoLab\RAPTA_DEMO.ipynb](/Google_CoLab/RAPTA_DEMO.ipynb).
5. In the first cell it will ask to upload the necessary files. Then upload the following from [Google_CoLab](/Google_CoLab/) folder:

    a. configure.py
    
    b. data.zip
    
    c. data_collector.py
    
    d. model.py
    
    e. predict_from_chkpnt.py
    
    f. run_model.py
    
    g. saved_model.zip
    
    h. utility.py

6. After uploading all the files just run all the cells. It will show the performance at the end.

### Local Machine:
1. Preparing Interpreter: Make sure all the libraries mentioned in [requirements.txt](requirements.txt) have been installed.
2. Running Code: Go to [main](/main/)

    a. Run "data_collecotor.py" for processing raw data

    b. Run "run_model.py" for training

    c. Run "predict_from_chkpnt.py" for testing

    d. Run "utility.py" for performance metrics

### Code explanation:
1. This code was developed as a one-stop solution, so that it can be used for raw data processing, running for single data as well as multiple data. To experience these diversified options, we only need to play with "configure.py".
2. If anyone wants to understand the model architecture only then please follow the file 'model.py.

### Data: 
Here is a [sample data](./data/model_purpose/PT_S38417) for PT_S38417 for 0.78V. Please check the [data.zip](/Google_CoLab/data.zip) and [RAPTA_Demo](/Google_CoLab/RAPTA_DEMO.ipynb) to understand the data processing.

## Citation
If you use this work, please cite the following paper.

"Chowdhury, Tanmoy, Ashkan Vakil, Banafsheh Saber Latibari, Seyed Aresh Beheshti Shirazi, Ali Mirzaeian, Xiaojie Guo, Sai Manoj PD et al. "RAPTA: A hierarchical representation learning solution for real-time prediction of path-based static timing analysis." In Proceedings of the Great Lakes Symposium on VLSI 2022, pp. 493-500. 2022."
