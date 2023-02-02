# Automated Defective Substrate Identification for Expedited Manufacturing

# Aim of the project:
In electronics, a wafer (also called a slice or substrate) is a thin slice of semiconductor, such as crystalline silicon (c-Si), used for the fabrication of integrated circuits and, in photovoltaics, to manufacture solar cells. The wafer serves as the substrate for microelectronic devices built in and upon the wafer. The project aims to successfully identify the state of the provided wafer by classifying it between one of the two-class +1 (good, can be used as a substrate) or -1 (bad, the substrate need to be replaced) and then train the model on this data so that it can continuously update itself with the environment and become more generalized with time. In this regard, a training and prediction dataset is provided to build a machine learning classification model, which can predict the wafer quality.

# Data Description:
The columns of provided data can be classified into 3 parts: wafer name, sensor values and label. The wafer name contains the batch number of the wafer, whereas the sensor values obtained from the measurement carried out on the wafer. The label column contains two unique values +1 and -1 that identifies if the wafer is good or need to be replaced. Additionally, we also require a schema file, which contains all the relevant information about the training files such as file names, length of date value in the file name, length of time value in the file name, number of columns, name of the columns, and their datatype.

# Directory creation:
All the necessary folders were created to effectively separate the files so that the end-user can get **easy access** to them.
# Data Validation:
In this step, we matched our dataset with the provided schema file to match the file names, the number of columns it should contain, their names as well as their datatype. If the files matched with the schema values then they are considered good files on which we can train or predict our model, however, if it didn't match then they are moved to the bad folder. Moreover, we also identify the columns with null values. If all the data in a column is missing then the file is also moved to the bad folder. On the contrary, if only a fraction of data in a column is missing then we initially fill it with NaN and considered it as good data.
# Data Insertion in Database:
First, open a connection to the database if it exists otherwise create a new one. A table with the name **train_good_raw_dt** or **pred_good_raw_dt** is created in the database, based on the training or prediction process, for inserting the good data files obtained from the data validation step. If the table is already present then new files are inserted in that table as we want training to be done on new as well as old training files. In the end, the data in a stored database is exported as a CSV file, to be used for the model training.
# Data Pre-processing and Model Training:
In the training section, first, the data is checked for the NaN values in the columns. If present, impute the NaN values using the **KNN imputer**. The columns with zero standard deviation were also identified and removed as they don't give any information during model training. A prediction schema was created based on the remained dataset columns. Afterwards, the KMeans algorithm is used to create clusters in the pre-processed data. The optimum number of clusters is selected by plotting the elbow plot, and for the dynamic selection of the number of clusters, we are using the "KneeLocator" function. The idea behind clustering is to implement different algorithms to train data in different clusters. The Kmeans model is trained over pre-processed data and the model is saved for further use in prediction. After clusters are created, we find the best model for each cluster. We are using four algorithms, **Random Forest**, **K-Neighbours**, **Logistic Regression** and **XGBoost**. For each cluster, both the algorithms are passed with the best parameters derived from **GridSearch**. We calculate the AUC scores for all the models and select the one with the best score. Similarly, the best model is selected for each cluster. For every cluster, the models are saved so that they can be used in future predictions. In the end, the confusion matrix of the model associated with every cluster is also saved to give the 
a glance over the performance of the models.
# Prediction: 
In data prediction, first, the essential directories are created. The data validation, data insertion and data processing steps are similar to the training section. The KMeans model created during training is loaded, and clusters for the pre-processed prediction data is predicted. Based on the cluster number, the respective model is loaded and is used to predict the data for that cluster. Once the prediction is made for all the clusters, the predictions along with the Wafer names are saved in a CSV file at a given location.
# Retraining:
After the prediction, the prediction data is merged with the previous training dataset and then the models were retrained on this data using the hyperparameter values obtained from the GridSearch. The cycle repeats with every prediction it does and learns from the newly acquired data, making it **more robust**.
# Deployment:
We will be deploying the model to **Heroku Cloud**.

# Project execution:

**Command line approach** <br />

**Step 1:** First create an environment and install the dependencies listed in the **requirements.txt** file. <br />

>(base) Project_folder>**conda create -n "environment_name" python=3.8** # Create an environment <br />
>(base) Project_folder>**conda activate "environment_name"** # Activate the created environment <br />
>(environment) Project_folder>**pip install -r requirements.txt** # Install all the dependencies <br />

**Step 2:** For model training, use the choose.py file.  <br />

>(environment) Project_folder>**python choose.py Train** # To train the models on default files  (provided in the project) <br />

>(environment) Project_folder>**python choose.py Train "User_training_files_location"** # To train the models on the files provided by the user <br />

**Step 3:** To predict the condition of a substrate, and retrain the models on a combined dataset of previously trained and this prediction dataset, run the following commands: <br />

>(environment) Project_folder>**python choose.py Predict** # To predict for the files provided in the project <br />

>(environment) Project_folder>**python choose.py Predict "User_prediction_files_location"** # To predict for the files provided by the user <br />
> (environment) Project_folder> Model retraining was successful. <br />
> (environment) Project_folder> After reviewing the performance please delete the prediction_output.csv file and confusion_matrix folder files in the end. <br />
> (environment) Project_folder> Prediction successful. <br />
> (environment) Project_folder> Check the folder  project_path/prediction/prediction_output.csv

# Alternatively:

**For an web interface and easy visualization:**  <br />

**Step 1:** First create an environment and install the dependencies listed in the **requirements.txt** file.  <br />

>(base) Project_folder>**conda create -n "environment_name" python=3.8** # Create an environment <br />
>(base) Project_folder>**conda activate "environment_name"** # Activate the created environment <br />
>(environment) Project_folder>**pip install -r requirements.txt** # Install all the dependencies

**Step 2:** To train a model run the command: <br />

>(environment) Project_folder>**python train_model_main.py** # It will provide an http:// link, where our program is running. Open the link in the webpage and proceed further. <br />
![image](https://user-images.githubusercontent.com/93785299/142681694-082f699f-0a58-4f5b-9670-168921fa065b.png)


**Step 3:** To predict the condition of a substrate, and retrain the models on a combined dataset of previously trained and this prediction dataset, run the following commands: <br />

>(environment) Project_folder>**python main.py** # It will provide an http:// link, where our program is running. Open the link in the webpage and proceed further. <br />
![image](https://user-images.githubusercontent.com/93785299/142681206-9f5a0da3-90dd-47c4-a5aa-4238ee61ccfa.png)

