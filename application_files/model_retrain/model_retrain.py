import os
import pandas as pd
import shutil
import json
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from application_files.model_retraining.model_retraining import ModelTraining


class ModelReTrain:
    def __init__(self, par_path):
        self.par_path = par_path
        self.saved_model_path = os.path.join(par_path, 'saved_model/')
        self.model_retraining = ModelTraining(self.par_path)

    def model_retrain(self, log):
        """
            Method Name: model_retrain
            Description: This function retrain the model again on the predicted data to make it more generalized for
             the future predictions of new dataset
            Output: None
         """
        log_path = os.path.join(self.par_path, 'logger/')  # path to the logger folder
        log_file = open(log_path + 'model_reraining.txt', 'a+')
        try:
            # Initialize the file paths
            train_data_path = os.path.join(self.par_path, 'train_combined_csv/')
            pred_path = os.path.join(self.par_path, 'prediction/')

            #  Load the predicted output data file
            pred_out_data = pd.read_csv(pred_path + 'prediction_output.csv')
            pred_out_data.drop(['Wafer'], axis=1, inplace=True)

            #  Load the data on which model was trained
            train_data = pd.read_csv(train_data_path + 'train_processed_input.csv')

            #  Combine the previously trained data with newly predicted dataset so that we can retrain the models and
            #  better predict the future prediction datasets
            combined_data = pd.merge(left=train_data, right=pred_out_data, how="outer", on=train_data.columns.tolist())
            #  Save the combined dataset in csv format
            combined_data.to_csv(train_data_path + 'train_processed_input.csv', header = True, index = False)

            X = combined_data.drop(['label'], axis=1)
            y = combined_data['label']
            # Balance the count of classes in the dataset
            data = self.model_retraining.balancing_cluster_count(X, y, log, log_path)
            log.log(log_file, 'Balances the biased dataset using the SMOTE')

            #  Find the unique cluster numbers
            km_cluster = len(data['cluster'].unique())

            model_parameters_path = os.path.join(self.par_path, 'grid_search_cv', 'grid_search_cv_parameters.json')
            # Load the hyperparameter values obtained during the model training
            with open(model_parameters_path, 'r') as f:
                best_params = json.load(f)
                f.close()

            # Initialize the dictionary
            score = {}
            model_obj = {}
            models = {'RandomForestClassifier': RandomForestClassifier(),
                      'KNeighboursClassifier': KNeighborsClassifier(),
                      'LogisticRegression': LogisticRegression(),
                      'XGBClassifier': XGBClassifier()
                      }
            for c in range(0, km_cluster):
                score[c] = {}
                model_obj[c] = {}
                for i in models.keys():
                    score[c][i] = {}
                    model_obj[c][i] = {}

            #  Delete the existing saved model folder and create a new one so that newly trained model is available for
            #  the next prediction.
            model_path = os.path.join(self.par_path, 'saved_model')
            if os.path.isdir(model_path):
                shutil.rmtree(model_path)
                os.makedirs(model_path)

            # Retrain the models
            self.model_retraining.model_retraining(data, km_cluster, log, log_path, best_params, score, model_obj)
            log.log(log_file, 'Model retraining successful')  # write log file
            log_file.close()  # close the log file

        except Exception as e:
            # Write the log file for an error and close it
            log.log(log_file, 'Error during model retraining')
            log.log(log_file, str(e))
            log_file.close()


