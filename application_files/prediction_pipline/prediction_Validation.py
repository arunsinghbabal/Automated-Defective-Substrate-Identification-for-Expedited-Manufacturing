import os
from application_files.log.logger import Logger
from application_files.create_directory.create_directory import DirectoryCreator
from application_files.prediction_data_validation.pred_data_validation import PredictionDataValidation
from application_files.prediction_sql.prediction_sql_db import PredictionSqlDb
from application_files.prediction_preprocessing.predicting_preprocessing import PredictionPreprocessing
from application_files.prediction_from_model.predictfrommodel import PredictFromModel
from application_files.model_file_operation.file_operation import FileOperationModelFinder
from application_files.model_retrain.model_retrain import ModelReTrain


class PredictionValidation:
    def __init__(self, par_path, pred_path):
        self.pred_path = pred_path
        self.par_path = par_path
        self.log = Logger()
        self.directory_creator = DirectoryCreator(self.par_path)
        self.pred_data_validation = PredictionDataValidation(self.par_path)
        self.pred_sql_db = PredictionSqlDb(self.par_path)
        self.pred_preprocessing = PredictionPreprocessing(self.par_path)
        self.model_pred = PredictFromModel(self.par_path)
        self.file_operation = FileOperationModelFinder()
        self.modelretrain = ModelReTrain(self.par_path)

    def pred_validation(self):
        """
            Method Name: pred_validation
            Description: This function validates and process the data and then train the models on it by calling
            the different functions
            Output: None
        """
        pred_log_path = os.path.join(self.par_path, 'logger', 'prediction/')  # path to the logger folder
        schema_path = os.path.join(self.par_path, 'schema.json')  # path to the schema file
        # Create necessary folders
        self.directory_creator.directory_creator(self.log, train=False)
        pred_log_file = open(pred_log_path + 'pred_validation.txt', 'a+')  # open the log file
        self.log.log(pred_log_file, 'Prediction directories created successfully')  # write the log file
        try:
            # Obtain the necessary parameters from the schema file
            sch_filename, sch_nos_col, sch_col_name, sch_len_date_stamp, sch_len_time_stamp = self.pred_data_validation.pred_from_schema_file(
                schema_path, self.log, pred_log_path)
            self.log.log(pred_log_file, 'Parameters obtained successfully from schema file')
            print('Parameters obtained successfully from schema file')

            # Validates the given raw data according to the schema parameters
            self.pred_data_validation.pred_data_validation(self.pred_path, sch_nos_col, sch_col_name, sch_len_date_stamp,
                                                           sch_len_time_stamp,  self.log, pred_log_path)
            self.log.log(pred_log_file, 'Prediction raw files successfully validated')
            print('Prediction raw files successfully validated')

            # Create a SQL databases, create a table with good input records and then save it in a file Input.csv 
            self.pred_sql_db.pred_database(sch_col_name, self.log, pred_log_path)
            self.log.log(pred_log_file, 'Prediction SQL database created successfully')
            print('Prediction SQL database created successfully')

            # Fill the missing values in the prediction data using KNNImputer, which uses n-neighbor algorithm
            X_imputed, wafer_names = self.pred_preprocessing.pred_data_impute(self.log, pred_log_path)
            self.log.log(pred_log_file, 'Prediction data imputed successfully')
            print('Prediction data imputed successfully')

            # Remove columns with zero variance
            X_processed = self.pred_preprocessing.pred_data_variance(X_imputed, self.log, pred_log_path)
            self.log.log(pred_log_file, 'Successfully removed columns with zero variance')
            print('Successfully removed columns with zero variance')

            X_checked = self.pred_preprocessing.file_schema_check(X_processed, self.log, pred_log_path)
            self.log.log(pred_log_file, 'Prediction file format matched the prediction schema')
            self.log.log(pred_log_file, 'Data preprocessed successfully')
            print('Prediction file format matched the prediction schema ')
            print('Data preprocessed successfully')

            # Predict the output based on X_processed data
            path, json_predictions = self.model_pred.model_pred(X_checked, wafer_names, self.file_operation,
                                                             self.log, pred_log_path)
            self.log.log(pred_log_file, 'Model prediction successfully')
            print('Model prediction successfully')

            # Retrain the model again on the predicted data to make it more generalized for the future predictions of new dataset
            self.modelretrain.model_retrain(self.log)

            # Delete unnecessary folders
            self.directory_creator.del_directory(self.log)

            print('Model retraining was successful ')
            print('After reviewing the performance please delete the prediction_output.csv file and confusion_matrix folder files in the end.')

            pred_log_file.close()  # close the log file
            return path, json_predictions

        except Exception as e:
            # Write the log file for an error and close it
            self.log.log(pred_log_file, 'Error during prediction_validation')
            self.log.log(pred_log_file, str(e))
            pred_log_file.close()
