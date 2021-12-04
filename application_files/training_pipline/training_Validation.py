import os
from application_files.log.logger import Logger
from application_files.create_directory.create_directory import DirectoryCreator
from application_files.training_data_validation.data_validation import DataValidation
from application_files.training_sql.training_sql_db import SqlDb
from application_files.training_preprocessing.training_preprocessing import TrainPreprocessing
from application_files.training_model.model_training import ModelTraining


class TrainValidation:
    def __init__(self, par_path, train_path):
        self.train_path = train_path
        self.par_path = par_path
        self.log = Logger()
        self.directory_creator = DirectoryCreator(self.par_path)
        self.data_validation = DataValidation(self.par_path)
        self.sql_db = SqlDb(self.par_path)
        self.preprocessing = TrainPreprocessing(self.par_path)
        self.model_training = ModelTraining(self.par_path)

    def train_validation(self):
        """
            Method Name: log
            Description: This function validates and process the data and then train the models on it by calling
            various functions
            Output: None
        """
        log_path = os.path.join(self.par_path, 'logger/')  # path to the logger folder
        # Create necessary folders
        self.directory_creator.directory_creator(self.log, train=True)
        log_file = open(log_path + 'train_validation.txt', 'a+')  # open the log file
        self.log.log(log_file, 'Directories created successfully')  # write the log file
        schema_path = os.path.join(self.par_path, 'schema.json')  # path to the schema file
        train_combined_csv_path = os.path.join(self.par_path, 'train_combined_csv/')

        try:
            # Obtain the necessary parameters from the schema file
            sch_filename, sch_nos_col, sch_col_name, sch_len_date_stamp, sch_len_time_stamp = self.data_validation.\
                from_schema_file(schema_path, self.log, log_path)
            self.log.log(log_file, 'Parameters obtained successfully from schema file')
            print('Parameters obtained successfully from schema file')

            # Validates the given raw data according to the schema parameters
            self.data_validation.data_validation(sch_nos_col, sch_col_name, sch_len_date_stamp, sch_len_time_stamp,
                                                 self.train_path, self.log, log_path)
            self.log.log(log_file, 'Raw files successfully validated')
            print('Raw files successfully validated')

            # Create a SQL databases, create a table with good input records and then save it in a file Input.csv
            self.sql_db.database(sch_col_name, self.log, log_path)
            self.log.log(log_file, 'Sql database created successfully')
            print('Sql database created successfully')

            # carry out the data preprocessing. Fill the missing values with relevant data
            X, y = self.preprocessing.data_impute(self.log, log_path)
            self.log.log(log_file, 'Data imputed successfully')
            print('Data imputed successfully')

            # Remove columns with zero variance
            X = self.preprocessing.data_variance(X, self.log, log_path)
            self.log.log(log_file, 'Removed columns with zero variance successfully')
            self.log.log(log_file, 'Data preprocessed successfully')
            print('Removed columns with zero variance successfully')
            print('Data preprocessed successfully')

            # Identify the optimal clusters in dataset
            km_cluster = self.model_training.data_knee_locator(X, self.log, log_path)
            self.log.log(log_file, 'Cluster number obtained successfully')
            print('Cluster number obtained successfully')

            # Assign the cluster numbers to the input data
            X_clustered, y = self.model_training.data_clustering(km_cluster, X, y, self.log, log_path)
            self.log.log(log_file, 'Data clustering and labeling was successful')
            print('Data clustering and labeling was successful')
            print('Data clustering and labeling was successful')

            # Balance the biased dataset. This step is not necessary because it will not impact the accuracy of the model. The reason is that the
            # decision trees follow their own hierarchy. Here we are only practicing the idea of balancing the dataset, which will not have positve or negative impact on the outcome..
            X_new = self.model_training.balancing_cluster_count(X_clustered, y, self.log, log_path)
            X_new.to_csv(train_combined_csv_path + 'train_processed_input.csv', header = True, index = False)
            self.log.log(log_file, 'Equalized the ratio of data in each cluster')
            print('Equalized the ratio of data in each cluster')

            # Obtain the best parameters for the models
            best_params, score, model_obj = self.model_training.model_best_params(X_new, km_cluster, self.log, log_path)
            self.log.log(log_file, 'Model best parameters selected')
            print('Model best parameters selected')

            # Train the models on the dataset and save the best one for every cluster
            model_path = self.model_training.model_training(X_new, km_cluster, self.log, log_path, best_params, score,
                                                       model_obj)
            self.log.log(log_file, 'Models trained successfully and saved in model_saved folder')
            print('Models trained successfully and saved in model_saved folder')
            log_file.close()  # close the log file

            return model_path
        except Exception as e:
            # Write the log file for an error and close it
            self.log.log(log_file, 'Error during train_validation')
            self.log.log(log_file, str(e))
            log_file.close()
