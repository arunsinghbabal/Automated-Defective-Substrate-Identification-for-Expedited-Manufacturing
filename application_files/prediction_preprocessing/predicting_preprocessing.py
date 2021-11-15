import pandas as pd
import os
import json
from sklearn.impute import KNNImputer


class PredictionPreprocessing:
    def __init__(self, par_path):
        self.par_path = par_path
        pass

    def pred_data_impute(self, log, pred_log_path):
        """
            Method Name: pred_data_impute
            Description: This function identifies the columns with missing values and impute those missing values using
             the KNNImputer.
            Output: X_imputed,wafer_names
        """
        # Provides the folder path contain all the good records in a single csv file
        pred_combined_csv_path = os.path.join(self.par_path,
                                              'pred_combined_csv/')
        pred_log_file = open(pred_log_path + 'pred_preprocessing.txt', 'a+')  # open the log file
        # Create a dataframe with input data records
        df = pd.read_csv(pred_combined_csv_path + 'pred_unprocessed_input.csv')
        # Drop unnecessary columns
        wafer_names = df['Wafer']
        df = df.drop(['Wafer'], axis=1)
        # Create a dataframe that contains the column names along with the count of missing values associated with it
        # and save it as input_null.csv
        df_null_record = pd.DataFrame(columns=['columns', 'missing_count'])
        df_null_record['columns'] = df.columns
        # null_col=X.columns[X.isnull().sum()>0].tolist()    columns with values >0
        # null_col=X.columns[X.isnull().any()].tolist()------alternate
        df_null_record['missing_count'] = df.isnull().sum().tolist()
        df_null_record.to_csv(pred_combined_csv_path + 'pred_unprocessed_input_null.csv', index=False, header=True,
                              columns=['columns', 'missing_count'], sep=',')
        try:
            # Using the KNNImputer, which identifies the missing data and replace it with a predicted values using
            # the k-Nearest Neighbors
            impute = KNNImputer(n_neighbors=3)
            df_imputed = pd.DataFrame(impute.fit_transform(df), columns=df.columns)
            log.log(pred_log_file, 'Prediction data imputed successfully')
            pred_log_file.close()  # close the log function
            return df_imputed, wafer_names
        except Exception as e:
            # Call the log function to append the error message
            log.log(pred_log_file, 'Prediction data imputation unsuccessful')
            log.log(pred_log_file, str(e))
            pred_log_file.close()  # closes the log function

    def pred_data_variance(self, X, log, pred_log_path):
        """
            Method Name: data_variance
            Description: This function finds the columns with zero variance and drops them.
            Output: X
        """
        pred_log_file = open(pred_log_path + 'pred_preprocessing.txt', 'a+')
        try:
            # Summarize the numerical data and assign it to a variable
            X_des = X.describe()
            #  Identify the zero variance columns and drop them
            col_std_zero = X_des.loc[:, X_des.loc['std'] == 0].columns.tolist()
            X.drop(col_std_zero, axis=1, inplace=True)
            log.log(pred_log_file, 'Successfully removed columns with zero variance')
            pred_log_file.close()  # closes the log function
            return X
        except Exception as e:
            # Call the log function to append the error message
            log.log(pred_log_file, 'Error during the removal of columns with zero deviation')
            log.log(pred_log_file, str(e))
            pred_log_file.close()  # closes the log function

    def file_schema_check(self, X, log, pred_log_path):
        """
            Method Name: file_schema_check
            Description: This function checks the matching between predicting file and prediction schema
            Output: X
        """
        pred_log_file = open(pred_log_path + 'pred_preprocessing.txt', 'a+')
        try:
            pred_schema_path = os.path.join(self.par_path, 'prediction_schema.json')
            with open(pred_schema_path, 'r') as f:
                sch = json.load(f)
                f.close()
            if sch['col_length'] == len(X.columns):
                if sch['col_names'] == list(X.columns):
                    log.log(pred_log_file, 'Prediction file format matched the prediction schema')
                    pred_log_file.close()  # closes the log function
                    return X
                else:
                    log.log(pred_log_file, 'Prediction file columns did not matched the prediction schema')
                    pred_log_file.close()  # closes the log function
                    print('Prediction file format did not matched the prediction schema')
                    raise
            else:
                print('Prediction file column length did not matched the prediction schema')
                raise

        except Exception as e:
            # Call the log function to append the error message
            log.log(pred_log_file, 'Error during matching the prediction schema with prediction file')
            log.log(pred_log_file, str(e))
            pred_log_file.close()  # closes the log function
