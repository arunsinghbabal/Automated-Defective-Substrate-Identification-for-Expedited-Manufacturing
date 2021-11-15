import pandas as pd
import os
from sklearn.impute import KNNImputer


class TrainPreprocessing:
    def __init__(self, par_path):
        self.par_path = par_path

    def data_impute(self, log, log_path):
        """
            Method Name: data_impute
            Description: This function identifies the columns with missing values and impute those missing
            values using the KNNImputer.
            Output: X_imputed,y
        """
        # Provides the folder path, contain all the good records in a single csv file
        train_combined_csv_path = os.path.join(self.par_path,
                                               'train_combined_csv/')
        log_file = open(log_path + 'preprocessing.txt', 'a+')  # open the log file
        # Create a dataframe with input data records
        df = pd.read_csv(train_combined_csv_path + 'train_unprocessed_input.csv')
        # Drop unnecessary columns
        df = df.drop(['Wafer'], axis=1)
        # assign the dependent and independent columns to variables
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        # Create a dataframe that contains the column names along with the count of missing values associated with it
        # and save it as input_null.csv
        df_null_record = pd.DataFrame(columns=['columns', 'missing_count'])
        df_null_record['columns'] = X.columns

        df_null_record['missing_count'] = X.isnull().sum().tolist()
        df_null_record.to_csv(train_combined_csv_path + 'train_unprocessed_input_null.csv', index=False, header=True,
                              columns=['columns', 'missing_count'], sep=',')
        try:
            # Using the KNNImputer, which identifies the missing data and replace it with a predicted
            # values using the k-Nearest Neighbors
            impute = KNNImputer(n_neighbors=3)
            X_imputed = pd.DataFrame(impute.fit_transform(X), columns=X.columns)
            log.log(log_file, 'Data imputed successfully')
            log_file.close()  # close the log function
            return X_imputed, y

        except Exception as e:
            # Call the log function to append the error message
            log.log(log_file, 'Data imputation unsuccessful')
            log.log(log_file, str(e))
            log_file.close()  # closes the log function

    def data_variance(self, X, log, log_path):
        """
            Method Name: data_variance
            Description: This function finds the columns with zero variance and drops them.
            Output: X
        """
        log_file = open(log_path + 'preprocessing.txt', 'a+')
        try:
            # Summarize the numerical data and assign it to a variable
            X_des = X.describe()
            #  Identify the zero variance columns and drop them
            col_std_zero = X_des.loc[:, X_des.loc['std'] == 0].columns.tolist()
            X.drop(col_std_zero, axis=1, inplace=True)
            log.log(log_file, 'Columns removed with zeo standard deviation successfully')
            log_file.close()  # closes the log function
            return X
        except Exception as e:
            # Call the log function to append the error message
            log.log(log_file, 'Error during the removal of columns with zero deviation')
            log.log(log_file, str(e))
            log_file.close()  # closes the log function
