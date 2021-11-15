import json
import pandas as pd
import os
import re
import shutil


class PredictionDataValidation:
    def __init__(self, par_path):
        self.par_path = par_path

    def pred_from_schema_file(self, schema_path, log, pred_log_path):
        """
            Method Name: pred_from_schema_file
            Description: This function obtained the necessary information from the schema file that are essential for
            validating the prediction raw files.
            Output: sch_filename = File name
                    sch_nos_col = Total number of columns
                    sch_col_name = Dictionary containing column name and its type
                    sch_len_date_stamp = No. of date characters in the filename
                    sch_len_time_stamp =  No. of time characters in the filename
        """

        pred_log_file = open(pred_log_path + 'pred_from_schema_file.txt', 'a+')
        try:
            # open the schema file and load it in a variable
            with open(schema_path, 'r') as f:
                sch = json.load(f)
                f.close()
            # Transfer the relevant syntax format of raw files in the variables
            sch_filename = sch['SampleFileName']
            sch_nos_col = sch['NumberofColumns'] - 1
            sch_col_name = sch['ColName']
            del sch_col_name['Output']
            sch_len_date_stamp = sch['LengthOfDateStampInFile']
            sch_len_time_stamp = sch['LengthOfTimeStampInFile']
            log.log(pred_log_file, 'Successfully assigned the file syntax information in the variables')
            pred_log_file.close()
            return sch_filename, sch_nos_col, sch_col_name, sch_len_date_stamp, sch_len_time_stamp

        except Exception as e:
            log.log(pred_log_file, 'Error during assigning the file syntax information in the variables')
            log.log(pred_log_file, str(e))
            pred_log_file.close()

    def pred_data_validation(self, pred_path, sch_nos_col, sch_col_name, sch_len_date_stamp, sch_len_time_stamp,
                              log, pred_log_path):
        """
            Method Name: pred_data_validation
            Description: This function validates the syntax of prediction raw files before carrying out the
            data preprocessing and training.
            Output: None
        """
        pred_log_file = open(pred_log_path + 'pred_data_validation.txt', 'a+')  # open the log file
        try:
            pred_raw_files_path = pred_path
            # list all the raw files in the folder
            pred_raw_files = [f for f in os.listdir(pred_raw_files_path)]
            pred_raw_filename_syntax = "['wafer']+['\_'']+[\d_]+[\d]+\.csv"  # Syntax of the standard file
            # Looped over each file to verify if it is suitable or not
            for i, f in enumerate(pred_raw_files):
                pred_bad_path = os.path.join(self.par_path, 'pred_raw_bad/')
                pred_good_path = os.path.join(self.par_path, 'pred_raw_good/')
                pred_raw_file_path = os.path.join(pred_raw_files_path, '{}'.format(f))
                # First check the filename for its similarity with syntax
                if re.match(pred_raw_filename_syntax, f):
                    log.log(pred_log_file, 'Prediction filename {} matched with the syntax'.format(f))
                    x = f.split(sep='.')[0].split(sep='_')
                    if (x[0] == 'wafer') and (len(x[1]) == sch_len_date_stamp) and (len(x[2]) == sch_len_time_stamp):
                        log.log(pred_log_file,
                                'Initial string, date and timestamp of prediction file matched perfectly')
                        df = pd.read_csv(pred_raw_file_path)
                        #df = pd.read_csv(pred_raw_file_path + '/{}'.format(f))
                        # Second verify the content of the file such as number of columns, their column name and if
                        # all the column values are null or not
                        if df.shape[1] == sch_nos_col:
                            log.log(pred_log_file,
                                    'No of columns in prediction file named {} are under acceptable limit'.format(f))
                            df.columns = sch_col_name.keys()
                            count = 0
                            # Find the column with all the null values if exist than consider file as a bad file
                            for col in df:
                                if df[col].isna().sum() == len(df):
                                    count += 1
                            if count > 0:
                                shutil.copy(pred_raw_file_path, pred_bad_path)
                            else:
                                # Fill the null values with NaN and consider it as a good file
                                df.fillna(value='NaN', inplace=True)
                                df.to_csv(pred_good_path + '{}'.format(f), header=True, index=False)
                        # write the log file with relevant error of the raw file and copy it to the raw_bad folder
                        else:
                            log.log(pred_log_file,
                                    'No of columns in prediction file named {} are higher than the acceptable'
                                    ' limit'.format(f))
                            shutil.copy(pred_raw_file_path, pred_bad_path)

                    else:
                        log.log(pred_log_file,
                                'Initial string, date and timestamp of the prediction file does not match with '
                                'the given syntax')
                        shutil.copy(pred_raw_file_path, pred_bad_path)
                else:
                    log.log(pred_log_file, 'Filename {} does not matches with the given syntax'.format(f))
                    shutil.copy(pred_raw_file_path, pred_bad_path)
            pred_log_file.close()

        except Exception as e:
            # write the log file for an error in data_validation and then close it
            log.log(pred_log_file, 'Error during validating the prediction raw files')
            log.log(pred_log_file, str(e))
            pred_log_file.close()
