import os
import shutil


class DirectoryCreator:
    def __init__(self, par_path):
        self.par_path = par_path

    def directory_creator(self, log, train):
        """
            Method Name: directory_creator
            Description: This function creates all the necessary folders essential for the project operation,
            if they are non-existent.
            Output: None
        """
        if train==True:
            # assign the possible paths to the related variables
            train_bad_path = os.path.join(self.par_path, 'train_raw_bad')
            train_good_path = os.path.join(self.par_path, 'train_raw_good')
            train_combine_path = os.path.join(self.par_path, 'train_combined_csv')
            sql_path = os.path.join(self.par_path, 'sql_db')
            model_path = os.path.join(self.par_path, 'saved_model')
            train_logger_path = os.path.join(self.par_path, 'logger')
            pred_path = os.path.join(self.par_path, 'prediction')
            km_model_path = os.path.join(self.par_path, 'km_model')
            grid_search_cv_path = os.path.join(self.par_path, 'grid_search_cv')
            confusion_matrix_path = os.path.join(self.par_path, 'confusion_matrix')


            # First create a folder to log all the details
            if not os.path.isdir(train_logger_path):
                os.makedirs(train_logger_path)
            # Open the log files
            train_log_file = open(train_logger_path + '/folder_preparation.txt', 'a+')

            try:
                train_folder_paths = [train_bad_path, train_good_path, train_combine_path, sql_path, model_path, train_logger_path, pred_path, km_model_path, grid_search_cv_path, confusion_matrix_path]
                # Verify if the folder exist or not. If it does not exist than create it.
                for path in train_folder_paths:
                    if not os.path.isdir(path):
                        os.makedirs(path)

                # Call the log function to append the success message
                log.log(train_log_file, 'Folder created successfully')
                train_log_file.close()  # close the log file

            except OSError as s:
                # Call the log function to append the error message
                log.log(train_log_file, 'Error during folder creation: %s' % s)
                train_log_file.close()  # close the log file

        else:
            # assign the possible paths to the related variables
            pred_bad_path = os.path.join(self.par_path, 'pred_raw_bad')
            pred_good_path = os.path.join(self.par_path, 'pred_raw_good')
            pred_combine_path = os.path.join(self.par_path, 'pred_combined_csv')
            sql_path = os.path.join(self.par_path, 'sql_db')
            train_logger_path = os.path.join(self.par_path, 'logger')
            pred_logger_path = os.path.join(train_logger_path, 'prediction')
            pred_path = os.path.join(self.par_path, 'prediction')
            confusion_matrix_path = os.path.join(self.par_path, 'confusion_matrix')

            # First create a folder to log all the details
            if not os.path.isdir(pred_logger_path):
                os.makedirs(pred_logger_path)

            # Open the log file
            pred_log_file = open(pred_logger_path + '/prediction_folder_preparation.txt', 'a+')

            try:
                pred_folder_paths = [pred_bad_path, pred_good_path, pred_combine_path, sql_path, train_logger_path, pred_logger_path, pred_path, confusion_matrix_path]

                # Verify if the folder exist or not. If it does not exist than create it.
                for path in pred_folder_paths:
                    if not os.path.isdir(path):
                        os.makedirs(path)

                # Call the log function to append the success message
                log.log(pred_log_file, 'Prediction folder created successfully')
                pred_log_file.close()

            except OSError as s:
                # Call the log function to append the error message
                log.log(pred_log_file, 'Error during folder creation: %s' % s)
                pred_log_file.close()

    def del_directory(self, log):
        """
            Method Name: del_directory
            Description: This function delete all the necessary folders
            Output: None
        """
        log_path = os.path.join(self.par_path, 'logger')
        log_file = open(log_path + '/folder_remover.txt', 'a+')
        try:
            # Initialize the folder path
            train_bad_path = os.path.join(self.par_path, 'train_raw_bad')
            train_good_path = os.path.join(self.par_path, 'train_raw_good')
            pred_bad_path = os.path.join(self.par_path, 'pred_raw_bad')
            pred_good_path = os.path.join(self.par_path, 'pred_raw_good')
            sql_path = os.path.join(self.par_path, 'sql_db')
            path_list = [train_bad_path, train_good_path, pred_bad_path, pred_good_path, sql_path]
            for path in path_list:
                if os.path.isdir(path):
                    # remove all the unnecessary folders
                    shutil.rmtree(path)
            log.log(log_file, 'Folders removed successfully')
            log_file.close()

        except Exception as e:
            # Write the log file for an error and close it
            log.log(log_file, 'Error during folder removal')
            log.log(log_file, str(e))
            log_file.close()


