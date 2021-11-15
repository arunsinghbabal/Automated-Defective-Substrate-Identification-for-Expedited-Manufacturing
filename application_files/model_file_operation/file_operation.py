import os
import pickle
import re


class FileOperationModelFinder:
    def __init__(self):
        pass

    def load_model(self, model_path, filename, log, log_path):
        """
            Method Name: load_model
            Description: This function loads the saved model
            Output: model
        """
        log_file = open(log_path + 'load_model.txt', 'a+')  # open the log file
        try:
            with open(os.path.join(model_path, filename + '.sav'), 'rb') as f:
                # open read the model file and load it in a variable
                model = pickle.load(f)
            log.log(log_file, 'Loaded model {m}.sav successfully'.format(m=filename))  # write the log
            log_file.close()
            return model

        except Exception as e:
            # Write error in log file and then close it
            log.log(log_file, ' model {m}.sav loading unsuccessful'.format(m=filename))
            log.log(log_file, str(e))
            log_file.close()

    def model_finder(self, model_path, cluster_no, log, log_path):
        """
            Method Name: model_finder
            Description: This function finds the optimal model for the specific cluster
            Output: optimal_model
        """
        log_file = open(log_path + 'model_finder.txt', 'a+')  # open the log file
        try:
            models = os.listdir(model_path)  # List all the files in the folder
            # Assign the model filename syntax to a variable
            model_syntax = "[\D]+[\W\d]+\.sav"
            for model in models:
                if re.match(model_syntax, model):  # Match the syntax file with the files in the folder
                    x = model.split(sep='.')[0]
                    if x.split(sep='_')[1] == str(
                            cluster_no):  # Check if the value after '_' is equal to the cluster_no
                        optimal_model = x
                        log.log(log_file, 'Optimal model search successfully')
                        log_file.close()
                        return optimal_model

        except Exception as e:
            # Write the log and then close it
            log.log(log_file, 'Optimal model search unsuccessful')
            log.log(log_file, str(e))
            log_file.close()


