import os
import pandas as pd
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder



class ModelTraining:
    def __init__(self, par_path):
        self.par_path = par_path
        self.saved_model_path = os.path.join(par_path, 'saved_model/')

    def balancing_cluster_count(self, X_clustered, y, log, log_path):
        """
            Method Name: balancing_cluster_count
            Description: This function balances the biased dataset using the SMOTE
            Output: X
        """
        log_file = open(log_path + 'model_training.txt', 'a+')
        count = y.value_counts()  # Finds the count of unique values in dependent column
        try:
            if count.max() / count.min() > 1.5:
                # Using SMOTE to balance the dataset. note: when applying SMOTE always use independent dataset
                # as y values and the dependent one as x
                sm = SMOTE(random_state=42)
                X_res, y_res = sm.fit_resample(X_clustered, y)
                X_res['label'] = y_res
                log.log(log_file, 'Data equalized in ratio')  # Write the log file
                log_file.close()  # close the log file
                return X_res
            else:
                # if data set is almost balanced then return the X, y values
                X_clustered['label'] = y
                log.log(log_file, 'Data is already in equalized state')
                log_file.close()
                return X_clustered
        except Exception as e:
            # First write the error in the log file and then close it
            log.log(log_file, 'Error during the data equalization')
            log.log(log_file, str(e))
            log_file.close()

    def model_retraining(self, X, km_cluster, log, log_path, best_params, score, model_obj):
        """
            Method Name: model_training
            Description: This function identifies the best model for each cluster and saves it in the saved_model folder
            Output: None
        """
        log_file = open(log_path + 'model_training.txt', 'a+')
        #  Delete the existing confusion matrix folder and create a new one so that only the newer data is
        #  available to user
        try:
            for c in range(0, km_cluster):
                # Initialize the dictionary with model and its best parameters obtained from the GridSearchCV
                model_best_params = {
                    'RandomForestClassifier': RandomForestClassifier(**best_params[str(c)]['RandomForestClassifier']),
                    'KNeighboursClassifier': KNeighborsClassifier(**best_params[str(c)]['KNeighboursClassifier']),
                    'LogisticRegression': LogisticRegression(**best_params[str(c)]['LogisticRegression']),
                    'XGBClassifier': XGBClassifier(**best_params[str(c)]['XGBClassifier'])}
                # Filter out the data which belongs to the certain cluster number c
                y_final = X.loc[X['cluster'] == c, 'label']
                X_final = X[X['cluster'] == c].drop(['cluster', 'label'], axis=1)

                # Split the data for training and testing
                X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=1 / 5, random_state=42)
                # Training the models one by one appending their score in the 'score' dictionary
                for m, p in model_best_params.items():
                    model = p
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    confusion_matrix_df = pd.crosstab(y_test, y_pred)
                    model_obj[c][m] = model  # saving the model instance in a dictionary
                    if len(y_test.value_counts()) == 1:
                        sc = accuracy_score(y_test, y_pred)
                        score[c][m] = sc
                    else:
                        sc = roc_auc_score(y_test, y_pred)
                        score[c][m] = sc
                model_name = max(score[c].items(), key=lambda x: x[1])[0]  # gives the model name with the highest score
                #confusion_matrix = os.path.join(self.par_path, 'confusion_matrix/')
                # Save the confusion matrices for the user to see the performance
                confusion_matrix_df.to_csv(os.path.join(self.par_path, 'confusion_matrix', '{a}_{b}_confusion_matrix.csv'.format(a=model_name, b=str(c))), index=True, header=True, sep=',')

                with open(os.path.join(self.saved_model_path, model_name + '_' + str(c) + '.sav'), 'wb') as f:
                    # Saves the highest score model in cluster c
                    pickle.dump(model_obj[c][model_name], f)

            # Initialize the dataframes with score and best parameter values, respectively
            sc_df = pd.DataFrame(score, columns=score.keys())
            # save the dataframes in a csv format
            sc_df.to_csv(os.path.join(self.par_path, 'confusion_matrix', 'score.csv'), index=True, header=True, sep=',')
            log.log(log_file, 'Model training successful')  # write log file
            log_file.close()  # close the log file

        except Exception as e:
            # Write the log file for an error and close it
            log.log(log_file, 'Error during model training')
            log.log(log_file, str(e))
            log_file.close()
