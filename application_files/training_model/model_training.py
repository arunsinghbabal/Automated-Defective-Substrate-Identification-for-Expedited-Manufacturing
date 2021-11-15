import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score


class ModelTraining:
    def __init__(self, par_path):
        self.par_path = par_path
        self.saved_model_path = os.path.join(par_path, 'saved_model/')
        self.model_params_path = os.path.join(par_path, 'grid_search_cv/')

    def data_knee_locator(self, X_new,  log, log_path):
        """
            Method Name: data_knee_locator
            Description: This function finds optimal number of clusters in our independent dataset. First it
            apply the KMeans clustering and obtains the sum of squared distance from the cluster centre. Later, it
            uses KneeLocator function to find the maximum curvature in the line, which is our optimal cluster number.
            Output: km_cluster
        """
        log_file = open(log_path + 'model_training.txt', 'a+')
        pred_path = os.path.join(self.par_path, 'prediction/')
        # Initializing the variable that will contain the sum of squared distance from the cluster center
        dis = []
        try:
            for i in range(1, 20):
                # Cluster all the data in assigned i number of clusters
                km = KMeans(n_clusters=i, random_state=42)
                km.fit(X_new)
                # append the distance value from cluster center in the dis variable
                dis.append(km.inertia_)
            # Finding the optimal cluster numbers using KneeLocator
            kl = KneeLocator(range(1, 20), dis, direction='decreasing', curve='convex')
            km_cluster = kl.knee
            # Plot the sum of squared distance from the cluster center and number of clusters for visualisation
            plt.plot(range(1, 20), dis)
            plt.axvline(kl.knee, c='r', linestyle='--')
            plt.legend(['Distance', 'Elbow'])
            plt.xlabel('Cluster no.')
            plt.ylabel('Sum of squared distance from the cluster')
            plt.title('To find optimal cluster-elbow method')
            # Save the plot as elbow_method.jpg
            plt.savefig(pred_path + 'elbow_method.jpg')
            plt.show()
            log.log(log_file, 'The optimal cluster number is {}'.format(km_cluster))  # write the log
            log_file.close()  # close the log file
            return km_cluster
        except Exception as e:
            # report the error in data_knee_locator function
            log.log(log_file, 'Error during finding the optimal cluster number')
            log.log(log_file, str(e))
            log_file.close()  # close the log file

    def data_clustering(self, km_cluster, X, y, log, log_path):
        """
            Method Name: data_clustering
            Description: The independent dataset is optimally clustered with a value obtained from the
            data_knee_locator function
            Output: X
        """
        log_file = open(log_path + 'model_training.txt', 'a+')  # open the log file
        try:
            # cluster the data in a optimal number of clusters
            km = KMeans(n_clusters=km_cluster, random_state=42)
            y_km = km.fit_predict(X)
            with open(os.path.join(self.par_path, 'km_model', 'km.sav'), 'wb') as f:
                # Save the Kmeans model after fitting it with our dataset
                pickle.dump(km, f)
            # Add a column named 'cluster' in the dataset contains the cluster number associated with the rows
            X['cluster'] = y_km
            log.log(log_file, 'Data clustered successfully')  # Write the log file
            # close the log file
            log_file.close()
            return X, y
        except Exception as e:
            # Write the error in the file and then close it
            log.log(log_file, 'Error during data clustering')
            log.log(log_file, str(e))
            log_file.close()

    def balancing_cluster_count(self, X_clustered, y, log, log_path):
        """
            Method Name: balancing_cluster_count
            Description: This function balances the biased dataset using the SMOTE
            Output: X
        """
        log_file = open(log_path + 'model_training.txt', 'a+')
        count = y.value_counts()  # Finds the count of unique values in dependent column
        print(y.shape)
        print(X_clustered)
        try:
            if count.max() / count.min() > 1.5:
                # Using SMOTE to balance the dataset. note: when applying SMOTE always use independent dataset
                # as y values and the dependent one as x
                sm = SMOTE(random_state=42)
                print('before_smote')
                X_res, y_res = sm.fit_resample(X_clustered, y)
                print(X_res.shape)
                print(y_res.shape)
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

    def model_best_params(self, X, km_cluster, log, log_path):
        """
            Method Name: model_best_params
            Description: This function uses the GridSearchCV to obtain best model parameters for the respective
             cluster number
            Output: best_params, score, model_obj
        """
        log_file = open(log_path + 'model_training.txt', 'a+')
        # Prepare a dictionary containing range of model parameters within which we will obtain the best parameters
        params = {'RandomForestClassifier': {"n_estimators": [10, 50, 100, 200],
                                             "max_features": ['auto', 'sqrt', 'log2'],
                                             "criterion": ['gini', 'entropy']},
                  'KNeighboursClassifier': {'n_neighbors': range(2, 15),
                                            'weights': ['uniform', 'distance'],
                                            'algorithm': ['ball_tree', 'kd_tree', 'brute']},
                  'LogisticRegression': {'solver': ['newton-cg', 'sag', 'lbfgs', 'liblinear'],
                                         'multi_class': ['ovr', 'multinomial'],
                                         'max_iter': [50, 100, 150, 200]},
                  'XGBClassifier': {'learning_rate': [0.001, 0.01, 0.1, 0.5],
                                    'max_depth': [3, 7, 10, 15, 20],
                                    'n_estimator': [10, 50, 100, 200]}
                  }
        # preparing the dictionary contain the model instances
        models = {'RandomForestClassifier': RandomForestClassifier(),
                  'KNeighboursClassifier': KNeighborsClassifier(),
                  'LogisticRegression': LogisticRegression(),
                  'XGBClassifier': XGBClassifier()
                  }
        best_params = {}
        score = {}
        model_obj = {}
        try:
            # Prepare a structured dictionaries, which will later be used to store bset parameters,
            # model scores and model objects
            for c in range(0, km_cluster):
                best_params[c] = {}
                score[c] = {}
                model_obj[c] = {}
                for i in models.keys():
                    best_params[c][i] = {}
                    score[c][i] = {}
                    model_obj[c][i] = {}
            # Loop over the count of optimal clusters and identify the best parameters for each model with respect to
            # the cluster number
            for c in range(0, km_cluster):
                # Filter out the data which belongs to the certain cluster number c
                y_final = X.loc[X['cluster'] == c, 'label']
                X_final = X[X['cluster'] == c].drop(['cluster', 'label'], axis=1)
                # Split the data for training and testing
                X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=1 / 5, random_state=42)
                # For a specific cluster number, find the best parameters of every model one by one
                for i in models.keys():
                    est = models[i]  # initialize the model
                    est_params = params[i]  # Assign the model parameters to a variable
                    # Assign the model instance and its parameters to GridSearchCV to obtain the best model parameters
                    gscv = GridSearchCV(estimator=est, param_grid=est_params, cv=10, verbose=3)
                    gscv.fit(X_train, y_train)
                    best_params[c][i] = gscv.best_params_
            # Save all the parameters obtained from the GridSearchCV
            with open(self.model_params_path + 'grid_search_cv_parameters.json', 'w') as fp:
                json.dump(best_params, fp)
            log.log(log_file, 'Found the best parameters for models')  # Write the log file in the end
            log_file.close()  # close the log file
            return best_params, score, model_obj
        except Exception as e:
            # Report an error in the log file and then close it
            log.log(log_file, 'Could not found  the best parameters for models')
            log.log(log_file, str(e))
            log_file.close()

    def model_training(self, X, km_cluster, log, log_path, best_params, score, model_obj):
        """
            Method Name: model_training
            Description: This function identifies the best model for each cluster and saves it in the saved_model folder
            Output: None
        """
        print('in')
        log_file = open(log_path + 'model_training.txt', 'a+')
        try:
            for c in range(0, km_cluster):
                # Initialize the dictionary with model and its best parameters obtained from the GridSearchCV
                model_best_params = {
                    'RandomForestClassifier': RandomForestClassifier(**best_params[c]['RandomForestClassifier']),
                    'KNeighboursClassifier': KNeighborsClassifier(**best_params[c]['KNeighboursClassifier']),
                    'LogisticRegression': LogisticRegression(**best_params[c]['LogisticRegression']),
                    'XGBClassifier': XGBClassifier(**best_params[c]['XGBClassifier'])}
                print(model_best_params)
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
                confusion_matrix_path = os.path.join(self.par_path, 'confusion_matrix/')
                confusion_matrix_df.to_csv(confusion_matrix_path + '{a}_{b}_confusion_matrix.csv'.format(a=model_name, b=str(c)), index=True, header=True, sep=',')
                with open(os.path.join(self.saved_model_path, model_name + '_' + str(c) + '.sav'), 'wb') as f:
                    # Saves the highest score model in cluster c
                    pickle.dump(model_obj[c][model_name], f)

            # Initialize the dataframes with score and best parameter values, respectively
            sc_df = pd.DataFrame(score, columns=score.keys())
            # save the dataframes in a csv format
            sc_df.to_csv(confusion_matrix_path + 'score.csv', index=True, header=True, sep=',')
            log.log(log_file, 'Model training successful')  # write log file
            log_file.close()  # close the log file
            return self.saved_model_path
        except Exception as e:
            # Write the log file for an error and close it
            log.log(log_file, 'Error during model training')
            log.log(log_file, str(e))
            log_file.close()
