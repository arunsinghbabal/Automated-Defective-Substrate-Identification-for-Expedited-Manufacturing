import pandas as pd
import os


class PredictFromModel:
    def __init__(self, par_path):
        self.par_path = par_path

    def model_pred(self, X_processed, wafer_names, file_operation, log, pred_log_path):
        """
            Method Name: model_pred
            Description: This function predict the output from the precessed X values
            Output: path, json_predictions
        """
        pred_log_file = open(pred_log_path + 'model_pred.txt', 'a+')  # open the log file
        try:
            pred_path = os.path.join(self.par_path, 'prediction/')
            pred_combined_path = os.path.join(self.par_path, 'pred_combined_csv/')
            model_path = os.path.join(self.par_path, 'saved_model/')
            km_model_path = os.path.join(self.par_path, 'km_model/')
            km_model = file_operation.load_model(km_model_path, 'km', log, pred_log_path)
            # Predict the cluster number for all the data using the previously trained kmeans model
            cluster = km_model.predict(X_processed)
            X_processed['cluster'] = cluster
            X_processed['Wafer'] = wafer_names
            clusters = X_processed['cluster'].unique()  # Find the unique clusters in whole dataset
            X_processed.to_csv(pred_combined_path + 'pred_processed_input.csv', header=True,
                               index=False)  # Save the processed dataset
            print('before loop')
            num=0

            for c in clusters:
                # Extract the data below to the cluster 'c'
                X_pred = X_processed[X_processed['cluster'] == c].drop(['cluster'], axis=1)
                wafer_name = X_pred['Wafer']  # Extract the wafer_name from the cluster data
                # Find the model related with the cluster 'c'
                model_name = file_operation.model_finder(model_path, c, log, pred_log_path)
                # Load the model
                model_obj = file_operation.load_model(model_path, model_name, log, pred_log_path)
                y_pred = list(model_obj.predict(X_pred.drop(['Wafer'], axis=1)))  # Predict the output
                # Create dataframe contains wafer_name and predicted value
                result = pd.DataFrame(list(zip(wafer_name, y_pred)), columns=['Wafer', 'label'])
                result['cluster'] = c
                cluster_output = pd.merge(X_pred, result, on=['Wafer'],  how="inner")
                # Save the prediction in prediction_output.csv file and append it with every loop
                if num == 0:
                    cluster_output.to_csv(pred_path + 'prediction_output.csv', index=False, mode='a+')
                    num = num+1
                else:
                    cluster_output.to_csv(pred_path + 'prediction_output.csv', header=False, index=False, mode='a+')



            log.log(pred_log_file, 'Prediction successfully')
            pred_log_file.close()
            path = pred_path + 'prediction_output.csv'
            return path, result.head().to_json(orient="records")

        except Exception as e:
            # Write the log and then close it
            log.log(pred_log_file, 'Prediction unsuccessful')
            log.log(pred_log_file, str(e))
            pred_log_file.close()
            raise e
