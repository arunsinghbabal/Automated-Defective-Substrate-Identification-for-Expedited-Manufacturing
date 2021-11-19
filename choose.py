import sys
from application_files.training_pipline.training_Validation import TrainValidation
from application_files.prediction_pipline.prediction_Validation import PredictionValidation
import os

argument = sys.argv
print(argument)
cur_path=os.getcwd()
if len(argument) == 3 and 'Train' in argument[1]:
    path = argument[2]
    print('in train')
    train_obj = TrainValidation(cur_path, path)
    out_path = train_obj.train_validation()
    print('Training successful')
    print('Check the folder ', out_path)
elif len(argument) == 2 and 'Train' in argument[1]:
    path = os.path.join(cur_path, 'train_raw_files')
    print('in train')
    train_obj = TrainValidation(cur_path, path)
    out_path = train_obj.train_validation()
    print('Training successful')
    print('Check the folder ', out_path)
elif len(argument) == 3 and 'Predict' in argument[1]:
    print('in predict')
    path = argument[2]
    pred_obj = PredictionValidation(cur_path, path)
    out_path, json_predictions = pred_obj.pred_validation()
    print('Prediction successful')
    print('Check the folder ', out_path)

elif len(argument) == 2 and 'Predict' in argument[1]:
    path = os.path.join(cur_path, 'pred_raw_files')
    pred_obj = PredictionValidation(cur_path, path)
    out_path, json_predictions = pred_obj.pred_validation()
    print('Prediction successful')
    print('Check the folder ', out_path)
