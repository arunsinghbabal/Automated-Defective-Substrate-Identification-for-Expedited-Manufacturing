import os
from application_files.prediction_pipline.prediction_Validation import PredictionValidation
from flask_cors import CORS, cross_origin
import json
from flask import Flask, request, render_template
from flask import Response


app=Flask(__name__)
CORS(app)


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('predict.html')

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRouteClient():
    try:
        if request.json is not None:
            path = request.json['filepath']
            pred_obj = PredictionValidation(os.getcwd(), path)
            out_path,json_predictions = pred_obj.pred_validation()
            return Response("Prediction File Location:\n" + str(out_path) + '\nand few of the predictions are:\n'
                            + str(json.loads(json_predictions)))
        elif request.form is not None:
            path = request.form['filepath']
            pred_obj = PredictionValidation(os.getcwd(), path)
            out_path, json_predictions = pred_obj.pred_validation()
            return Response("Prediction File location:\n" + str(out_path) + '\nand few of the predictions are:\n'
                            + str(json.loads(json_predictions)))
        else:
            print('Nothing Matched')
    except ValueError:
        return Response("Error Occurred! %s" %ValueError)
    except KeyError:
        return Response("Error Occurred! %s" %KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" %e)

if __name__ == "__main__":
    app.run()