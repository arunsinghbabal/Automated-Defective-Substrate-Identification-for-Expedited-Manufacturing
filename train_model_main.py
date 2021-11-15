import os
from application_files.training_pipline.training_Validation import TrainValidation
import json
from flask import Flask, request, render_template
from flask import Response
from flask_cors import CORS, cross_origin

app=Flask(__name__)
CORS(app)


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('train.html')

@app.route("/train", methods=['POST'])
@cross_origin()
def trainRouteClient():
    try:
        if request.json is not None:
            path = request.json['filepath']
            train_obj = TrainValidation(os.getcwd(), path)
            out_path = train_obj.train_validation()
            return Response("Saved model File Location:\n" + str(out_path))
        elif request.form is not None:
            path = request.form['filepath']
            train_obj = TrainValidation(os.getcwd(), path)
            out_path = train_obj.train_validation()
            return Response("Saved model File Location:\n" + str(out_path))
        else:
            print('Model Not Matched')
    except ValueError:
        return Response("Error Occurred! %s" %ValueError)
    except KeyError:
        return Response("Error Occurred! %s" %KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" %e)

if __name__ == "__main__":
    app.run()