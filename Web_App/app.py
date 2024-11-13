from flask import Flask, request, render_template, redirect
import numpy as np
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)

from tensorflow.keras.models import load_model
import pickle
import os

import dataValidation
import fileHandling
import dataExtraction
import search

app = Flask(__name__)

TACTICS = ["Initial Access", "Execution","Persistence","Privilege Escalation","Defense Evasion","Credential Access","Discovery","Lateral Movement", "Collection","Exfiltration","Command and Control","Impact","Impact2","Reconnaissance"]

# Load models
detectionModel = load_model('./Models/1D-CNN.h5')
models = {}
for i in range(0,14):  # 14 TTP models
    if i == 0 or i == 7 or i == 9 or i == 13:
        models[i]= ''
    else:
        models[i] = load_model(f'./Models/1D-CNN_{TACTICS[i]}.h5')


@app.route('/', methods=['GET','POST'])
def index():
    # Return the upload HTML file
    return render_template('index.html')

@app.route('/upload', methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        try:
            file = request.files['file']
            print(file.filename[-3:-1])
            print(os.path.splitext(file.filename))
            if os.path.splitext(file.filename)[-1] == '.npy':
                fileHandling.saveFile(file)
                return redirect(f'/predict/{file.filename}')
            elif os.path.splitext(file.filename)[-1] == '.json':
                print('json file being saved')
                fileHandling.saveFile(file)
                dataExtraction.feature_extract(file.filename)
                return redirect(f'/predict/{os.path.splitext(file.filename)[0]}.npy')
        except Exception as e:
            print(f"Error:{e}")
    else:
        return render_template('index.html')

#uncomment for original code
# @app.route('/predict/<file>', methods=['GET'])
# def predict(file):
#     if request.method == "GET":
#         # Read the file content
#         npy_file = np.load(f'files/npy/{file}')
#         if not dataValidation.validateArrayShape(npy_file):
#             npy_file = dataValidation.fixShape(npy_file)
#         # Reshape or preprocess data as required by your model
#         # npy_file = npy_file.reshape(-1, 105, 258)
#         prediction = model.predict(npy_file)
#         # Customize output as needed
#         # result = prediction.round()
#         # result = prediction
#         # Determine label and color based on prediction
#         label = "Malicious" if prediction[0] > 0.5 else "Benign"
#         label_color = "danger" if prediction[0] > 0.5 else "success"  # Bootstrap classes for color
#         return render_template('predict.html', filename=file, result=prediction[0], label=label, label_color=label_color)
    
@app.route('/predict/<file>', methods=['GET'])
def predict(file):
    data = np.load(f'files/npy/{file}')
    results = {}

    # Determine if malicious
    malScore = detectionModel.predict(data)
    results['MalScore'] = malScore
    if malScore > 0.5:
        # If malicious determine Tactics
        for i in range(14):
            # Iterate through each model to make predictions
            # This 4 checks is to accomdate the 4 missing models. 
            if i == 0 or i == 7 or i == 9 or i == 13:
                results[TACTICS[i]] = 0
            else:
                results[TACTICS[i]] = models[i].predict(data)
        # Return html after getting TTP Predictions. 
        return render_template('predict.html', results=results, filename=file, label="Malicious",label_color = "danger")
    # If not malicious, just return html straigt.  
    else:
        return render_template('predict.html', results = results, filename=file, label="Benign", label_color = "success")
    
    
    # for ttp, model in models.items():
    #     prediction = model.predict(data)  # Adjust this to reshape according to the model input requirements
    #     results[ttp] = prediction
    #     aggregate_prediction += prediction

    # # Determine if the overall prediction is benign or malicious
    # average_prediction = aggregate_prediction / len(models)
    # label = "Malicious" if average_prediction > 0.5 else "Benign"
    # label_color = "danger" if prediction[0] > 0.5 else "success"  # Bootstrap classes for color

    


if __name__ == '__main__':
    app.run(port=3000, debug=True)