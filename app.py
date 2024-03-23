from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import sklearn
import pickle

# importing model
model = pickle.load(open('crop_model.pkl', 'rb'))


# creating flask app
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    N = int(request.form['Nitrogen'])
    P = int(request.form['Phosphorus'])
    K = int(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['PH'])
    rainfall = float(request.form['Rainfall'])

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    predict = model.predict(single_pred)

    crop_dict = {1: 'rice', 2: 'maize', 3: 'jute', 4: 'cotton', 5: 'coconut', 6: 'papaya', 7: 'orange', 8: 'apple', 9: 'muskmelon', 10: 'watermelon',
                 11: 'grapes', 12: 'mango', 13: 'banana', 14: 'pomegranate', 15: 'lential', 16: 'blackgram', 17: 'mungbean', 18: 'mothbeans', 19: 'pigeonpeas',
                 20: 'kidneybeans', 21: 'chickpea', 22: 'coffee'}

    if predict[0] in crop_dict:
        crop = crop_dict[predict[0]]
        result = "{} will be best crop for cultivation".format(crop)
    else:
        result = "Can't culvited according input given"
    return render_template('index.html', result=result)


# main function
if __name__ == "__main__":
    app.run(debug=True)
