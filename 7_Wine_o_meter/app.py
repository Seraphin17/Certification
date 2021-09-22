
import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, url_for, request, jsonify


app = Flask(__name__)

'Web application deployed : https://ynoudjoukouang-wine1.herokuapp.com/'

model = joblib.load('model.joblib')
columns_item = ['fixed_acidity', 'volatile_acidity', 'citric_acid','residual_sugar', 
                'chlorides', 'free_sulfur_dioxide','total_sulfur_dioxide', 'density', 
                    'pH', 'sulphates', 'alcohol']


@app.route("/documentation", methods=["GET", "POST"])
def docu():
    if request.method == "GET":
        return render_template("documentation.html")


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")
    
    # Collecting the data from manual input
    if request.method == "POST":
        fixed_acidity = request.form['fixed_acidity']
        volatile_acidity = request.form['volatile_acidity']
        citric_acid = request.form['citric_acid']
        residual_sugar = request.form['residual_sugar']
        chlorides = request.form['chlorides']
        free_sulfur_dioxide = request.form['free_sulfur_dioxide']
        total_sulfur_dioxide = request.form['total_sulfur_dioxide']
        density = request.form['density']
        pH = request.form['pH']
        sulphates = request.form['sulphates']
        alcohol = request.form['alcohol']
    
        #Creating a pandas DataFrame with the collected data
        collected_data = pd.DataFrame([[fixed_acidity,volatile_acidity,citric_acid,
        residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,
        density,pH,sulphates,alcohol
        ]], columns=columns_item, dtype='float')


        classifier = model
        prediction = classifier.predict(collected_data)
        prediction = float(prediction[0])


        return render_template("index.html", result='The characteristics of this wine give the mark of : {} / 10.'.format(prediction))


@app.route("/predict_API", methods=["POST"])
def predict():
    # Check if request has a JSON content
    if request.json:
        # Get the JSON as dictionnary
        req = request.get_json()
        # Check mandatory key
        if "input" in req.keys():
            # Load model
            classifier = joblib.load('model.joblib')
            # Predict
            prediction = classifier.predict(req["input"])
            # Return the result as JSON but first we need to transform the
            # result so as to be serializable by jsonify()
            prediction = float(prediction[0])
            return jsonify({"predict": prediction}), 200
        else:
            return jsonify({"msg": "Error: not a JSON or no specific key in your request"})
    return jsonify({"msg": "Error: not a JSON or no specific key in your request"})



if __name__ == "__main__":
    app.run(debug=True)