# app.py

import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['file']

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        data = pd.read_csv(filepath)

        # Example dataset columns:
        # Year, Sales

        X = data[['Year']]
        y = data['Sales']

        model = LinearRegression()
        model.fit(X, y)

        predictions = model.predict(X)

        mse = mean_squared_error(y, predictions)
        accuracy = 100 - mse

        future_year = [[2026]]
        future_prediction = model.predict(future_year)

        # Graph
        plt.figure(figsize=(6,4))
        plt.scatter(X, y)
        plt.plot(X, predictions)
        plt.xlabel("Year")
        plt.ylabel("Sales")
        plt.title("Predictive Analytics")

        graph_path = 'static/graph.png'
        plt.savefig(graph_path)

        return render_template(
            'result.html',
            prediction=round(future_prediction[0], 2),
            accuracy=round(accuracy, 2),
            graph=graph_path
        )

if __name__ == '__main__':
    app.run(debug=True)