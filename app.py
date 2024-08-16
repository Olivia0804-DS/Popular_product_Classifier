from flask import Flask, request, jsonify,render_template, redirect, url_for
import pandas as pd
import json
import numpy as np
import pickle


with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)


# Define upload folder
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save the file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Read the CSV file
        input_data = pd.read_csv(filepath)

        # Extract product titles
        product_titles = input_data['title']  # Replace with your actual column name

        # Predict using the loaded model
        predictions = model.predict(input_data)

        # Determine popularity
        popularity = ['Popular' if pred == 1 else 'Not Popular' for pred in predictions]

        # Prepare results
        results = list(zip(product_titles, popularity))

        return render_template('result.html', results=results)

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
