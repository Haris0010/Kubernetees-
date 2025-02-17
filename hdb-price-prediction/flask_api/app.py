from flask import Flask, request, jsonify, render_template_string
import os
import requests
import time
import pandas as pd

app = Flask(__name__)

UPLOAD_FOLDER = '/app/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Upload Datasets</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }
            .container { max-width: 500px; margin: auto; padding: 20px; border-radius: 10px; box-shadow: 0px 0px 10px #aaa; }
            h1 { color: #333; }
            input, button { margin-top: 10px; padding: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Upload Your Datasets</h1>
            <form action="/upload" method="post" enctype="multipart/form-data">
                <label for="dataset1">Training Dataset:</label>
                <input type="file" name="dataset1" required><br><br>
                <label for="dataset2">Prediction Dataset:</label>
                <input type="file" name="dataset2" required><br><br>
                <button type="submit">Upload & Start Processing</button>
            </form>
        </div>
    </body>
    </html>
    '''

@app.route('/upload', methods=['POST'])
def upload_files():
    # Ensure both files are provided
    if 'dataset1' not in request.files or 'dataset2' not in request.files:
        return jsonify({"error": "Both training and prediction datasets are required!"}), 400

    dataset1 = request.files['dataset1']
    dataset2 = request.files['dataset2']

    if dataset1.filename == '' or dataset2.filename == '':
        return jsonify({"error": "Invalid file names!"}), 400

    dataset1_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset1.filename)
    dataset2_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset2.filename)

    dataset1.save(dataset1_path)
    dataset2.save(dataset2_path)

    # Step 1: Call Preprocessing Service
    preprocessing_response = requests.post("http://preprocessing-service:8000/process", json={
        "train_file": dataset1_path,
        "pred_file": dataset2_path
    })

    if preprocessing_response.status_code == 200:
        return jsonify({"message": "Preprocessing started. Please wait for results!"})
    else:
        return jsonify({"error": "Failed to start preprocessing!"}), 500

@app.route('/results', methods=['POST'])
def receive_results():
    """Receives final prediction results from Prediction Service."""
    data = request.json
    predictions = data.get("predictions", [])

    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Predictions</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }
            .container { max-width: 600px; margin: auto; padding: 20px; border-radius: 10px; box-shadow: 0px 0px 10px #aaa; }
            h1 { color: #333; }
            table { width: 100%; border-collapse: collapse; margin-top: 20px; }
            th, td { border: 1px solid #ddd; padding: 10px; text-align: center; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Prediction Results</h1>
            <table>
                <tr><th>Index</th><th>Predicted Value</th></tr>
                {% for i, pred in enumerate(predictions) %}
                <tr><td>{{ i+1 }}</td><td>{{ pred }}</td></tr>
                {% endfor %}
            </table>
        </div>
    </body>
    </html>
    ''', predictions=predictions)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)