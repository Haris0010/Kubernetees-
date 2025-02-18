from flask import Flask, request, redirect, url_for, render_template_string, jsonify
import os
import pandas as pd
import logging
from kubernetes import client, config
import time
import requests

app = Flask(__name__)
UPLOAD_FOLDER = '/app/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

logging.basicConfig(level=logging.INFO)

# Load Kubernetes configuration inside cluster
config.load_incluster_config()
apps_v1 = client.AppsV1Api()
core_v1 = client.CoreV1Api()

def scale_deployment(name, replicas):
    body = {'spec': {'replicas': replicas}}
    apps_v1.patch_namespaced_deployment_scale(name, 'default', body)
    logging.info(f'Scaled {name} to {replicas} replicas.')

def wait_for_pod(label):
    while True:
        pods = core_v1.list_namespaced_pod('default', label_selector=f'app={label}')
        for pod in pods.items:
            if pod.status.phase in ['Succeeded', 'Running']:
                logging.info(f'{label} deployment completed.')
                return
        time.sleep(5)

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
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
            body {
                font-family: 'Poppins', sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                background: linear-gradient(135deg, #fbc2eb, #a6c1ee, #ffdde1, #c1f0d6, #fdd9b5);
                background-size: 300% 300%;
                animation: smoothBackground 7s infinite linear;
            }
            @keyframes smoothBackground {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }
            .container {
                padding: 40px;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
                text-align: center;
                max-width: 500px;
                animation: fadeIn 1s ease-in-out;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(-15px); }
                to { opacity: 1; transform: translateY(0); }
            }
            h1 {
                color: #333;
                margin-bottom: 25px;
                font-size: 28px;
                font-weight: 700;
                background: linear-gradient(90deg, #ff9a9e, #fad0c4);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-shadow: 2px 2px 8px rgba(255, 65, 108, 0.3);
            }
            form {
                display: flex;
                flex-direction: column;
                gap: 20px;
            }
            label {
                font-weight: 600;
                color: #ff7597;
                margin-bottom: 5px;
                display: block;
                font-size: 18px;
            }
            input[type="file"] {
                padding: 14px;
                border: 2px solid #ff7597;
                border-radius: 10px;
                background: #fffaf0;
                cursor: pointer;
                transition: 0.3s;
                font-size: 16px;
            }
            input[type="file"]:hover {
                border-color: #ff7597;
                transform: scale(1.05);
                box-shadow: 0px 5px 10px rgba(255, 117, 151, 0.3);
            }
            button {
                padding: 16px;
                border: none;
                border-radius: 10px;
                background: linear-gradient(135deg, #ff9a9e, #fad0c4);
                color: white;
                font-size: 22px;
                font-weight: 700;
                cursor: pointer;
                transition: 0.3s;
                box-shadow: 0 10px 25px rgba(255, 117, 151, 0.3);
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            button:hover {
                background: linear-gradient(135deg, #fad0c4, #ff9a9e);
                transform: scale(1.15);
                box-shadow: 0 15px 30px rgba(255, 117, 151, 0.5);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Upload Your Datasets</h1>
            <form action="/upload" method="post" enctype="multipart/form-data">
                <label for="dataset1">Training Dataset:</label>
                <input type="file" name="dataset1" id="dataset1" required>
                <label for="dataset2">Prediction Dataset:</label>
                <input type="file" name="dataset2" id="dataset2" required>
                <button type="submit">Upload & Preview</button>
            </form>
        </div>
    </body>
    </html>
    '''

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'dataset1' not in request.files or 'dataset2' not in request.files:
        return redirect(request.url)
    dataset1 = request.files['dataset1']
    dataset2 = request.files['dataset2']
    if dataset1.filename == '' or dataset2.filename == '':
        return redirect(request.url)
    dataset1_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset1.filename)
    dataset2_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset2.filename)
    dataset1.save(dataset1_path)
    dataset2.save(dataset2_path)

    try:
        scale_deployment('preprocessing-deployment', 1)
        wait_for_pod('preprocessing')
        scale_deployment('training-deployment', 1)
        wait_for_pod('training')
        scale_deployment('evaluation-deployment', 1)
        wait_for_pod('evaluation')
        scale_deployment('prediction-deployment', 1)
        wait_for_pod('prediction')

        response = requests.get('http://prediction-service:8000/get_predictions')
        predictions = response.json().get('predictions', [])
        return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Predictions</title>
            <style>
                body { font-family: 'Poppins', sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; background: linear-gradient(135deg, #a1c4fd, #c2e9fb); margin: 0; }
                .container { padding: 40px; background: white; border-radius: 20px; box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1); text-align: center; }
                h1 { color: #333; font-size: 26px; margin-bottom: 15px; }
                table { margin-top: 20px; width: 100%; border-collapse: collapse; }
                th, td { padding: 12px; border: 1px solid #ddd; text-align: center; }
                th { background: #a1c4fd; color: white; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Prediction Results</h1>
                <table>
                    <tr><th>Index</th><th>Predicted Value</th></tr>
                    {% for i in range(predictions|length) %}
                    <tr><td>{{ i+1 }}</td><td>{{ predictions[i] }}</td></tr>
                    {% endfor %}
                </table>
            </div>
        </body>
        </html>
        ''', predictions=predictions)
    except Exception as e:
        logging.error(f"Error in upload: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)