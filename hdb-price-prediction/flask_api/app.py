from flask import Flask, request, render_template_string, jsonify
import os
import pandas as pd
import logging
from kubernetes import client, config
import time
import requests


app = Flask(__name__)

UPLOAD_FOLDER = '/app/data'
MODEL_SAVE_PATH = os.getenv('MODEL_SAVE_PATH')
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
        return "No files found", 400
    dataset1 = request.files['dataset1']
    dataset2 = request.files['dataset2']
    if dataset1.filename == '' or dataset2.filename == '':
        return "No files selected", 400
    dataset1_path = os.path.join(app.config['UPLOAD_FOLDER'], "sg-resale-flat-prices.csv")
    dataset2_path = os.path.join(app.config['UPLOAD_FOLDER'], "prediction_data.csv")
    dataset1.save(dataset1_path)
    dataset2.save(dataset2_path)

    try:

        # Send POST requests for each process
        preprocessing_response = requests.post("http://preprocessing-service:8000/process")
        if preprocessing_response.status_code != 200:
            return f"Error in Preprocessing: {preprocessing_response.text}", 500
        
        model_training_response = requests.post("http://training-service:8001/train")
        if model_training_response.status_code != 200:
            return f"Error in Model Training: {model_training_response.text}", 500

        eval_response = requests.post("http://evaluation-service:8002/evaluate")
        if eval_response.status_code != 200:
            return f"Error in Evaluation: {eval_response.text}", 500

        prediction_response = requests.post("http://prediction-service:8003/predict")
        if prediction_response.status_code != 200:
            return f"Error in Prediction: {prediction_response.text}", 500
        
        get_predictions_response = requests.get("http://prediction-service:8003/get_predictions")
        if get_predictions_response.status_code == 200:
            df = pd.read_csv(f'{MODEL_SAVE_PATH}predictions_output.csv')
            rows = df.values.tolist()
            headers = df.columns.tolist()

            html_content = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>CSV Data</title>
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
                        max-width: 800px;
                        max-height: 70vh;
                        overflow-y: auto; /* Added vertical scroll */
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
                    table {
                        width: 100%;
                        border-collapse: collapse;
                        margin-top: 20px;
                    }
                    table, th, td {
                        border: 1px solid #ff7597;
                    }
                    th, td {
                        padding: 12px;
                        text-align: left;
                    }
                    th {
                        background-color: #ff9a9e;
                        color: white;
                    }
                    td {
                        background-color: #fffaf0;
                    }
                    tr:hover {
                        background-color: #f8d8da;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>CSV Data</h1>
                    <table>
                        <thead>
                            <tr>
                                {% for header in headers %}
                                    <th>{{ header }}</th>
                                {% endfor %}
                            </tr>
                        </thead>
                        <tbody>
                            {% for row in rows %}
                                <tr>
                                    {% for cell in row %}
                                        <td>{{ cell }}</td>
                                    {% endfor %}
                                </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </body>
            </html>
            """
            # Render the HTML content with the data
            return render_template_string(html_content, rows=rows, headers=headers)
        else:
            return f"Error in Displaying Predictions: {get_predictions_response.text}", 500

    except Exception as e:
        logging.error(f"Error during file upload and processing: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
