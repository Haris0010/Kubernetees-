# HDB Price Prediction - Kubernetes AI Pipeline

This project deploys an AI pipeline in Kubernetes, handling data preprocessing, model training, evaluation, prediction, and a Flask web UI.

---

## Project Directory Structure
```bash
hdb-price-prediction/
│── evaluation/
│   ├── Dockerfile
│   ├── evaluate.py
│   ├── requirements.txt
│
│── flask_api/
│   ├── Dockerfile
│   ├── app.py
│   ├── requirements.txt
│
│── kubernetes/
│   ├── configmap.yaml
│   ├── evaluate-deployment.yaml
│   ├── evaluate-service.yaml
│   ├── flask-upload-deployment.yaml
│   ├── flask-upload-service-cluster.yaml
│   ├── flask-upload-service-node.yaml
│   ├── kustomization.yaml
│   ├── prediction-deployment.yaml
│   ├── prediction-service.yaml
│   ├── preprocessing-deployment.yaml
│   ├── preprocessing-service.yaml
│   ├── pv.yaml
│   ├── pvc.yaml
│   ├── role.yaml
│   ├── rolebinding.yaml
│   ├── training-deployment.yaml
│   ├── training-service.yaml
│
│── model_training/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── train.py
│
│── prediction/
│   ├── Dockerfile
│   ├── predict.py
│   ├── requirements.txt
│
│── preprocessing/
│   ├── Dockerfile
│   ├── preprocess.py
│   ├── requirements.txt
```

---

## Overview of Components

| **Component**      | **Functionality** |
|--------------------|------------------|
| ``preprocessing``  | Cleans, encodes, and preprocesses the dataset. |
| ``model_training`` | Trains machine learning models. |
| ``evaluation``     | Evaluates and selects the best-performing model. |
| ``prediction``     | Loads the best model and makes predictions. |
| ``flask_api``      | Web UI for data upload and pipeline execution. |
| ``kubernetes``     | Contains Kubernetes YAML files for deployments and services. |

---

## System Architecture

![](./Images/system-architecture.png "System Architecture")

From the flask webite, users will upload the data and it will get stored inside a persistent storage (PS), the preprocessing container will then retrieve the uploaded data from PS and carry out preprocessing. 

After preprocessing, preprocessed data will then be stored inside the PS. The training container then takes the preprocessed data from the PS, trains each model and stores all the model inside the PS. 

The evaluation container then takes the different models inside the PS and gives an output of the best saved model from the PS.

The prediction container then takes the best saved model and preprocessed prediction data from the PS. The container will produce an output prediction csv which is stored in the PS as well as showed in the flask webite.

## Deployment Details

### 1. Preprocessing Container
* **Function:** Reads raw data, cleans it, encodes categorical features, and saves processed data.
* **Deployment File:** <code>preprocessing-deployment.yaml</code>
* **Service File:** <code>preprocessing-service.yaml</code>
* **Exposed Port:** <code>8000</code>
* **Docker Image:** <code>hhym123/preprocessing:v5.0</code>

### 2. Training Container
* **Function:** Trains machine learning models (RandomForest, LinearRegression, GradientBoosting) and saves them as <code>.joblib</code> files.
* **Deployment File:** <code>training-deployment.yaml</code>
* **Service File:** <code>training-service.yaml</code>
* **Exposed Port:** <code>8001</code>
* **Docker Image:** <code>walkingduck/training:v405</code>

### 3. Evaluation Container
* **Function:** Evaluates trained models using r2_score, selects the best one, and saves it.
* **Deployment File:** <code>evaluate-deployment.yaml</code>
* **Service File:** <code>evaluate-service.yaml</code>
* **Exposed Port:** <code>8002</code>
* **Docker Image:** <code>walkingduck/evaluation:v7</code>

### 4. Prediction Container
* **Function:** Loads the best model and performs predictions on input data.
* **Deployment File:** <code>prediction-deployment.yaml</code>
* **Service File:** <code>prediction-service.yaml</code>
* **Exposed Port:** <code>8003</code>
* **Docker Image:** <code>hhym123/predict:v5.0</code>

### 5. Flask API Container
* **Function:** Provides a web UI for uploading datasets and triggering pipeline execution.
* **Deployment File:** <code>flask-upload-deployment.yaml</code>
* **Service File:**
  * <code>flask-upload-service-cluster.yaml</code> (Internal access)
  * <code>flask-upload-service-node.yaml</code> (External access)
* **Exposed Port:** <code>5000</code> (Externally accessible via <code>NodePort: 30007</code>)
* **Docker Image:** <code>hhym123/flask:v17.0</code>

## Persistent Storage
* The pipeline uses Persistent Volumes (PV) and Persistent Volume Claims (PVC) for storing data.
* **PV File:** <code>pv.yaml</code> (Storage Location: <code>/mnt/data/my-storage</code>)
* **PVC File:** <code>pvc.yaml</code>

## Configuration & Permissions
* **Configuration Map:** <code>configmap.yaml</code> (Stores file paths and model storage location)
* **Role & RoleBinding:**
  * <code>role.yaml</code> (Grants permission to scale deployments and list/watch pods)
  * <code>rolebinding.yaml</code> (Binds permissions to service accounts)


## How to Deploy

### 1. Start Minikube
```sh
minikube start
```

### 2. Set Up Persistent Storage
```sh
minikube ssh
mkdir -p /mnt/data/my-storage
sudo chmod 777 /mnt/data/my-storage
exit
```

### 3. Apply Kubernetes Configurations
Navigate to the Kubernetes directory:
```sh
cd hdb-price-prediction/kubernetes
kubectl apply -k .
```

### 4. Check if all pods are running
```sh
kubectl get pods
```

### 5. Access the Flask API
```sh
minikube service flask-api-service-node --url
```
Click the generated URL to open the Flask web UI.

---

## Scaling & Monitoring

- Scale a deployment (e.g., training container to 2 replicas):
  ```sh
  kubectl scale deployment training-deployment --replicas=2
  ```

- Monitor logs for a container:
  ```sh
  kubectl logs -f <pod-name>
  ```

- Delete all deployments & services:
  ```sh
  kubectl delete -f kubernetes/
  ```

---

## Future Improvements

- Add GPU acceleration for faster training.
- Implement weighted voting in model selection.
- Automate hyperparameter tuning.

---
