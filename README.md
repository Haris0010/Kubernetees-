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
minikube service flask-api-service --url
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
