"""
Real-time inference — sends a wine sample to the MLflow model server (port 5001).

Requires model server running:
    export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
    mlflow models serve -m "models:/wine_quality/production" -p 5001 --env-manager=local

Usage: python3 inference.py
"""

import requests

url = 'http://localhost:5001/invocations'

#data_dict = {"dataframe_split": X_test.to_dict(orient='split')}

data_dict = {
    "dataframe_split": {
        "columns": [
            "fixed_acidity", "volatile_acidity", "citric_acid",
            "residual_sugar", "chlorides", "free_sulfur_dioxide",
            "total_sulfur_dioxide", "density", "pH", "sulphates",
            "alcohol", "is_red"
        ],
        "data": [[7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.0000009978, 3.51, 0.56, 1000.4, 1]]
    }
}

response = requests.post(url, json=data_dict)
prediction = response.json()["predictions"][0]

# print density value
print(f"Density: {data_dict['dataframe_split']['data'][0][7]}")
print(f"Alcohol: {data_dict['dataframe_split']['data'][0][10]}")
print(f"Probability of high quality: {prediction:.4f}")
print(f"Prediction: {'High Quality' if prediction >= 0.5 else 'Not High Quality'}")
