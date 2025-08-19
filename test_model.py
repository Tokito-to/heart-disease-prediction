from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import numpy as np

import joblib
from keras.models import load_model

# Load model and scaler
model = load_model("models/heart_model.keras")
sc = joblib.load("models/scaler.pkl")

# input data
input_data = np.array([[62,0,3,140,268,0,2,160,0,3.6,2,2,1]])

# Scale the input data
input_data_scaled = sc.transform(input_data)

# Predict the probability for the input data
y_pred_prob = model.predict(input_data_scaled)
y_pred = (y_pred_prob > 0.5).astype(int)

if y_pred[0] == 1:
    print("Prediction: High Likelihood of Heart Disease.")
elif y_pred[0] == 0:
    print("Prediction: Low Likelihood of Heart Disease.")
else:
    print("Prediction Could Not Be Determined.")
