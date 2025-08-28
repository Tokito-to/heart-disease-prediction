from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import numpy as np
import argparse

import joblib
from keras.models import load_model

parser = argparse.ArgumentParser(description="Heart Disease Prediction")
parser.add_argument('-m', '--model', type=str, required=True, help='Path to the trained model (.keras)')
parser.add_argument('-s', '--scaler', type=str, required=True, help='Path to the scaler (.pkl)')
args = parser.parse_args()

# Load model and scaler
model = load_model(args.model)
sc = joblib.load(args.scaler)

# input data
input_data = np.array([[65,0,2,160,360,0,2,151,0,0.8,0,0]])

# Scale the input data
input_data_scaled = sc.transform(input_data)

# Predict the probability for the input data
y_pred_prob = model.predict(input_data_scaled)
y_pred = (y_pred_prob > 0.5).astype(int)

# color codes
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

if y_pred[0] == 1:
    print(f"{RED}Prediction: High Likelihood of Heart Disease.{RESET}")
elif y_pred[0] == 0:
    print(f"{GREEN}Prediction: Low Likelihood of Heart Disease.{RESET}")
else:
    print(f"{YELLOW}Prediction Could Not Be Determined.{RESET}")

