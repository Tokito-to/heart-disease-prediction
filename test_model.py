import joblib
import argparse

import numpy as np
import pandas as pd

from silence_tensorflow import silence_tensorflow
silence_tensorflow()

from keras.models import load_model  # noqa: E402

parser = argparse.ArgumentParser(description="Pre-Trained Model Test")
parser.add_argument(
    '-m',
    '--modelfile',
    type=str,
    required=True,
    help='Path to the trained modelfile (.keras)'
)
parser.add_argument(
    '-l',
    '--logfile',
    type=str,
    required=True,
    help='Path to modelfile logfile (.pkl)'
)
parser.add_argument(
    '-s',
    '--scaler',
    type=str,
    required=True,
    help='Path to the scaler (.pkl)'
)
args = parser.parse_args()

# Load modelfile and scaler
modelfile = load_model(args.modelfile)
logfile = joblib.load(args.logfile)
sc = joblib.load(args.scaler)

# input data
input_data = np.array([[65, 0, 2, 160, 360, 0, 2, 151, 0, 0.8, 0, 0]])

# Scale the input data
input_data_scaled = sc.transform(input_data)
input_df = pd.DataFrame(input_data_scaled)

# Select input data (GA-Feature Selection)
selected_features = logfile['selected_features']
selected_feature_indices = list(selected_features)
selected_input_data = input_df.iloc[:, selected_feature_indices]

# Predict the probability for the input data
y_pred_prob = modelfile.predict(selected_input_data)
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
