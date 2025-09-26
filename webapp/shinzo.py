import joblib
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify, render_template, send_from_directory

# Silence TensorFlow warnings
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

from keras.models import load_model  # noqa: E402

# Load the pre-trained model, logfile, and scaler
model = load_model("models/PReLU_heart_model.keras")
logfile = joblib.load("models/logs/PReLU_model_logs.pkl")
sc = joblib.load("models/scaler.pkl")

app = Flask(
    __name__,
    template_folder='src',
    static_folder='src',
    static_url_path=''
)


@app.route('/templates/<path:filename>')
def custom_static(filename):
    return send_from_directory('templates', filename)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json or request.form
    features = [
        int(data['age']),
        int(data['sex']),
        int(data['cp']),
        int(data['trestbps']),
        int(data['chol']),
        int(data['fbs']),
        int(data['restecg']),
        int(data['thalach']),
        int(data['exang']),
        float(data['oldpeak']),
        int(data['slope']),
        int(data['ca']),
    ]

    try:
        # Scale Input Data
        input_data = np.array([features])
        input_data_scaled = sc.transform(input_data)
        input_df = pd.DataFrame(input_data_scaled)

        # Select input data (GA-Feature Selection)
        selected_features = logfile['selected_features']
        selected_feature_indices = list(selected_features)
        selected_input_data = input_df.iloc[:, selected_feature_indices]

        # Model Predection
        probability = model.predict(selected_input_data)
        prediction = (probability > 0.5).astype(int)[0][0]

        if prediction == 1:
            result = "High Likelihood of Heart Disease."
        elif prediction == 0:
            result = "Low Likelihood of Heart Disease."

        return jsonify({'msg': result, 'prediction': int(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
    # Port 0.0.0.0 is to run directly on jupyter (not recommended)
    # update to localhost i.e 127.0.0.1 for production
