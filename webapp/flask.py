from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__, template_folder='src', static_folder='src', static_url_path='')

@app.route('/templates/<path:filename>')
def custom_static(filename):
    return send_from_directory('templates', filename)

# Load model and scaler
import joblib
from keras.models import load_model
model = load_model("models/ReLU_heart_model.keras")
sc = joblib.load("models/scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json or request.form

    try:
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

        input_data = np.array([features])
        input_scaled = sc.transform(input_data)
        probability = model.predict(input_scaled)
        prediction = (probability > 0.5).astype(int)[0][0]

        if prediction == 1:
            result = "High Likelihood of Heart Disease."
        elif prediction == 0:
            result = "Low Likelihood of Heart Disease."

        return jsonify({ 'msg': result, 'prediction': int(prediction) })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
