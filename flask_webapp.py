from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__, template_folder='webapp_src', static_folder='webapp_src', static_url_path='')

@app.route('/templates/<path:filename>')
def custom_static(filename):
    return send_from_directory('templates', filename)

# Load model and scaler
import joblib
from keras.models import load_model
model = load_model("models/heart_model.keras")
sc = joblib.load("models/scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json or request.form

    try:
        features = [
            float(data['age']),
            float(data['sex']),
            float(data['cp']),
            float(data['trestbps']),
            float(data['chol']),
            float(data['fbs']),
            float(data['restecg']),
            float(data['thalach']),
            float(data['exang']),
            float(data['oldpeak']),
            float(data['slope']),
            float(data['ca']),
            float(data['thal'])
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
    app.run(host='0.0.0.0', port=5000, debug=True)
