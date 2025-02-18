from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('../models/fraud_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Validate input
    data = request.json
    if not data or 'features' not in data:
        return jsonify({"error": "Invalid input format"}), 400
    
    # Convert to DataFrame
    features = pd.DataFrame([data['features']])
    
    # Make prediction
    try:
        proba = model.predict_proba(features)[0][1]
        return jsonify({
            "fraud_probability": float(proba),
            "is_fraud": proba > 0.5  # Threshold at 0.5
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)