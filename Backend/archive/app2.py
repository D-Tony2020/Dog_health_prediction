

from flask import Flask, request, jsonify
import json
import numpy as np
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the model
print("Loading the model...")
model_naive_bayes = joblib.load('./naive_base_with_breedtop50.joblib')
print("Model loaded successfully.")

@app.route('/predict_health_conditions', methods=['POST'])
def predict_health_conditions():
    try:
        # Get the feature list from the request data
        request_data = request.get_json()

        # Check if 'featureList' is present in the request
        if 'featureList' not in request_data:
            return jsonify({'error': 'Missing featureList in the request body'})

        # Extract feature list from the request data
        feature_list = request_data['featureList']

        # Log received data for debugging
        print("Received Feature List:", feature_list)

        # Ensure feature_list is a valid list
        if not isinstance(feature_list, list):
            raise ValueError("Invalid feature list format")

        # Reshape the feature list to match the model input shape
        features = np.array(feature_list).reshape(1, -1)

        # Perform prediction using the loaded model
        outcome_list = model_naive_bayes.predict_proba(features)
        outcome_list = np.round(outcome_list[0], 6)

        # Log prediction result for debugging
        print("Outcome List:", outcome_list)

        # Return the prediction as JSON
        return jsonify({'prediction': outcome_list.tolist()})

    except Exception as e:
        # Handle exceptions and log the error
        print("Error:", str(e))
        return jsonify({'error': 'Error occurred during prediction. ' + str(e)})

if __name__ == "__main__":
    app.run(port=5000, debug=True)
