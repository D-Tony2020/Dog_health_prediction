from flask import Flask, render_template, request
import json
import numpy as np
import joblib

app = Flask(__name__)

# Print statement indicating that the model is being loaded
print("Loading the model...")
model_naive_bayes = joblib.load('naive_base_with_breedtop50.joblib')
print("Model loaded successfully.")
@app.route('/')
def home():
    return render_template('templates.html')

@app.route('/predict_health_conditions', methods=['POST'])
def predict_health_conditions():
    try:
        # Get the feature list from the form data
        feature_list = request.form.get('featureList')

        # Convert the feature list back to a Python list (assuming it's stored as a JSON string)
        feature_list = json.loads(feature_list)

        # Ensure feature_list is a valid list
        if not isinstance(feature_list, list):
            raise ValueError("Invalid feature list format")

        # Reshape the feature list to match the model input shape
        features = np.array(feature_list).reshape(1, -1)

        # Add print statements for debugging
        print("Input Features:", features)

        # Perform prediction using the loaded model
        outcome_list = model_naive_bayes.predict_proba(features)
        outcome_list = np.round(outcome_list[0], 2)
        print("Outcome List:", outcome_list)
        return render_template('result.html', feature_list=feature_list, predict=format(outcome_list.tolist()))

    except Exception as e:
        # Handle exceptions and log the error
        print("Error:", str(e))
        return "Error occurred during prediction."

if __name__ == "__main__":
    app.run(port=5000, debug=True)

