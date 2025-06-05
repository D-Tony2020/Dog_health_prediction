

from flask import Flask, request, jsonify
import json
import numpy as np
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the model
print("Loading the model...")
model = joblib.load('./extreme_gradient_boosting_extend.joblib')
print("Model loaded successfully.")

breedGroup = {
    'Ancient_and_Spitz': False,
    'Australian-like': False,
    'Chihuahua': False,
    'Dachshund': False,
    'Golden': False,
    'Herding dogs - Other': False,
    'Hound': False,
    'Labs': False,
    'Mastiff-like Group 1': False,
    'Mastiff-like Group 2': False,
    'Mixed Lab and Golden': False,
    'Mixed Large': False,
    'Mixed Medium': False,
    'Mixed Other': False,
    'Mixed Small': False,
    'Shepherd': False,
    'Spaniels': False,
    'Terriers': False,
    'Toy - Other': False,
    'Working dogs - Non-sport': False,
}

subBreed = {
    'True Mastiffs': False,
    'Ancient and Spitz': False,
    'Australian': False,
    'Bulldogs': False,
    'Chihuahua': False,
    'Collie, Corgi, Sheepdog': False,
    'Dachshund': False,
    'Golden': False,
    'Golden mix': False,
    'Herding-other': False,
    'Hound': False,
    'Lab mix': False,
    'Labs': False,
    'Large': False,
    'Large Terriers': False,
    'Maltese': False,
    'Medium': False,
    'Non-sport': False,
    'Other': False,
    'Pointer': False,
    'Setter': False,
    'Shepherd': False,
    'Shepherd - Other': False,
    'Shih Tzu': False,
    'Sight Hounds': False,
    'Similar to retrievers': False,
    'Small': False,
    'Small Terriers': False,
    'Sporting': False,
    'Toy - Other': False,
}

breedType = {
    'Ancient and Spitz': False,
    'Herding dogs': False,
    'Mastiff-Like': False,
    'Mixed': False,
    'Retriever': False,
    'Scent Hounds': False,
    'Spaniels': False,
    'Terriers': False,
    'Toy dogs': False,
    'Working dogs': False,
}


# FilePath = "../Dataset/Supplementary_data/breed_grouping_HLES_extended.csv"
# df = pd.read_csv(FilePath, delimiter=",", encoding="utf-8")


FilePath = "./breed_grouping_HLES_extended.csv"

try:
    df = pd.read_csv(FilePath, delimiter=",", encoding="utf-8")
    print("CSV loaded successfully.")
except Exception as e:
    print(f"Error reading CSV file: {str(e)}")


@app.route('/predict_health_conditions', methods=['POST'])
def predict_health_conditions():
    try:
        # Get the feature list from the request data
        request_data = request.get_json()
        print(request_data)
        # Check if 'featureList' is present in the request
        if 'featureList' not in request_data:
            return jsonify({'error': 'Missing featureList in the request body'})


        #  if not isinstance(feature_list, list):
        #     raise ValueError("Invalid feature list format")

    
        # Extract feature list from the request data
        feature_list = request_data['featureList']
        print("Received Feature List:", feature_list)
        form_data = request_data['formdata']
        print("Received form data",form_data)


           # Get weight
        size = form_data['weight']
        print("Selected Size:", size)

        # Get the selected primaryBreed from the form data
        selected_primary_breed = form_data['primaryBreed']
        print("Selected Primary Breed:", selected_primary_breed)

        # Get the selected primaryBreed from the form data
        selected_secondary_breed = form_data['secondaryBreed']
        print("Selected secondary Breed:", selected_secondary_breed)

        breed_pureormix = form_data['breed_pure_or_mix']
        if  (breed_pureormix =='mixed'):
            dd_breed = determine_breed(selected_primary_breed, selected_secondary_breed)
            print("Selected dd Breed:", dd_breed)
        else:
            dd_breed = selected_primary_breed
            print("Selected dd Breed:", dd_breed)

        # Apply the classification function only when dd_breed is 'Unknown Mix'
        if dd_breed == 'Unknown Mix':
            dd_breed = classify_breed(size)
            print("Selected dd Breed:", dd_breed)


        # Find the row where dd_breed matches
        matching_row = df[df['dd_breed'] == dd_breed].iloc[0]
        # Get the dd_breed Breed Group, subBreed, breedType value from the matching row
        breed_group_value = matching_row['Breed Group']
        subBreed_value = matching_row['Sub Breed']
        breedType_value = matching_row['Breed Type']

        #if selected_primary_breed == 'Unknown' and selected_secondary_breed != 'Unknown':  pop a window to tell users to choose the known secondary breed in the primary one
        #if selected_primary_breed != 'Unknown' and selected_secondary_breed == 'Unknown': nothing happens

        if selected_primary_breed != 'Unknown' and selected_secondary_breed != 'Unknown':
            # Find the row where secondary_breed  primaryBreed matches
            matching_row_primary = df[df['dd_breed'] == selected_primary_breed].iloc[0]
            matching_row_secondary = df[df['dd_breed'] == selected_secondary_breed].iloc[0]

            # Get the primary Breed Group, subBreed, breedType value from the matching row
            breed_group_value_primary = matching_row_primary['Breed Group']
            subBreed_value_primary = matching_row_primary['Sub Breed']
            breedType_value_primary = matching_row_primary['Breed Type']

            # Get the secondary Breed Group, subBreed, breedType value from the matching row
            breed_group_value_secondary = matching_row_secondary['Breed Group']
            subBreed_value_secondary = matching_row_secondary['Sub Breed']
            breedType_value_secondary = matching_row_secondary['Breed Type']

            # Breed Group
            condition1 = (breed_group_value_primary == breed_group_value_secondary) and \
                         (breed_group_value_primary != 'Unknown') and \
                         (breed_group_value_primary != breed_group_value)
            if condition1:
                # Copy 'Primary Breed Group' to 'Breed Group' where the condition is met
                breed_group_value = breed_group_value_primary

            # Breed Type
            condition2 = (breedType_value_primary == breedType_value_secondary) and \
                         (breedType_value_primary != 'Unknown') and \
                         (breedType_value_primary != breedType_value)
            if condition2:
                # Copy 'Primary Breed Type' to 'Breed Type' where the condition is met
                breedType_value = breedType_value_primary

            # Sub Breed
            condition3 = (subBreed_value_primary == subBreed_value_secondary) and \
                         (subBreed_value_primary != 'Unknown') and \
                         (subBreed_value_primary != subBreed_value)
            if condition3:
                # Copy 'Primary subBreed' to 'subBreed' where the condition is met
                subBreed_value = subBreed_value_primary

        # Set the corresponding value in breedGroup, subBreed, breedType value to True
        breedGroup[breed_group_value] = True
        subBreed[subBreed_value] = True
        breedType[breedType_value] = True

        # Convert the  dictionary to a list
        breedGroup_list = list(breedGroup.values())
        subBreed_list = list(subBreed.values())
        breedType_list = list(breedType.values())

        # Specify the starting index (20 in this case) to insert values
        start_index_1 = 51
        # Insert values from breedGroup_list into feature_list starting from index 20
        for i, value in enumerate(breedGroup_list):
            feature_list.insert(start_index_1 + i, value)

        start_index_3 = 97
        # Insert values from breedType_list into feature_list starting from index 20
        for i, value in enumerate(breedType_list):
            feature_list.insert(start_index_3 + i, value)

        start_index_2 = 107
        # Insert values from subBreed_list into feature_list starting from index 20
        for i, value in enumerate(subBreed_list):
            feature_list.insert(start_index_2 + i, value)

        # Reshape the feature list to match the model input shape
        features = np.array(feature_list).reshape(1, -1)
        print('Hello',feature_list)
        print("THis is length", len(feature_list))

        # Perform prediction using the loaded model
        outcome_list = model.predict_proba(features)
        outcome_list = np.round(outcome_list[0], 6)
        print("Outcome List:", outcome_list)
        return jsonify({'prediction': outcome_list.tolist()})

    except Exception as e:
        # Handle exceptions and log the error
        print("Error:", str(e))
        return "Error occurred during prediction."


def determine_breed(selected_primary_breed, selected_secondary_breed):
    if selected_primary_breed != 'Unknown':
        return f"{selected_primary_breed} Mix"
    elif selected_primary_breed == 'Unknown' and selected_secondary_breed != 'Unknown':
        return f"{selected_secondary_breed} Mix"
    elif selected_primary_breed == 'Unknown' and selected_secondary_breed == 'Unknown':
        return "Unknown Mix"
    else:
        pass


def classify_breed(size):
    if size == 'small':
        return 'Mixed Breed Small (0 - 20lbs)'
    elif size == 'medium':
        return 'Mixed Breed Medium (21 - 60lbs)'
    elif size == 'large':
        return 'Mixed Breed Large (61lbs+)'
    else:
        return 'Unknown Mix'


    #     # Reshape the feature list to match the model input shape
    #     features = np.array(feature_list).reshape(1, -1)

    #     # Perform prediction using the loaded model
    #     outcome_list = model_naive_bayes.predict_proba(features)
    #     outcome_list = np.round(outcome_list[0], 6)

    #     # Log prediction result for debugging
    #     print("Outcome List:", outcome_list)

    #     # Return the prediction as JSON
    #     return jsonify({'prediction': outcome_list.tolist()})

    # except Exception as e:
    #     # Handle exceptions and log the error
    #     print("Error:", str(e))
    #     return jsonify({'error': 'Error occurred during prediction. ' + str(e)})

if __name__ == "__main__":
    app.run(port=5000, debug=True)
