import pandas as pd
import os
import numpy as np
import pandas as pd
import joblib
from functools import reduce

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.multiclass import OneVsRestClassifier
#from skmultilearn.model_selection import MultiLabelStratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# Models
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from helper import *

# breed
feat_breed = pd.read_pickle('../features/breed.pkl')
feat_breed_v2 = pd.read_pickle('../features/breed_v2.pkl')

feat_breed_top50 = pd.read_pickle('../features/breed_top50.pkl')
feat_breed_top50_v2 = pd.read_pickle('../features/breed_top50_v2.pkl')

feat_breed_group = pd.read_pickle('../features/breed_group.pkl')
feat_sub_breed = pd.read_pickle('../features/sub_breed.pkl')
feat_breed_type = pd.read_pickle('../features/breed_type.pkl')

feat_breed_pure_or_mix = pd.read_pickle('../features/breed_pure_or_mix.pkl')

# age
feat_age = pd.read_csv('../features/age_with_id.csv')

# sex
feat_sex = pd.read_csv('../features/one_hot_encoded_sex_with_id.csv')

# weight
feat_weight = pd.read_pickle('../features/weight.pkl')

# Climate
# feat_HotWeather = pd.read_csv('../features/one_hot_encoded_HotWheater_with_id.csv')
# feat_ModerateWeather = pd.read_csv('../features/one_hot_encoded_ModerateWheather_with_id.csv')
# feat_ColdWeater_with_id = pd.read_csv('../features/one_hot_encoded_ColdWheater_with_id.csv')
feat_cold_month = pd.read_pickle('../features/ColdMonths.pkl')
feat_hot_month = pd.read_pickle('../features/HotMonths.pkl')

# Diet
feat_diet = pd.read_pickle('../features/diet.pkl') # 5 columns

# Physical Activity
feat_pa_total_hours = pd.read_csv('../features/PhysicalActivity_total_hours.csv')
feat_pa_surface = pd.read_csv('../features/PhysicalActivity_surface.csv')
feat_pa_weather = pd.read_csv('../features/PhysicalActivity_wheather.csv')

# Owner Demographics
feat_od_income = pd.read_pickle('../features/od_income.pkl')
feat_od_education = pd.read_pickle('../features/od_education.pkl')

# Residentual
feat_prim_census_division = pd.read_pickle('../features/primary_residence_census_division.pkl')

# disease
#feat_disease_input = pd.read_csv('../features/one_hot_encoded_disease_input.csv')
feat_disease_output_binary = pd.read_csv('../features/disease_output_binary.csv')
feat_disease_output = pd.read_csv('../features/disease_output.csv')

# age_condition
#feat_age_condition = pd.read_pickle('../features/age_condition.pkl')
feat_age_condition_type = pd.read_pickle('../features/age_condition_type.pkl')


print("loaded features")


features_list = [

    # breed
    #feat_breed,
    #feat_breed_v2,
    #feat_breed_top50,
    #feat_breed_top50_v2,
    #feat_breed_group,
    #feat_sub_breed,
    #feat_breed_type,
    #feat_breed_pure_or_mix,

    # age
    #feat_age,   # 24881 dog_ids

    # sex
    #feat_sex,

    # weight
    #feat_weight

    # diet
    #feat_diet,  # 33141 dog_ids for df_diet_consistency

    #feat_HotWeater,
    #feat_ModerateWeather,
    #feat_ColdWeater_with_id,

    # Climate
    #feat_pa_surface,
    #feat_pa_total_hours,
    #feat_pa_weather,   # 26406 dog_ids
    #feat_hot_month,
    #feat_cold_month,

    # Physical Activity
    #feat_pa_total_hours,
    #feat_pa_surface,

    # Owner demographics
    #feat_od_income,    # 29096 dog_ids
    #feat_od_education

    # Residentual
    feat_prim_census_division,

    # disease
    #feat_disease_input,
    #feat_disease_output_binary,
    #feat_disease_output
]



exp1 = []
exp2 = [feat_breed_top50_v2]
exp3 = [feat_breed_group]
exp4 = [feat_breed_type, feat_sub_breed, feat_breed_pure_or_mix]
exp5 = [feat_breed_top50_v2, feat_breed_group, feat_breed_type, feat_sub_breed, feat_breed_pure_or_mix,
        feat_age]
exp6 = [feat_breed_top50_v2, feat_breed_group, feat_breed_type, feat_sub_breed, feat_breed_pure_or_mix,
        feat_age, feat_sex]
exp7 = [feat_breed_top50_v2, feat_breed_group, feat_breed_type, feat_sub_breed, feat_breed_pure_or_mix,
        feat_age, feat_weight]
exp8 = [feat_breed_top50_v2, feat_breed_group, feat_breed_type, feat_sub_breed, feat_breed_pure_or_mix,
        feat_age, feat_sex, feat_weight, feat_hot_month, feat_cold_month]
exp9 = [feat_breed_top50_v2, feat_breed_group, feat_breed_type, feat_sub_breed, feat_breed_pure_or_mix,
        feat_age, feat_sex, feat_weight, feat_prim_census_division]
exp10 = [feat_breed_top50_v2, feat_breed_group, feat_breed_type, feat_sub_breed, feat_breed_pure_or_mix,
         feat_age, feat_weight, feat_cold_month, feat_hot_month, feat_prim_census_division]
exp11 = [feat_breed_top50_v2, feat_breed_group, feat_breed_type, feat_sub_breed, feat_breed_pure_or_mix,
         feat_age, feat_sex, feat_weight,
         feat_cold_month, feat_hot_month,
         feat_prim_census_division,
         feat_pa_surface, feat_pa_total_hours, 
         feat_pa_weather
         ]
exp12 = [feat_breed_top50_v2, feat_breed_group, feat_breed_type, feat_sub_breed, feat_breed_pure_or_mix,
         feat_age, feat_sex, feat_weight,
         feat_cold_month, feat_hot_month,
         feat_prim_census_division,
         feat_diet]
exp13 = [feat_breed_top50_v2, feat_breed_group, feat_breed_type, feat_sub_breed, feat_breed_pure_or_mix,
         feat_age, feat_sex, feat_weight,
         feat_cold_month, feat_hot_month,
         feat_prim_census_division,
         feat_pa_surface, feat_pa_total_hours, 
         feat_pa_weather,
         feat_diet
         ]
exp14 = [feat_breed_top50_v2, feat_breed_group, feat_breed_type, feat_sub_breed, feat_breed_pure_or_mix,
         feat_age, feat_sex, feat_weight,
         feat_cold_month, feat_hot_month,
         feat_prim_census_division,
         feat_pa_surface, feat_pa_total_hours, feat_pa_weather,
         feat_diet,
         feat_od_education, feat_od_income]

experiments = {
    #'exp1': exp1,
    #'exp2': exp2,
    #'exp3': exp3,
    #'exp4': exp4,
    #'exp5': exp5,
    #'exp6': exp6,
    #'exp7': exp7,
    #'exp8': exp8,
    #'exp9': exp9,
    #'exp10': exp10,
    'exp11': exp11,
    #'exp12': exp12,
    'exp13': exp13,
    'exp14': exp14,
    #'exp15': exp15,
    #'exp16': exp16,
    #'exp17': exp17,
    #'exp18': exp18,
    #'exp19': exp19,
    #'exp20': exp20,
    #'exp21': exp21
}


results_auc_train = {}
results_auc_val = {}
results_accuracy = {}
results_auc_train_vega = {}
results_auc_val_vega = {}
results_accuracy_vega = {}

for exp_name, exp_features in experiments.items():

    print("======================================================================================")
    print("======================================================================================")
    print(f"{exp_name}================================================================================")
    print("======================================================================================")
    print("======================================================================================")

    # List of DataFrames to be merged
    list_input_features = features_list  # Add more DataFrames as needed

    # Merge DataFrames iteratively using reduce
    input_features = reduce(lambda left, right: pd.merge(left, right, on='dog_id'), exp_features)

    # Merge with disease output feature
    # data = pd.merge(input_features, feat_disease_output_binary, on='dog_id')
    data = pd.merge(feat_age_condition_type, input_features, on='dog_id')

    # Normalize the specified features in one line and save in the same columns
    features_to_normalize = ['dd_weight_lbs', 'age_diagnosis_years', 'rounded_age_years', 'df_diet_consistency', 'df_feedings_per_day', 'total_active_hours_y', 'pa_hot_weather_months_per_year', 'pa_cold_weather_months_per_year', 'od_max_education', 'od_annual_income_range_usd', 'pa_moderate_weather_daily_hours_outside', 'pa_cold_weather_daily_hours_outside', 'pa_cold_weather_daily_hours_outside']
    
    # select features that are in data
    features_to_normalize_in_data = [feature for feature in features_to_normalize if feature in data.columns]

    # Initialize MinMaxScaler
    scaler = MinMaxScaler()

    # Normalize
    if features_to_normalize_in_data:
        data[features_to_normalize_in_data] = scaler.fit_transform(data[features_to_normalize_in_data])
        # print(data[features_to_normalize_in_data])

    X = data.drop(['dog_id'] + ['condition_type_' + condition_type for condition_type in [
    'Eye', 'Ear/Nose/Throat', 'Mouth/Dental/Oral', 'Skin', 'Cardiac', 'Respiratory',
    'Gastrointestinal', 'Liver/Pancreas', 'Kidney/Urinary', 'Reproductive', 'Bone/Orthopedic',
    'Brain/Neurologic', 'Endocrine', 'Hematopoietic', 'Other Congenital Disorder',
    'Infection/Parasites', 'Toxin Consumption', 'Trauma', 'Immune-mediated', 'cancer'
    ]], axis=1)

    y_columns = ['condition_type_' + condition_type for condition_type in [
    'Eye', 'Ear/Nose/Throat', 'Mouth/Dental/Oral', 'Skin', 'Cardiac', 'Respiratory',
    'Gastrointestinal', 'Liver/Pancreas', 'Kidney/Urinary', 'Reproductive', 'Bone/Orthopedic',
    'Brain/Neurologic', 'Endocrine', 'Hematopoietic', 'Other Congenital Disorder',
    'Infection/Parasites', 'Toxin Consumption', 'Trauma', 'Immune-mediated', 'cancer'
    ]]

    y = data[y_columns]

    # Convert y to a binary format
    y_binary = (y == 1)

    ############################################################################################################################
    ######## SELECT MODEL ######################################################################################################
    ############################################################################################################################

    #select_model = "Naive Bayes"
    #select_model = "Logistic Regression"
    #select_model = "MLP"
    #select_model = "Gradient Boosting"
    select_model = "Extreme Gradient Boosting"
    #select_model = "Random Forest"


    if select_model=="Naive Bayes":
    # ========== Naive Bayes ========
        # Initialize the Naive Bayes model
        nb_model = MultinomialNB()

        # Wrap the model with OneVsRestClassifier
        ovr_classifier = OneVsRestClassifier(nb_model)

    elif select_model=="Logistic Regression":
        # Initialize the Logistic Regression model
        lr_model = LogisticRegression()

        # Wrap the model with OneVsRestClassifier
        ovr_classifier = OneVsRestClassifier(lr_model)

    elif select_model=="MLP":
    # ========= MLP ==========
        # normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        num_features = X.shape[1]
        if num_features <= 100:
                hidden_layer_sizes = (60,)
        elif num_features <= 200:
                hidden_layer_sizes = (60, 40)
        else:
            hidden_layer_sizes = (80, 60, 40)

        # Initialize the MLP model
        mlp_model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=1000, alpha=0.01, early_stopping=True, random_state=42)
        # Wrap the model with OneVsRestClassifier
        ovr_classifier = OneVsRestClassifier(mlp_model)

    elif select_model=="Gradient Boosting":
        model = GradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.1,
            n_estimators=100,
            subsample=0.8,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=None,
            random_state=42
        )

        # Wrap the model with OneVsRestClassifier
        ovr_classifier = OneVsRestClassifier(model)

    elif select_model=="Extreme Gradient Boosting":
        print("selected Extreme Gradient Boosting")
        # Initialize the Extreme Gradient Boosting model
        #model = xgb.XGBClassifier()
        model = xgb.XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, reg_lambda=1, reg_alpha=0.1, subsample=0.8, colsample_bytree=0.8)


        # Wrap the model with OneVsRestClassifier
        ovr_classifier = OneVsRestClassifier(model)

    elif select_model=="Random Forest":
        # Create a Random Forest Classifier
        model = RandomForestClassifier(
            n_estimators=100,
            #max_depth=None,  # You can set this to a specific value if needed
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',  # You can adjust this based on your data
            random_state=42  # Set a random state for reproducibility
        )

        # Wrap the model with OneVsRestClassifier
        ovr_classifier = OneVsRestClassifier(model)

    # Initialize MultilabelStratifiedKFold
    n_splits = 5  # You can adjust the number of splits as needed
    ml_stratified_kfold = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Lists to store AUC scores and sample counts
    auc_scores_train_per_condition = {condition: [] for condition in y_columns}
    auc_scores_val_per_condition = {condition: [] for condition in y_columns}
    sample_counts_train_per_condition = {condition: [] for condition in y_columns}
    sample_counts_val_per_condition = {condition: [] for condition in y_columns}
    auc_scores_val_per_fold = []
    auc_scores_train_per_fold = []

    accuracy_scores_per_fold = []
    auc_scores_per_fold_train = []
    auc_scores_per_fold_val = []
    # Lists to store AUC scores and sample counts
    auc_scores_per_condition = {condition: [] for condition in y_columns}
    sample_counts_per_condition = {condition: [] for condition in y_columns}

    # Iterate through the splits
    for fold, (train_index, val_index) in enumerate(ml_stratified_kfold.split(X, y_binary)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y_binary.iloc[train_index], y_binary.iloc[val_index]

        # Train the model
        ovr_classifier.fit(X_train, y_train)

        # Make predictions on the training set
        y_train_pred_proba = ovr_classifier.predict_proba(X_train)

        # Make predictions on the validation set
        y_val_pred_proba = ovr_classifier.predict_proba(X_val)


        # Calculate the AUC score for each disease
        auc_scores_train = [roc_auc_score(y_train[column], y_train_pred_proba[:, i]) for i, column in enumerate(y_train.columns)]
        auc_scores_val = [roc_auc_score(y_val[column], y_val_pred_proba[:, i]) for i, column in enumerate(y_val.columns)]

        # Calculate AUC score for each disease
        fold_auc_val = roc_auc_score(y_val.values.ravel(), y_val_pred_proba.ravel())
        auc_scores_val_per_fold.append(fold_auc_val)

        fold_auc_train = roc_auc_score(y_train.values.ravel(), y_train_pred_proba.ravel())
        auc_scores_train_per_fold.append(fold_auc_train)

        # Vega auc calc
        auc_scores_per_fold_train.append(np.mean(auc_scores_train))
        auc_scores_per_fold_val.append(np.mean(auc_scores_val))

        # get sample counts
        for i, condition in enumerate(y_columns):
            sample_count = y_train[condition].sum()  # Count of positive samples for the condition
            sample_counts_train_per_condition[condition].append(sample_count)

        for i, condition in enumerate(y_columns):
            sample_count = y_val[condition].sum()  # Count of positive samples for the condition
            sample_counts_val_per_condition[condition].append(sample_count)

        print("EXPERIMENT: ", exp_name)
        print(f"Fold {fold+1} ========================================")
        #print(f"\nFold {fold+1} AUC Scores for Diseases TRAINING-set:")
        for i, auc_score in enumerate(auc_scores_train, start=1):
            #print(f"{y_train.columns[i-1]}: {auc_score}")
            auc_scores_train_per_condition[y_train.columns[i-1]].append(auc_score)

        #print(f"\nFold {fold+1} AUC Scores for Diseases VALIDATION-set:")
        for i, auc_score in enumerate(auc_scores_val, start=1):
            #print(f"{y_val.columns[i-1]}: {auc_score}")
            auc_scores_val_per_condition[y_val.columns[i-1]].append(auc_score)

        if exp_name == 'exp20':
            exp20_auc_scores_per_condition = auc_scores_per_condition.copy()

    # Vega

    average_auc_score_train = np.mean(auc_scores_per_fold_train)
    average_auc_score_val = np.mean(auc_scores_per_fold_val)
    results_auc_train_vega[exp_name] = average_auc_score_train
    results_auc_val_vega[exp_name] = average_auc_score_val

    # Print the results
    print("Training Set AUC Scores:")
    for exp_name, auc_score in results_auc_train_vega.items():
        print(f"{exp_name}: Average AUC = {auc_score:.4f}")

    print("\nValidation Set AUC Scores:")
    for exp_name, auc_score in results_auc_val_vega.items():
        print(f"{exp_name}: Average AUC = {auc_score:.4f}")


    # Calculate average AUC scores for training and validation sets
    average_auc_scores_per_fold_train = calculate_average_auc_scores(auc_scores_train_per_condition, sample_counts_train_per_condition)
    #print("==========")
    average_auc_scores_per_fold_val = calculate_average_auc_scores(auc_scores_val_per_condition, sample_counts_val_per_condition)

    # Print the results
    #for i, average_auc in enumerate(average_auc_scores_per_fold_train, start=1):
        #print(f'AUC-Score Train Fold-{i}: {average_auc}')

    #for i, average_auc in enumerate(average_auc_scores_per_fold_val, start=1):
        #print(f'AUC-Score Val Fold-{i}: {average_auc}')

    # Calculate and print the total average AUC score and the maximum difference for training set
    total_average_auc_train = np.mean(average_auc_scores_per_fold_train)
    max_difference_train = np.max(np.abs(average_auc_scores_per_fold_train - total_average_auc_train))
    #print(f'Total Average AUC Score train: {total_average_auc_train * 100:.2f} ± {max_difference_train * 100:.2f}%')

    # Calculate and print the total average AUC score and the maximum difference for validation set
    total_average_auc_val = np.mean(average_auc_scores_per_fold_val)
    max_difference_val = np.max(np.abs(average_auc_scores_per_fold_val - total_average_auc_val))
    #print(f'Total Average AUC Score val: {total_average_auc_val * 100:.2f} ± {max_difference_val * 100:.2f}%')


    # Calculate average AUC per condition for VAL
    average_auc_val_per_condition = {
        condition: sum(auc_scores) / len(auc_scores) for condition, auc_scores in auc_scores_val_per_condition.items()
    }
    # Calculate overall average AUC weighted by the number of samples
    overall_average_auc_val_sum = 0
    for auc_scores, sample_counts in zip(auc_scores_val_per_condition.values(), sample_counts_val_per_condition.values()):
        weighted_sum = sum(auc * sample_count for auc, sample_count in zip(auc_scores, sample_counts))
        overall_average_auc_val_sum += weighted_sum / sum(sample_counts)

    overall_average_auc_val = overall_average_auc_val_sum / len(auc_scores_val_per_condition)  # divide by the number of conditions

    # Calculate average AUC per condition for TRAIN
    average_auc_train_per_condition = {
        condition: sum(auc_scores) / len(auc_scores) for condition, auc_scores in auc_scores_train_per_condition.items()
    }
    # Calculate overall average AUC weighted by the number of samples
    overall_average_auc_train_sum = 0
    for auc_scores, sample_counts in zip(auc_scores_train_per_condition.values(), sample_counts_train_per_condition.values()):
        weighted_sum = sum(auc * sample_count for auc, sample_count in zip(auc_scores, sample_counts))
        overall_average_auc_train_sum += weighted_sum / sum(sample_counts)

    overall_average_auc_train = overall_average_auc_train_sum / len(auc_scores_train_per_condition)  # divide by the number of conditions


    # Average AUC-score per condition
    #print("\n Average AUC train-score per condition:")
    #for condition, avg_auc in average_auc_train_per_condition.items():
    #    print(f"{condition}: {avg_auc}")

    print("\n Average AUC val-score per condition:")
    for condition, avg_auc in average_auc_val_per_condition.items():
        print(f"{condition}: {avg_auc}")

    # Get average AUC values as a list
    average_auc_val_list = list(average_auc_val_per_condition.values())
    average_auc_train_list = list(average_auc_train_per_condition.values())

    # Print average AUC scores
    print("\nWeighted Average AUC Score:")
    #print(f"Overall: {overall_average_auc}")
    print(f"Train: {overall_average_auc_train * 100:.2f} ± {max_difference_train * 100:.2f}%")
    print(f"Val: {overall_average_auc_val * 100:.2f} ± {max_difference_val * 100:.2f}%")

    results_auc_train[exp_name] = f"Train: {overall_average_auc_train * 100:.2f} ± {max_difference_train * 100:.2f}%"
    results_auc_val[exp_name] = f"Val: {overall_average_auc_val * 100:.2f} ± {max_difference_val * 100:.2f}%"

# Print the results
print("Training Set AUC Scores:")
for exp_name, auc_score in results_auc_train.items():
    print(f"{exp_name}: Average AUC = {auc_score}")

print("\nValidation Set AUC Scores:")
for exp_name, auc_score in results_auc_val.items():
    print(f"{exp_name}: Average AUC = {auc_score}")


model_filepath = f'../models/{select_model}_v3.joblib'
joblib.dump(ovr_classifier, model_filepath)