import pandas as pd
import os
import numpy as np
import pandas as pd
import joblib
from functools import reduce
import xgboost as xgb
from xgboost import XGBClassifier
#from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.multiclass import OneVsRestClassifier
# from skmultilearn.model_selection import MultiLabelStratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

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
# feat_HotWheater = pd.read_csv('../features/one_hot_encoded_HotWheater_with_id.csv')
# feat_ModerateWheather = pd.read_csv('../features/one_hot_encoded_ModerateWheather_with_id.csv')
# feat_ColdWheater_with_id = pd.read_csv('../features/one_hot_encoded_ColdWheater_with_id.csv')
feat_cold_month = pd.read_pickle('../features/ColdMonths.pkl')
feat_hot_month = pd.read_pickle('../features/HotMonths.pkl')

# Diet
feat_diet = pd.read_pickle('../features/diet.pkl') # 5 columns

# Physical Activity
feat_pa_total_hours = pd.read_csv('../features/PhysicalActivity_total_hours.csv')
feat_pa_surface = pd.read_csv('../features/PhysicalActivity_surface.csv')
feat_pa_wheather = pd.read_csv('../features/PhysicalActivity_wheather.csv')

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
feat_age_condition = pd.read_pickle('../features/age_condition.pkl')
feat_age_condition_type = pd.read_pickle('../features/age_condition_type.pkl')


print("loaded features")


features_list = [

    # breed
    #feat_breed,
    #feat_breed_v2,
    #feat_breed_top50,
    feat_breed_top50_v2,
    #feat_breed_group,
    #feat_sub_breed,
    #feat_breed_type,
    #feat_breed_pure_or_mix,

    # age
    #feat_age,   # 24881 dog_ids

    # sex
    feat_sex,

    # weight
    #feat_weight

    # diet
    #feat_diet,  # 33141 dog_ids for df_diet_consistency

    #feat_HotWheater,
    #feat_ModerateWheather,
    #feat_ColdWheater_with_id,

    # Climate
    #feat_pa_surface,
    #feat_pa_total_hours,
    #feat_pa_wheather,   # 26406 dog_ids
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

feat_age_condition_type

#exp1 = [feat_breed_group]
#exp2 = [feat_breed_top50_v2]
#exp3 = [feat_age]
#exp4 = [feat_sex,feat_breed_type]
#exp5 = [feat_sub_breed]
#exp6 = [feat_breed_pure_or_mix]
#exp7 = [feat_breed_top50_v2,feat_breed_group,feat_weight]
#exp8 = [feat_breed_top50_v2,feat_breed_group,feat_age,feat_sex,feat_age_condition,feat_age_condition_type]
#exp9 = [feat_breed_top50_v2,feat_breed_group,feat_age,feat_sex,feat_pa_surface,feat_pa_total_hours,feat_pa_wheather]
#exp10 = [feat_breed_top50_v2,feat_breed_group,feat_age,feat_sex,feat_od_income,feat_od_education,feat_cold_month,feat_hot_month,feat_prim_census_division]
#exp11 = [feat_breed_top50_v2,feat_breed_group,feat_age,feat_sex,feat_od_income,feat_od_education]
#exp12 = [feat_breed_top50_v2,feat_breed_group,feat_age,feat_sex,feat_od_income,feat_od_education,feat_cold_month,feat_hot_month,feat_prim_census_division,feat_pa_surface,feat_pa_total_hours,feat_pa_wheather]
#exp13 = [feat_breed_top50_v2,feat_breed_group,feat_cold_month,feat_hot_month,feat_prim_census_division,feat_pa_surface,feat_pa_total_hours,feat_pa_wheather]
#exp14 = [feat_breed_top50_v2,feat_breed_group,feat_cold_month,feat_hot_month,feat_prim_census_division,feat_pa_surface,feat_pa_total_hours,feat_pa_wheather,feat_age,feat_sex,feat_diet]
#exp15 = [feat_breed_top50_v2,feat_breed_group,feat_od_income,feat_od_education,feat_cold_month,feat_hot_month,feat_prim_census_division,feat_pa_surface,feat_pa_total_hours,feat_pa_wheather,feat_age,feat_sex,feat_diet]
#exp16 = [feat_breed_top50_v2,feat_breed_group,feat_od_income,feat_od_education,feat_cold_month,feat_hot_month,feat_prim_census_division,feat_pa_surface,feat_pa_total_hours,feat_pa_wheather,feat_age,feat_sex,feat_breed_pure_or_mix,feat_breed_type,feat_sub_breed,feat_diet]
#exp17 = [feat_breed_top50_v2,feat_breed_group,feat_pa_surface,feat_pa_total_hours,feat_pa_wheather,feat_age,feat_sex,feat_breed_pure_or_mix,feat_breed_type,feat_sub_breed,feat_diet,feat_weight]
#exp18 = [feat_breed_top50_v2,feat_breed_group,feat_od_income,feat_od_education,feat_breed_pure_or_mix,feat_breed_type,feat_sub_breed,feat_diet,feat_age_condition,feat_age_condition_type]
#exp19 = [feat_breed_top50_v2,feat_breed_group,feat_pa_surface,feat_pa_total_hours,feat_pa_wheather,feat_sub_breed,feat_diet,feat_weight]
exp20 = [# breed,
    #feat_breed,
    #feat_breed_v2,
    feat_breed_top50,
    feat_breed_top50_v2,
    feat_breed_group,
    feat_sub_breed,
    feat_breed_type,
    feat_breed_pure_or_mix,

    # age
    #feat_age,   # 24881 dog_ids

    # sex
    feat_sex,

    # weight
    feat_weight,

    # diet
    feat_diet,  # 33141 dog_ids for df_diet_consistency

    #feat_HotWheater,
    #feat_ModerateWheather,
    #feat_ColdWheater_with_id,

    # Climate
    feat_pa_surface,
    feat_pa_total_hours,
    #feat_pa_wheather,   # 26406 dog_ids
    feat_hot_month,
    feat_cold_month,

    # Physical Activity
    feat_pa_total_hours,
    feat_pa_surface,

    # Owner demographics
    #feat_od_income,    # 29096 dog_ids
    feat_od_education,

    # Residentual
    feat_prim_census_division,

    # disease
    #feat_disease_input,
    #feat_disease_output_binary,
    #feat_disease_output
]

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
    #'exp11': exp11,
    #'exp12': exp12,
    #'exp13': exp13,
    #'exp15': exp15,
    #'exp16': exp16,
    #'exp17': exp17,
    #'exp18': exp18,
    #'exp19': exp19,
    'exp20': exp20,
}
results_auc_train = {}
results_auc_val = {}
results_accuracy = {}

for exp_name, exp_features in experiments.items():
    # List of DataFrames to be merged
    list_input_features = features_list  # Add more DataFrames as needed

    # Merge DataFrames iteratively using reduce
    input_features = reduce(lambda left, right: pd.merge(left, right, on='dog_id'), exp_features)

    # Merge with disease output feature
    data = pd.merge(input_features, feat_age_condition_type, on='dog_id')
    print(data.shape)

    # Assuming 'data' is your DataFrame
    rows_with_nan = data[data.isna().any(axis=1)]
    columns_with_nan = data.columns[data.isna().any()].tolist()
    # Display the rows with NaN values
    rows_with_nan[columns_with_nan]

    # Normalize the specified features in one line and save in the same columns
    features_to_normalize = ['age_diagnosis_years', 'df_diet_consistency', 'df_feedings_per_day', 'total_active_hours_y', 'pa_hot_weather_months_per_year', 'pa_cold_weather_months_per_year', 'total_active_hours_y', 'od_annual_income_range_usd']

    # select features that are in data
    features_to_normalize_in_data = [feature for feature in features_to_normalize if feature in data.columns]

    # Initialize MinMaxScaler
    scaler = MinMaxScaler()

    # Normalize
    if features_to_normalize_in_data:
        data[features_to_normalize_in_data] = scaler.fit_transform(data[features_to_normalize_in_data])
        # print(data[features_to_normalize_in_data])

    # Separate features and labels
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

    # Initialize the Naive Bayes model
    model = xgb.XGBClassifier()

    # Wrap the model with OneVsRestClassifier
    ovr_classifier = OneVsRestClassifier(model)

    # Initialize MultilabelStratifiedKFold
    n_splits = 5  # You can adjust the number of splits as needed
    ml_stratified_kfold = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    accuracy_scores_per_fold = []
    auc_scores_per_fold_train = []
    auc_scores_per_fold_val = []
    # Lists to store AUC scores and sample counts
    auc_scores_per_condition = {condition: [] for condition in y_columns}
    sample_counts_per_condition = {condition: [] for condition in y_columns}

    # Iterate through the splits
    for fold, (train_index, val_index) in enumerate(tqdm(ml_stratified_kfold.split(X, y_binary), desc="Cross-validation")):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y_binary.iloc[train_index], y_binary.iloc[val_index]

        # Train the model
        ovr_classifier.fit(X_train, y_train)

        print("Make Prediction Val:")

        # Make predictions on the validation set
        #y_pred_proba = ovr_classifier.predict_proba(X_val)
        #y_pred_binary = (y_pred_proba >= 0.5).astype(int)

        print("Make Prediction Train:")

        # Make predictions on the training and validation set
        y_pred_proba_train = ovr_classifier.predict_proba(X_train)
        y_pred_proba_val = ovr_classifier.predict_proba(X_val)

        # Calculate the AUC score for each disease on the training set
        auc_scores_train = [roc_auc_score(y_train[column], y_pred_proba_train[:, i]) for i, column in enumerate(y_train.columns)]
        auc_scores_per_fold_train.append(np.mean(auc_scores_train))
        print("auc_scores_per_fold_train: ", auc_scores_per_fold_train)

        # Calculate the AUC score for each disease on the validation set
        auc_scores_val = [roc_auc_score(y_val[column], y_pred_proba_val[:, i]) for i, column in enumerate(y_val.columns)]
        auc_scores_per_fold_val.append(np.mean(auc_scores_val))
        print("auc_scores_per_fold_val: ", auc_scores_per_fold_val)

    average_auc_score_train = np.mean(auc_scores_per_fold_train)
    average_auc_score_val = np.mean(auc_scores_per_fold_val)
    results_auc_train[exp_name] = average_auc_score_train
    results_auc_val[exp_name] = average_auc_score_val


# Print the results
print("Training Set AUC Scores:")
for exp_name, auc_score in results_auc_train.items():
    print(f"{exp_name}: Average AUC = {auc_score:.4f}")

print("\nValidation Set AUC Scores:")
for exp_name, auc_score in results_auc_val.items():
    print(f"{exp_name}: Average AUC = {auc_score:.4f}")

# for exp_name, accuracy_score in results_accuracy.items():
#     print(f"{exp_name}: Average Accuracy = {accuracy_score:.4f}")


# Calculate overall average AUC weighted by the number of samples
#overall_average_auc_sum = 0
#for auc_scores, sample_counts in zip(auc_scores_per_condition.values(), sample_counts_per_condition.values()):
#    weighted_sum = sum(auc * sample_count for auc, sample_count in zip(auc_scores, sample_counts))
#    overall_average_auc_sum += weighted_sum / sum(sample_counts)

#overall_average_auc = overall_average_auc_sum / len(auc_scores_per_condition)  # divide by the number of conditions
#print('overall_average_auc',overall_average_auc)

#model_filepath = '../models/ex_grad_boost.joblib'
#joblib.dump(ovr_classifier, model_filepath)