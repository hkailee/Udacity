#!/usr/bin/env python3
__author__ = 'mdc_hk'
version = '1.0'

#================================= Import library =================================================
import argparse, datetime, os, sys, time, logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# user-defined functions
from utils import scale

#=================================== Input Argument check ======================================
# Checks if in proper number of arguments are passed gives instructions on proper use.
def argsCheck(numArgs):
	if len(sys.argv) < numArgs or len(sys.argv) > numArgs:
		print('Algorithm to predict for LOS based on vitals')        
		print('Usage: $ python', sys.argv[0], '-e <emergency csv input dir>', 
             '-hz <hospitalization csv input dir>', 
             '-o <output dir>')
		print('Example: $ python', sys.argv[0], '-e test/emergency.csv', 
             '-hz test/hospitalization.csv', 
             '-o test/predict.csv')
		exit(1) # Aborts program. (exit(1) indicates that an error occurred)

argsCheck(7) # Checks if the number of arguments are correct.

# Arguments
parser = argparse.ArgumentParser(description='Predicting Length of Stay')
parser.add_argument("-e", required=False, type=str, help="Emergency filename")
parser.add_argument("-hz", required=False, type=str, help="hospitalization filename")
parser.add_argument("-o", required=True, type=str, help="Output filename")
args = parser.parse_args()

#=============================== Reading Datasets =============================================
# Stores file one for input checking.
test_emergency_file  = args.e
test_hospitalization_file = args.hz
test_output_file  = args.o

# Setting up working and fluSeq directories...
workingFolder = os.getcwd()

# Logging events...
logging.basicConfig(filename=workingFolder + '/Log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
startTime = time.time()
logging.info('Runfolder path: ' + workingFolder)

# Read training files
df_discharge = pd.read_csv('train/discharge.csv', sep=',', index_col=0)
df_emergency = pd.read_csv('train/emergency.csv', sep=',', index_col=0)
df_hospitalization = pd.read_csv('train/hospitalization.csv', sep=',', index_col=0)
df_merged = pd.merge(df_hospitalization, df_emergency, how='outer', left_index=True, right_index=True)

# Read test files
df_emergency_test = pd.read_csv(test_emergency_file, sep=',', index_col=0)
df_hospitalization_test = pd.read_csv(test_hospitalization_file, sep=',', index_col=0)
df_merged_test = pd.merge(df_hospitalization_test, df_emergency_test, how='outer', left_index=True, right_index=True)

# Merge training files
for pid in df_merged.index:
    df_merged.at[pid, 'rLOS'] = df_discharge.at[pid, 'LOS_D'] - df_merged.at[pid, 'Hospital_day'] + 1

#========================================= Features Engineering =========================================
for param in ['SBP_daily', 'DBP_daily', 'HR_daily', 'RR_daily', 'SBP_ED', 'DBP_ED', 
              'Pulse_ED', 'Resp_Rate']:

    # training dataset
    df_merged['max_{}'.format(param)] = df_merged.filter(regex=("^{}.*").format(param)).max(axis=1)
    df_merged['min_{}'.format(param)] = df_merged.filter(regex=("^{}.*").format(param)).min(axis=1)
    df_merged['count_{}'.format(param)] = df_merged.filter(regex=("^{}.*").format(param)).count(axis=1)
    df_merged = df_merged.drop(columns=list(df_merged.filter(regex=("^{}.*".format(param))).columns))

    # testing dataset
    df_merged_test['max_{}'.format(param)] = df_merged_test.filter(regex=("^{}.*").format(param)).max(axis=1)
    df_merged_test['min_{}'.format(param)] = df_merged_test.filter(regex=("^{}.*").format(param)).min(axis=1)
    df_merged_test['count_{}'.format(param)] = df_merged_test.filter(regex=("^{}.*").format(param)).count(axis=1)
    df_merged_test = df_merged_test.drop(columns=list(df_merged_test.filter(regex=("^{}.*".format(param))).columns))


for param in ['Temp_daily', 'Temperature_ED', 'RA_or_Sup_O2_ED']:

    # training dataset
    df_merged['max_{}'.format(param)] = df_merged.filter(regex=("^{}.*").format(param)).max(axis=1)
    df_merged['count_{}'.format(param)] = df_merged.filter(regex=("^{}.*").format(param)).count(axis=1)
    df_merged = df_merged.drop(columns=list(df_merged.filter(regex=("^{}.*".format(param))).columns))

    # testing dataset
    df_merged_test['max_{}'.format(param)] = df_merged_test.filter(regex=("^{}.*").format(param)).max(axis=1)
    df_merged_test['count_{}'.format(param)] = df_merged_test.filter(regex=("^{}.*").format(param)).count(axis=1)
    df_merged_test = df_merged_test.drop(columns=list(df_merged_test.filter(regex=("^{}.*".format(param))).columns))


for param in ['SpO2_daily', 'SpO2_ED']:

    # training dataset
    df_merged['min_{}'.format(param)] = df_merged.filter(regex=("^{}.*").format(param)).min(axis=1)
    df_merged['count_{}'.format(param)] = df_merged.filter(regex=("^{}.*").format(param)).count(axis=1)
    df_merged = df_merged.drop(columns=list(df_merged.filter(regex=("^{}.*".format(param))).columns))

    # testing dataset
    df_merged_test['min_{}'.format(param)] = df_merged_test.filter(regex=("^{}.*").format(param)).min(axis=1)
    df_merged_test['count_{}'.format(param)] = df_merged_test.filter(regex=("^{}.*").format(param)).count(axis=1)
    df_merged_test = df_merged_test.drop(columns=list(df_merged_test.filter(regex=("^{}.*".format(param))).columns))


#========================= Preprocessing for train, validation, test datasets ===================
# features and labels, rLOS & Split them into training and tesitng sets
df_features = df_merged.drop(columns=['rLOS'])
df_labels = df_merged['rLOS']

X_train, X_valid, y_train, y_valid = train_test_split(df_features, 
                                                            df_labels, 
                                                            test_size = 0.3, 
                                                            random_state = 42)

logging.info("Training set has {} samples.".format(X_train.shape[0]))
logging.info("Validation set has {} samples.".format(X_valid.shape[0]))
print("[INFO] - Training set has {} samples.".format(X_train.shape[0]))
print("[INFO] - Validation set has {} samples.".format(X_valid.shape[0]))


# Preprocessing pipeline
# Inspired by https://medium.com/vickdata/a-simple-guide-to-scikit-learn-pipelines-4ac0d974bdcf
categorical_features = ['New_Medication_ED', 'max_RA_or_Sup_O2_ED', 'New_med_yn']
categorical_dummy_features = ['New_med_dosechange']
numeric_features = [x for x in df_features.columns if x not in categorical_features + categorical_dummy_features]

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(missing_values=np.NaN, strategy='most_frequent'))])

onehot_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(missing_values=np.NaN, strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features), 
        ('cat_onehot', onehot_transformer, categorical_dummy_features)])

pipe = Pipeline(steps=[('preprocessor', preprocessor),])
X_train_transformed = pipe.fit_transform(X_train)
X_valid_transformed = pipe.transform(X_valid)
X_test_transformed = pipe.transform(df_merged_test)

onehot_features = list(pipe['preprocessor'].transformers_[2][1]['onehot']\
                          .get_feature_names(categorical_dummy_features))

clean_categorical_features = list([feature for feature in categorical_features + onehot_features\
                              if feature not in ['New_med_dosechange', 'New_med_dosechange_1']])

X_train_transformed = pd.DataFrame(X_train_transformed, 
                                   columns=numeric_features + categorical_features + onehot_features, 
                                     index=X_train.index)[numeric_features + clean_categorical_features].astype(float)
X_valid_transformed = pd.DataFrame(X_valid_transformed, 
                                  columns=numeric_features + categorical_features + onehot_features, 
                                 index=X_valid.index)[numeric_features + clean_categorical_features].astype(float)
X_test_transformed = pd.DataFrame(X_test_transformed, 
                                  columns=numeric_features + categorical_features + onehot_features, 
                                 index=df_merged_test.index)[numeric_features + clean_categorical_features].astype(float)

# Min-max scaling for all features (0, 1)
X_train_transformed[clean_categorical_features] = X_train_transformed[clean_categorical_features].apply(lambda x: scale(x))
X_valid_transformed[clean_categorical_features] = X_valid_transformed[clean_categorical_features].apply(lambda x: scale(x))
X_test_transformed[clean_categorical_features] = X_test_transformed[clean_categorical_features].apply(lambda x: scale(x))

#========================= XGBoost training with selected features ============================

xgb_reg = xgboost.XGBRegressor(random_state=42, nthread=4)
xgb_reg.fit(X_train_transformed[['Hospital_day', 'min_SBP_ED', 'min_SpO2_ED', 'Weight_ED']], y_train)
y_predicted = xgb_reg.predict(X_test_transformed[['Hospital_day', 'min_SBP_ED', 'min_SpO2_ED', 'Weight_ED']])
X_test_transformed['y_predicted'] = y_predicted

# preparing output df
df_merged_test_dummy = df_merged_test.copy()
df_merged_test_dummy['y_predicted'] = X_test_transformed['y_predicted']
df_merged_test_dummy.reset_index(drop=False, inplace=True)
df_output = pd.DataFrame(index=df_merged_test_dummy['Patient'].unique())

for pid in df_output.index:
    df_tmp = df_merged_test_dummy[df_merged_test_dummy.Patient==pid].set_index('Hospital_day', drop=False)
    for day in df_tmp['Hospital_day']:
        df_output.at[pid, 'predicted_rLOS_at_day_{}'.format(day)] = df_tmp.at[day, 'y_predicted'].round().astype(int) - 1

# export the output
df_output.to_csv(test_output_file, sep=',')
logging.info("Predicted results saved to {}.".format(workingFolder + '/' + test_output_file))
print("[INFO] - Predicted results saved to {}.".format(workingFolder + '/' + test_output_file))