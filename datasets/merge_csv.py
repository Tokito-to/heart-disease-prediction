import pandas as pd

FILE_LIST = [ './heart_disease_cleaned.csv', './cleaned_merged_heart_dataset.csv', './Cardiovascular_Disease_Dataset.csv' ]

# Mapping
SEX_MAPPING = {'Male': 1, 'Female': 0}
CP_MAPPING = {
    'typical angina': 0,
    'atypical angina': 1,
    'non-anginal': 2,
    'asymptomatic': 3
}
RESTECG_MAPPING = {
    'normal': 0,
    'st-t abnormality': 1,
    'lv hypertrophy': 2
}
SLOPE_MAPPING = {
    'upsloping': 0,
    'flat': 1,
    'downsloping': 2
}
THAL_MAPPING = {
    'normal': 1,
    'fixed defect': 2,
    'reversable defect': 3
}
THAL_INT_MAPPING = { 0:0, 1:1, 2:2, 3:3, 3: 1, 6: 2, 7: 3 }
CP_INT_MAPPING = { 0: 0, 1: 0, 2: 1, 3: 2, 4: 3 }
SLOPE_INT_MAPPING = { 0: 0, 1: 0, 2: 1, 3: 2 }
TARGET_INT_MAPPING = { 0: 0, 1: 1, 2: 1, 3: 1, 4: 1 }

# Filters
FILTERS = {
    'sex': [0, 1],
    'cp': [0, 1, 2, 3],
    'fbs': [0, 1],
    'restecg': [0, 1, 2],
    'exang': [0, 1],
    'slope': [0, 1, 2],
    'ca': [0, 1, 2, 3],
    'target': [0, 1]
}

dataframes = []

# Preprocess Data
for file in FILE_LIST:
    data = pd.read_csv(file, sep=',', na_values=['?'])
    data.dropna(inplace=True)
    if 'id' in data.columns:
        data.drop('id', axis=1, inplace=True)
    if 'dataset' in data.columns:
        data.drop('dataset', axis=1, inplace=True)
    if 'patientid' in data.columns:
        data.rename(columns={'gender': 'sex'}, inplace=True)
        data.rename(columns={'chestpain': 'cp'}, inplace=True)
        data.rename(columns={'restingBP': 'trestbps'}, inplace=True)
        data.rename(columns={'serumcholestrol': 'chol'}, inplace=True)
        data.rename(columns={'fastingbloodsugar': 'fbs'}, inplace=True)
        data.rename(columns={'restingrelectro': 'restecg'}, inplace=True)
        data.rename(columns={'maxheartrate': 'thalach'}, inplace=True)
        data.rename(columns={'exerciseangia': 'exang'}, inplace=True)
        data.rename(columns={'noofmajorvessels': 'ca'}, inplace=True)
        data.drop('patientid', axis=1, inplace=True)
    if 'thal' in data.columns:
        data.drop('thal', axis=1, inplace=True)
    data.rename(columns={'num': 'target'}, inplace=True)
    data.rename(columns={'thalch': 'thalach'}, inplace=True)
    data.rename(columns={'thalachh': 'thalach'}, inplace=True)
    if data['sex'].dtype == 'object':
        data['sex'] = data['sex'].apply(lambda x: SEX_MAPPING[x])
    if data['cp'].dtype == 'int64':
        data['cp'] = data['cp'].apply(lambda x: CP_INT_MAPPING[x])
    if data['cp'].dtype == 'object':
        data['cp'] = data['cp'].apply(lambda x: CP_MAPPING[x])
    if data['trestbps'].dtype == 'float64':
        data['trestbps'] = data['trestbps'].astype(int)
    if data['chol'].dtype == 'float64':
        data['chol'] = data['chol'].astype(int)
    if data['restecg'].dtype == 'object':
        data['restecg'] = data['restecg'].apply(lambda x: RESTECG_MAPPING[x])
    data['thalach'] = data['thalach'].astype(int)
    if data['slope'].dtype == 'int64':
        data['slope'] = data['slope'].apply(lambda x: SLOPE_INT_MAPPING[x])
    if data['slope'].dtype == 'object':
        data['slope'] = data['slope'].apply(lambda x: SLOPE_MAPPING[x])
    if data['ca'].dtype == 'float64':
        data['ca'] = data['ca'].astype(int)
    #if data['thal'].dtype == 'float64':
    #    data['thal'] = data['thal'].apply(lambda x: THAL_INT_MAPPING[x])
    #if data['thal'].dtype == 'int64':
    #    data['thal'] = data['thal'].apply(lambda x: THAL_INT_MAPPING[x])
    #if data['thal'].dtype == 'object':
    #    data['thal'] = data['thal'].apply(lambda x: THAL_MAPPING[x])

    data['target'] = data['target'].apply(lambda x: TARGET_INT_MAPPING[x])

    dataframes.append(data)

data = pd.concat(dataframes, ignore_index=True)

precleanup = len(data)
print(f'Merged Dataset Size: {precleanup}')

# Cleanup
#   age; sex (1,0); cp (1-4); trestbps; chol; fbs (1,0); restecg (0,1,2);
#   thalach; exang (1,0); oldpeak; slope (1,2,3); ca; thal (3,6,7);
#  class att: 0 is healthy, 1,2,3,4 is sick.

initial_rows = data.shape[0]

# Apply Filters
for column, values in FILTERS.items():
    prev_rows = data.shape[0]
    data = data[data[column].isin(values)]
    rows_dropped = prev_rows - data.shape[0]
    print(f"Rows dropped by filter '{column}': {rows_dropped}")

chol_zero = (data['chol'] == 0).sum()
print(f'Rows dropped by filter chol = 0: {chol_zero}')
data = data[data['chol'] != 0]

print(f"Total rows dropped: {initial_rows - data.shape[0]}")

postcleanup = len(data)

# Drop Duplicates
duplicates = data.duplicated().sum()
print(f"Number of duplicates: {duplicates}")

data.drop_duplicates(subset=None, inplace=True)
print(f'Final Dataset Size: {len(data)}')

# Write cleaned dataset
data.to_csv('../heart_dataset.csv', index=False)
