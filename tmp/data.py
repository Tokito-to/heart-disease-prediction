import joblib
import pandas as pd

from src.config import parameters
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load dataset
def dataset_info(dataset):
    data = pd.read_csv(dataset)

    feature_names = data.columns[data.columns != parameters["target"]].tolist()
    target_names = parameters["target"]

    return feature_names, target_names


def load_dataset(dataset, feature_names, target_names):
    data = pd.read_csv(dataset)
    X = data[feature_names].values
    y = data[target_names].values

    # Oversample using SMOTE
    X, y = SMOTE(random_state=42).fit_resample(X, y)

    return data, X, y


def test_train_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    # Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Dump scaler # We need this to use pre-trained model
    # pre-trained model require same scaler as training scaler
    joblib.dump(sc, 'models/scaler.pkl')

    return X_train, y_train, X_test, y_test


def data_dictionary(dataset):
    data = load_dataset('../heart_dataset.csv')
    data.update(load_Xy(data["data"], data["feature"], data["target"]))
    data.update(test_train_split(data["X"], data["y"]))

    return data
