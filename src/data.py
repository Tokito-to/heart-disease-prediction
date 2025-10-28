import pandas as pd
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Dataset
def load_dataset(dataset, target_list):
    data = pd.read_csv(dataset)
    feature_names = data.drop(columns=target_list).columns.tolist()
    target_names = target_list

    return {
        "data":  data,
        "features":  feature_names,
        "target":  target_names
    }


def load_Xy(data, feature, target):
    X = data[feature].values
    y = data[target].values

    return {"X": X, "y": y}


def test_train_split(X, y):
    # Oversample using SMOTE
    X, y = SMOTE(random_state=42).fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    # Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test
    }


def data_dictionary(dataset_path, target_list):
    data = load_dataset(dataset_path, target_list)
    data.update(load_Xy(data["data"], data["feature"], data["target"]))
    data.update(test_train_split(data["X"], data["y"]))

    return data
