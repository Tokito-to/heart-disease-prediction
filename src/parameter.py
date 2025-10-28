from src.data import data_dictionary

# Data Parameters
# Populates [data, feature, target, X, y, X_train, y_train, X_test, y_test]
dataset_path = '../heart_dataset.csv'
target_list = ['target']

dataset = data_dictionary(dataset_path, target_list)

# GA Parameters
ga_parameters = {
    "generation": 2,
    "population": 5
}

# Hyperparameter Ranges
hyprparameter_ranges = {
    "l1": (128, 192),
    "l2": (64, 128),
    "l3": (32, 64),
    "learning_rate": (0.0001, 0.01),
    "dropout_rate": (0.0, 0.4),
    "l2_regularization": (0.000001, 0.01),
    "alpha": (0.01, 0.3)
}
