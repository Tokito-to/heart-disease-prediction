import random
import numpy as np

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import minmax_scale

from deap import creator

# Dataset Parameters
parameters = {
    "dataset": 'heart_dataset.csv',
    "target": 'target'
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

ga_parameters = {
    "generation": 2,
    "population": 10
}


def hyprparameters(hyprparameter_ranges):

    hyperparameter = []

    for key, (low, high) in hyprparameter_ranges.items():

        if isinstance(low, int) and isinstance(high, int):
            random_value = random.randint(low, high)
        elif isinstance(low, float) and isinstance(high, float):
            random_value = random.uniform(low, high)

        hyperparameter.append(random_value)

    return hyperparameter


# Feature Importance
def feature_importance(X_train, y_train):
    # Compute feature importance (RF + MI + SHAP)
    rf = RandomForestClassifier(random_state=42).fit(X_train, y_train)

    explainer = shap.Explainer(rf)
    shap_values = explainer(X_train, check_additivity=False)
    shap_values_class1 = shap_values.values[:, :, 1]

    rf_importance = rf.feature_importances_
    mi_importance = mutual_info_classif(X_train, y_train)
    shap_importance = np.abs(shap_values_class1).mean(axis=0)

    combined_score = (
        minmax_scale(rf_importance) +
        minmax_scale(mi_importance) +
        minmax_scale(shap_importance)
    ) / 3
    ranked_features = np.argsort(combined_score)[::-1]

    return ranked_features


# Smart Individual Initialization
def smart_individual():

    k = random.randint(6, 10)
    feature_mask = [0] * 12

    ranked_features = [10,  3,  4, 11,  2,  9,  7,  6,  0,  1,  5,  8]
    for idx in ranked_features[:k]:
        feature_mask[idx] = 1

    hyprparameter = hyprparameters(hyprparameter_ranges)

    return creator.Individual(feature_mask + hyprparameter)


# Callback Functions
def callbacks():
    # Early stopping
    early_stopping = EarlyStopping(
        start_from_epoch=25,
        monitor='val_loss',
        min_delta=0.001,
        patience=5
    )

    # Reduce Learning
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        min_lr=0.0001,
        factor=0.8,
        patience=2
    )

    return [early_stopping, reduce_lr]
