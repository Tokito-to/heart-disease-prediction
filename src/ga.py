import random
import numpy as np

import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import minmax_scale

from src.parameter import dataset, hyprparameter_ranges
from src.model import create_ann_model

from sklearn.metrics import accuracy_score

from deap import base, creator, tools, algorithms

feature_names = dataset["feature"]
X_train = dataset["X_train"]
X_test = dataset["X_test"]
y_test = dataset["y_test"]


# Hyprparameter Generator
def random_hyprparameters(hyprparameter_ranges):

    hyprparameter = []

    for key, (low, high) in hyprparameter_ranges.items():

        if isinstance(low, int) and isinstance(high, int):
            random_value = random.randint(low, high)
        elif isinstance(low, float) and isinstance(high, float):
            random_value = random.uniform(low, high)

        hyprparameter.append(random_value)

    return hyprparameter


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


# Feature Selection using top-K ranked features + Random Hyprparamaters
def smart_individual(ranked_features, hyprparameter_ranges, input_shape):
    k = random.randint(6, 10)
    feature_mask = [0] * 12

    for idx in ranked_features[:k]:
        feature_mask[idx] = 1

    hyprparameter = random_hyprparameters(hyprparameter_ranges)

    return feature_mask + hyprparameter


# Repair function
def repair(individual, ranked_features, input_shape):
    hypr_shape = input_shape + len(hyprparameter_ranges)

    feature_mask = individual[:input_shape]
    hyprparameter = individual[input_shape:hypr_shape]

    # Feature Selection
    for i in range(input_shape):
        if feature_mask[i] in (0, 1):
            feature_mask[i] = int(individual[i])
        else:
            feature_mask[i] = 1

    if sum(feature_mask[:input_shape]) == 0:
        k = random.randint(6, 10)
        for idx in ranked_features[:k]:
            feature_mask[idx] = 1

    # Hyprparameters
    for i, (key, (low, high)) in enumerate(hyprparameter_ranges.items()):
        clipped = np.clip(hyprparameter[i], low, high)

        if isinstance(low, int) and isinstance(high, int):
            hyprparameter[i] = int(clipped)
        elif isinstance(low, float) and isinstance(high, float):
            hyprparameter[i] = float(clipped)

    individual[:input_shape] = feature_mask
    individual[input_shape:hypr_shape] = hyprparameter

    return individual


# Evaluation
def evaluate(individual, input_shape, generation):
    feature_mask = individual[:input_shape]
    hyprparameters = individual[input_shape:]
    selected_indices = []
    selected_features = []

    for i, bit in enumerate(feature_mask):
        if bit == 1:
            selected_indices.append(i)

    for i, bit in enumerate(feature_mask):
        if bit == 1:
            selected_features.append(feature_names[i])

    selected_input_shape = len(selected_indices)

    global X_train_selected, X_test_selected
    X_train_selected = X_train[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]

    model, _ = create_ann_model(hyprparameters, selected_input_shape)
    y_probability = model.predict(X_test_selected)
    y_prediction = (y_probability > 0.5)

    accuracy = accuracy_score(y_test, y_prediction)

    return (accuracy,)


def ga_setup():
    toolbox = base.Toolbox()

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox.register(
        "individual", tools.initIterate, creator.Individual,
        lambda: smart_individual(
            ranked_features, hyprparameter_ranges, input_shape
        ))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)

    return toolbox
