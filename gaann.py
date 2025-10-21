import csv
import random
import joblib

import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import minmax_scale

from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import tensorflow as tf  # noqa: F401,E402

from keras.backend import clear_session  # noqa: E402
from keras.models import Sequential  # noqa: E402
from keras.layers import ReLU, LeakyReLU, PReLU, ELU, Activation  # noqa: F401,E402,E501
from keras.layers import Dense, InputLayer  # noqa: E402
from keras.layers import Dropout, BatchNormalization  # noqa: E402
from keras.callbacks import EarlyStopping, ReduceLROnPlateau  # noqa: E402
from keras.optimizers import Adam  # noqa: E402
from keras.regularizers import l2  # noqa: E402
from keras.metrics import AUC  # noqa: E402

from deap import base, creator, tools, algorithms  # noqa: E402

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

# GA Parameters
ga_parameters = {
    "generation": 2,
    "population": 5
}

# Load dataset
data = pd.read_csv('heart_dataset.csv')
feature_names = data.columns[data.columns != "target"].tolist()
input_shape = len(feature_names)

X = data.iloc[:, :input_shape].values
y = data["target"].values

# Oversample using SMOTE
X, y = SMOTE(random_state=42).fit_resample(X, y)

# Train-test split
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

# Compute feature importance (RF + MI + Shap)
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

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

# Logging
log_file = 'models/logs/ReLU_model.csv'
with open(log_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    header = ['Generation', 'SelectedFeatures']
    for key in hyprparameter_ranges:
        header.append(key)
    header.append('Accuracy')
    writer.writerow(header)

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

# ANN Model
def create_ann_model(hyprparameters, input_shape):
    layer_keys = []

    for key in hyprparameter_ranges:
        if key.startswith('l') and key[1:].isdigit():
            layer_keys.append(key)

    hidden_layers = len(layer_keys)

    clear_session()
    layer_units = hyprparameters[:hidden_layers]
    lr, dr, l2_reg, alpha = hyprparameters[hidden_layers:]

    model = Sequential()
    model.add(InputLayer(shape=(input_shape,)))

    # Hidden Layers
    for neurons in layer_units:
        model.add(Dense(int(neurons), kernel_regularizer=l2(l2_reg)))
        model.add(ReLU())
        model.add(BatchNormalization())
        model.add(Dropout(dr))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy', AUC(name='auc')]
    )

    model_history = model.fit(
        X_train_selected, y_train,
        validation_split=0.30,
        epochs=145, batch_size=35,
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )

    return model, model_history


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


# Feature Selection using top-K ranked features + Random Hyprparamaters
def smart_individual(ranked_features, hyprparameter_ranges, input_shape):
    k = random.randint(6, 10)
    feature_mask = [0] * 12

    for idx in ranked_features[:k]:
        feature_mask[idx] = 1

    hyprparameter = random_hyprparameters(hyprparameter_ranges)

    return feature_mask + hyprparameter


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

    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        log = [generation, ','.join(selected_features)]
        for parameter in hyprparameters:
            log.append(parameter)
        log.append(f"{accuracy * 100:.4f} %")
        writer.writerow(log)

    return (accuracy,)


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


# GA setup
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

# Run GA
generations = ga_parameters["generation"]
population_size = ga_parameters["population"]
best_accuracies = []

population = toolbox.population(n=population_size)

print("\n...Starting Genetic Algorithm Optimization...\n")

for generation in range(1, generations + 1):
    print(f"=== Generation {generation} ===")

    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
    for i, individual in enumerate(offspring):
        offspring[i] = repair(individual, ranked_features, input_shape)

    for individuals in offspring:
        individuals.fitness.values = evaluate(
            individuals, input_shape, generation
        )

    population = toolbox.select(offspring, k=len(population))
    best_individual = tools.selBest(population, k=1)[0]
    best_accuracies.append(best_individual.fitness.values[0])

    print("Best individual so far:", best_individual)
    print(f"Best accuracy: {best_individual.fitness.values[0] * 100:.2f}%")

# Final model training
print("Training model with best parameters...")
best_individual = tools.selBest(population, k=1)[0]
hypr_shape = input_shape + len(hyprparameter_ranges)

feature_mask = best_individual[:input_shape]
hyprparameters = best_individual[input_shape:hypr_shape]
selected_indices = []

for i, bit in enumerate(feature_mask):
    if bit == 1:
        selected_indices.append(i)

X_train_selected = X_train[:, selected_indices]
X_test_selected = X_test[:, selected_indices]
selected_input_shape = len(selected_indices)

model, model_history = create_ann_model(hyprparameters, selected_input_shape)

model.save('models/ReLU_heart_model.keras')

log_data = {
    'history': model_history.history,
    'selected_features': selected_indices,
    'hyprparameters': hyprparameters
}
joblib.dump(log_data, 'models/logs/ReLU_model_logs.pkl')

# Final evaluation
y_probability = model.predict(X_test_selected)
y_prediction = (y_probability > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_prediction)

print(f"Final Test Accuracy: {accuracy * 100:.2f}%")
