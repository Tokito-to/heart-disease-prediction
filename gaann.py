from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import csv
import random

import numpy as np
import pandas as pd

import tensorflow as tf

from keras.backend import clear_session
from keras.models import Sequential
from keras.layers import ReLU, LeakyReLU, PReLU, ELU, Activation
from keras.layers import Dense, Dropout, BatchNormalization, InputLayer
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.metrics import AUC

from deap import base, creator, tools, algorithms

from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
data = pd.read_csv('heart_dataset.csv')
feature_names = list(data.columns[:12])
X = data.iloc[:, :12].values
y = data["target"].values

# Oversample using SMOTE
from imblearn.over_sampling import SMOTE
X, y = SMOTE(random_state=42).fit_resample(X, y)

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Dump scaler # We need this to use pre-trained model
# pre-trained model require same scaler as training scaler
import joblib
joblib.dump(sc, 'models/scaler.pkl')

# Reduce Learning and Early stop
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001, verbose=0)

# Compute feature importance (RF + MI)
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import minmax_scale

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

explainer = shap.Explainer(rf, X_train, feature_names=[f"F{i}" for i in range(X_train.shape[1])])
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

# Model builder
def create_ann_model(params, fit_model=True, input_shape=None):
    clear_session()
    n1, n2, n3, lr, dr, l2_reg, alpha = params

    model = Sequential()
    model.add(InputLayer(shape=(input_shape,)))

    model.add(Dense(int(n1), kernel_regularizer=l2(l2_reg)))
    model.add(ReLU())
    model.add(BatchNormalization())
    model.add(Dropout(dr))

    model.add(Dense(int(n2), kernel_regularizer=l2(l2_reg)))
    model.add(ReLU())
    model.add(BatchNormalization())
    model.add(Dropout(dr))

    model.add(Dense(int(n3), kernel_regularizer=l2(l2_reg)))
    model.add(ReLU())
    model.add(BatchNormalization())
    model.add(Dropout(dr))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy', AUC(name='auc')])

    if fit_model:
        model.fit(X_train_selected, y_train, validation_split=0.30, epochs=145, batch_size=35, callbacks=[early_stopping, reduce_lr], verbose=0)

    return model

# Logging
log_file = 'models/logs/ReLU_model.csv'
with open(log_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Generation', 'SelectedFeatures', 'n1', 'n2', 'n3', 'lr', 'dr', 'l2', 'alpha', 'Accuracy', 'TP', 'FP', 'FN', 'TN'])

# Evaluation
def evaluate(individual, generation=0):
    feature_mask = individual[:12]
    selected_features = [feature_names[i] for i, bit in enumerate(feature_mask) if bit == 1]
    hyperparams = individual[12:]

    selected_indices = [i for i, bit in enumerate(feature_mask) if bit == 1]
    if not selected_indices:
        return (0.0,)

    global X_train_selected, X_test_selected
    X_train_selected = X_train[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]

    model = create_ann_model(hyperparams, fit_model=True, input_shape=len(selected_indices))
    y_pred = model.predict(X_test_selected)
    y_pred_bin = (y_pred > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred_bin)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_bin).ravel()

    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([ generation, ','.join(selected_features), int(hyperparams[0]),
                  int(hyperparams[1]), int(hyperparams[2]), round(hyperparams[3], 6),
                  round(hyperparams[4], 4), round(hyperparams[5], 6),
                  round(hyperparams[6], 4), round(acc, 4), tp, fp, fn, tn ])


    return (acc,)

# Hyperparameter ranges
param_ranges = {
    "n1": (128, 192),
    "n2": (64, 128),
    "n3": (32, 64),
    "lr": (0.0001, 0.01),
    "dr": (0.0, 0.4),
    "l2": (0.000001, 0.01),
    "alpha": (0.01, 0.3)
}

# Hyperparam generator
def safe_uniform(low, high):
    return max(0.00001, random.uniform(max(0, low), max(0, high)))

def random_param():
    return [
        random.randint(*param_ranges["n1"]),     # n1: neurons layer 1
        random.randint(*param_ranges["n2"]),     # n2: neurons layer 2
        random.randint(*param_ranges["n3"]),     # n3: neurons layer 3
        safe_uniform(*param_ranges["lr"]),       # lr: learning rate
        safe_uniform(*param_ranges["dr"]),       # dr: dropout rate
        safe_uniform(*param_ranges["l2"]),       # l2: L2 regularization
        safe_uniform(*param_ranges["alpha"])     # alpha: PReLU alpha
    ]

# Feature selection mask using top-K ranked features
def smart_initial_feature_mask(k=8):
    mask = [0] * 12
    for idx in ranked_features[:k]:
        mask[idx] = 1
    return mask

def smart_individual():
    feature_mask = smart_initial_feature_mask(k=random.randint(6, 10))  # add some randomness
    hyperparams = random_param()
    return feature_mask + hyperparams

# Repair function
def repair(individual):
    for i in range(12):
        individual[i] = int(individual[i]) if individual[i] in [0, 1] else 1
    if sum(individual[:12]) == 0:
        individual[random.randint(0, 11)] = 1
    individual[12] = int(np.clip(individual[12], *param_ranges["n1"]))
    individual[13] = int(np.clip(individual[13], *param_ranges["n2"]))
    individual[14] = int(np.clip(individual[14], *param_ranges["n3"]))
    individual[15] = float(np.clip(individual[15], *param_ranges["lr"]))
    individual[16] = float(np.clip(individual[16], *param_ranges["dr"]))
    individual[17] = float(np.clip(individual[17], *param_ranges["l2"]))
    individual[18] = float(np.clip(individual[18], *param_ranges["alpha"]))
    return individual

# GA setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, smart_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# Run GA
population = toolbox.population(n=64)  # population size
N_GENS = 8                             # number of generations
best_accuracies = []

print("Starting Genetic Algorithm Optimization...\n")

for gen in range(N_GENS):
    print(f"=== Generation {gen+1} ===")

    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
    offspring = list(map(repair, offspring))

    for ind in offspring:
        ind.fitness.values = evaluate(ind, generation=gen+1)

    population = toolbox.select(offspring, k=len(population))
    best_ind = tools.selBest(population, k=1)[0]
    best_accuracies.append(best_ind.fitness.values[0])

    print("Best individual so far:", best_ind)
    print("Best accuracy: {:.2f}%\n".format(best_ind.fitness.values[0] * 100))

# Final model training
print("Training model with best parameters...")

best_params = tools.selBest(population, k=1)[0]
feature_mask = best_params[:12]
hyperparams = best_params[12:]

selected_indices = [i for i, bit in enumerate(feature_mask) if bit == 1]
X_train_selected = X_train[:, selected_indices]
X_test_selected = X_test[:, selected_indices]

model = create_ann_model(hyperparams, fit_model=False, input_shape=len(selected_indices))
model_history = model.fit(X_train_selected, y_train, validation_split=0.30, epochs=145, batch_size=35,
                          callbacks=[early_stopping, reduce_lr], verbose=0)

model.save('models/ReLU_heart_model.keras')

log_data = {
    'history': model_history.history,
    'selected_features': selected_indices,
    'hyperparams': hyperparams
}
joblib.dump(log_data, 'models/logs/ReLU_model_logs.pkl')

# Final evaluation
y_prob = model.predict(X_test_selected)
y_pred = (y_prob > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
print("Final Test Accuracy: {:.2f}%".format(accuracy * 100))

