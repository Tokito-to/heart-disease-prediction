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
X = data.iloc[:, :13].values
y = data["target"].values

# Undersample dataset
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

# ANN
def create_ann_model(params):
    clear_session()
    n1, n2, n3, lr, dr, l2_reg, alpha = params

    model = Sequential()
    model.add(InputLayer(shape=(X_train.shape[1],)))

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
    model.fit(X_train, y_train, validation_split=0.30, epochs=145, batch_size=35, callbacks=[early_stopping, reduce_lr], verbose=0)
    return model

# Logging
log_file = 'models/logs/ReLU_model.csv'
with open(log_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Generation', 'n1', 'n2', 'n3', 'lr', 'dr', 'l2', 'alpha', 'Accuracy', 'TP', 'FP', 'FN', 'TN'])

# Evaluation function with logging
def evaluate(individual, generation=0):
    model = create_ann_model(individual)

    y_pred = model.predict(X_test)
    y_pred_bin = (y_pred > 0.45).astype(int)

    acc = accuracy_score(y_test, y_pred_bin)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_bin).ravel()

    # Log individual result
    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            generation,
            int(individual[0]), int(individual[1]), int(individual[2]),
            round(individual[3], 6), round(individual[4], 4), round(individual[5], 6),
            round(individual[6], 4),
            round(acc, 4), tp, fp, fn, tn
        ])

    return (acc,)

# Hyperparameter ranges
param_ranges = {
    "n1": (100, 200),
    "n2": (50, 150),
    "n3": (30, 100),
    "lr": (0.0001, 0.01),
    "dr": (0.01, 0.3),
    "l2": (0.0001, 0.01),
    "alpha": (0.01, 0.3)
}

# Generate random individual
def random_param():
    def safe_uniform(low, high):
        return max(0.00001, random.uniform(max(0, low), max(0, high)))

    return [
        random.randint(*param_ranges["n1"]),     # n1: neurons layer 1
        random.randint(*param_ranges["n2"]),     # n2: neurons layer 2
        random.randint(*param_ranges["n3"]),     # n3: neurons layer 3
        safe_uniform(*param_ranges["lr"]),       # lr: learning rate
        safe_uniform(*param_ranges["dr"]),       # dr: dropout rate
        safe_uniform(*param_ranges["l2"]),       # l2: L2 regularization
        safe_uniform(*param_ranges["alpha"])     # alpha: PReLU alpha
    ]

# Repair mutated individuals to stay within bounds
def repair(individual):
    individual[0] = int(np.clip(individual[0], *param_ranges["n1"]))
    individual[1] = int(np.clip(individual[1], *param_ranges["n2"]))
    individual[2] = int(np.clip(individual[2], *param_ranges["n3"]))
    individual[3] = float(np.clip(individual[3], *param_ranges["lr"]))
    individual[4] = float(np.clip(individual[4], *param_ranges["dr"]))
    individual[5] = float(np.clip(individual[5], *param_ranges["l2"]))
    individual[6] = float(np.clip(individual[6], *param_ranges["alpha"]))
    return individual

# GA setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, random_param)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# Run GA
population = toolbox.population(n=50)  # population size
N_GENS = 10                            # number of generations
best_accuracies = []

print("Starting Genetic Algorithm Optimization...\n")

for gen in range(N_GENS):
    print(f"=== Generation {gen+1} ===")

    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
    offspring = list(map(repair, offspring))

    # Evaluate with logging
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
final_model = create_ann_model(best_params)

final_model.save('models/ReLU_heart_model.keras')

# Final evaluation
y_final_pred = final_model.predict(X_test)
y_final_pred = (y_final_pred > 0.5).astype(int)
final_acc = accuracy_score(y_test, y_final_pred)

print("Final Test Accuracy: {:.2f}%".format(final_acc * 100))
