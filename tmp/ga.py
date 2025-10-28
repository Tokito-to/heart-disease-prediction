import random
import numpy as np

from src.config import hyprparameter_ranges, ga_parameters, smart_individual
from src.model import create_ann_model
from deap import base, creator, tools, algorithms

from sklearn.metrics import accuracy_score


# Evaluation
def evaluate(individual, input_shape, X_train, y_train, X_test, y_test):
    hyperparams = individual[input_shape:]
    feature_mask = individual[:input_shape]
    selected_indices = []

    for i, bit in enumerate(feature_mask):
        if bit == 1:
            selected_indices.append(i)

    input_shape = len(selected_indices)

    X_train_selected = X_train[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]

    model = create_ann_model(
        hyperparams, True, input_shape,
        X_train_selected, y_train
    )
    y_probability = model.predict(X_test_selected)
    y_prediction = (y_probability > 0.5).astype(int)

    acc = accuracy_score(y_test, y_prediction)

    return (acc,)


# Repair Function
def repair(individual, ranked_features, input_shape):
    hypr_shape = input_shape + len(hyprparameter_ranges)
    
    # Split Individual
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

    # Merge back to indicidual
    individual[:input_shape] = feature_mask
    individual[input_shape:hypr_shape] = hyprparameter

    return individual


# GA Setup
def ga_setup(evaluate, repair, ranked_features, input_shape):
    toolbox = base.Toolbox()
    toolbox.register(
        "individual", tools.initIterate, creator.Individual, smart_individual
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)

    return toolbox


# GA Run Function
def ga_run(toolbox, ranked_features, X_train, y_train, X_test, y_test, input_shape):
    generations = ga_parameters["generation"]
    population_size = ga_parameters["population"]
    best_accuracies = []

    population = toolbox.population(n=population_size)

    print("Starting Genetic Algorithm Optimization...\n")

    for generation in range(generations):
        generation += 1
        print(f"=== Generation {generation} ===")

        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
        for i, individual in enumerate(offspring):
            offspring[i] = repair(individual, ranked_features, input_shape)

        # Evaluate offspring
        for ind in offspring:
            ind.fitness.values = toolbox.evaluate(
                ind, input_shape,
                X_train, y_train, X_test, y_test,
            )

        population = toolbox.select(offspring, len(population))

        # Track best individual
        best_individual = tools.selBest(population, 1)[0]
        best_accuracies.append(best_individual.fitness.values[0])

        print("Best individual so far:", best_individual)
        print(f"Best accuracy: {best_individual.fitness.values[0] * 100:.2f}%")

    return best_accuracies, best_individual
