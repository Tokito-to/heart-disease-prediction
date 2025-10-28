import os

from silence_tensorflow import silence_tensorflow
silence_tensorflow()

from src.data import dataset_info, load_dataset, test_train_split  # noqa: E402
from src.config import parameters, feature_importance  # noqa: E402
from src.config import smart_individual, ga_parameters  # noqa: E402
from src.ga import evaluate, repair  # noqa: E402
# from src.ga import ga_setup, ga_run  # noqa: E402

from deap import base, creator, tools, algorithms  # noqa: E402

script_dir = os.path.dirname(__file__)
os.chdir(script_dir)

global ranked_features, input_shape
dataset = parameters["dataset"]
feature_names, target_names = dataset_info(dataset)
input_shape = len(feature_names)
target_shape = len(target_names)

# Load Dataset
data, X, y = load_dataset(dataset, feature_names, target_names)
X_train, y_train, X_test, y_test = test_train_split(X, y)

ranked_features = feature_importance(X_train, y_train)

# GA setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# toolbox = ga_setup(evaluate, repair, ranked_features, input_shape)
toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, smart_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

generations = ga_parameters["generation"]
population_size = ga_parameters["population"]
best_accuracies = []

population = toolbox.population(n=population_size)

print("Starting Genetic Algorithm Optimization...\n")

for generation in range(generations):
    generation += 1
    print(f"=== Generation {generation} ===")

    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
    offspring = list(map(repair, offspring))    # Evaluate offspring

    for ind in offspring:
        ind.fitness.values = toolbox.evaluate(ind, input_shape, X_train, y_train, X_test, y_test)

    population = toolbox.select(offspring, len(population))

    # Track best individual
    best_individual = tools.selBest(population, 1)[0]
    best_accuracies.append(best_individual.fitness.values[0])

    print("Best individual so far:", best_individual)
    print(f"Best accuracy: {best_individual.fitness.values[0] * 100:.2f}%")


print(f"Best Individual: {best_individual}")
