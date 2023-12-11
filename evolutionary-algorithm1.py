#Evolutionary Algorithm Study
#Created by Tsubasa Kato using ChatGPT (GPT-4)
#(C)Tsubasa Kato - Inspire Search Corporation 2023/12/11 23:02PM
#Company Website: https://www.inspiresearch.io/en
import random

# Define parameters for the evolutionary algorithm
POPULATION_SIZE = 100
MAX_GENERATIONS = 50
MUTATION_RATE = 0.1
TOURNAMENT_SIZE = 5

# Target function: 2x + 3
def target_function(x):
    return 2 * x + 3

# Generate a random linear function (ax + b)
def random_function():
    a = random.uniform(-10, 10)
    b = random.uniform(-10, 10)
    return (a, b)

# Evaluate the fitness of a function (lower is better)
def fitness(function):
    error = 0
    for x in range(-10, 11):  # Sample points
        predicted = function[0] * x + function[1]
        actual = target_function(x)
        error += abs(predicted - actual)
    return error

# Tournament selection
def tournament_selection(population):
    selected = []
    for _ in range(POPULATION_SIZE):
        tournament = random.sample(population, TOURNAMENT_SIZE)
        winner = min(tournament, key=lambda individual: individual[1])
        selected.append(winner[0])
    return selected

# Crossover (simple one-point crossover)
def crossover(parent1, parent2):
    if random.random() < 0.5:
        return (parent1[0], parent2[1])
    else:
        return (parent2[0], parent1[1])

# Mutation (small random changes)
def mutate(function):
    a, b = function
    if random.random() < MUTATION_RATE:
        a += random.uniform(-1, 1)
    if random.random() < MUTATION_RATE:
        b += random.uniform(-1, 1)
    return (a, b)

# Evolutionary algorithm
def evolutionary_algorithm():
    # Initialize population
    population = [(random_function(), 0) for _ in range(POPULATION_SIZE)]

    for generation in range(MAX_GENERATIONS):
        # Evaluate fitness
        population = [(individual, fitness(individual)) for individual, _ in population]

        # Selection
        selected = tournament_selection(population)

        # Crossover and Mutation
        next_generation = []
        for i in range(0, POPULATION_SIZE, 2):
            parent1, parent2 = selected[i], selected[i + 1]
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            next_generation.append(mutate(child1))
            next_generation.append(mutate(child2))

        population = [(individual, 0) for individual in next_generation]

        # Report best solution of this generation
        # Corrected line for reporting the best solution of each generation
        best_individual = min(population, key=lambda individual: fitness(individual[0]))
        print(f"Generation {generation + 1}: Best Function = {best_individual[0]}, Fitness = {fitness(best_individual[0])}")


    return min(population, key=lambda individual: fitness(individual[0]))[0]

# Run the algorithm
best_program = evolutionary_algorithm()
print("Best program:", best_program)
