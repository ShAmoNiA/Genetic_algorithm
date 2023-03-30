import random
import math

# define the genetic algorithm parameters
POPULATION_SIZE = 100
GENE_LENGTH = 20
MUTATION_RATE = 0.01
TOURNAMENT_SIZE = 5

# define the fitness function
def fitness(individual):
    x = int("".join(map(str, individual)), 2) # convert binary string to integer
    y = math.sin(10 * math.pi * x) / x + math.log(x) # evaluate the function
    return y

# define the selection operators
def tournament_selection(population, tournament_size):
    tournament = random.sample(population, tournament_size)
    fittest_individual = max(tournament, key=fitness)
    return fittest_individual

def roulette_wheel_selection(population):
    fitness_scores = [fitness(individual) for individual in population]
    total_fitness = sum(fitness_scores)
    selection_probabilities = [fitness_score/total_fitness for fitness_score in fitness_scores]
    selected_index = random.choices(range(len(population)), weights=selection_probabilities, k=1)[0]
    return population[selected_index]

# define the crossover operators
def one_point_crossover(parent1, parent2):
    crossover_point = random.randint(0, GENE_LENGTH-1)
    offspring = parent1[:crossover_point] + parent2[crossover_point:]
    return offspring

def uniform_crossover(parent1, parent2):
    offspring = [parent1[i] if random.random() < 0.5 else parent2[i] for i in range(GENE_LENGTH)]
    return offspring

# initialize the population with random individuals
population = []
for i in range(POPULATION_SIZE):
    individual = [random.randint(0, 1) for j in range(GENE_LENGTH)]
    population.append(individual)

# evolve the population
for generation in range(100):
    # select the parents for reproduction
    parent1 = tournament_selection(population, TOURNAMENT_SIZE)
    parent2 = tournament_selection(population, TOURNAMENT_SIZE)

    # crossover the parents to create the offspring
    # use different crossover operators with equal probability
    if random.random() < 0.5:
        offspring = one_point_crossover(parent1, parent2)
    else:
        offspring = uniform_crossover(parent1, parent2)

    # mutate the offspring
    for i in range(len(offspring)):
        if random.random() < MUTATION_RATE:
            offspring[i] = 1 - offspring[i]

    # replace a random individual in the population with the offspring
    population[random.randint(0, POPULATION_SIZE-1)] = offspring

    # print the fittest individual of each generation
    fittest_individual = max(population, key=fitness)
    print("Generation {}: {}".format(generation, fittest_individual))
