import random

# define the genetic algorithm parameters
POPULATION_SIZE = 100
GENE_LENGTH = 20
MUTATION_RATE = 0.01

# define the fitness function
def fitness(individual):
    return sum(individual)

# initialize the population with random individuals
population = []
for i in range(POPULATION_SIZE):
    individual = [random.randint(0, 1) for j in range(GENE_LENGTH)]
    population.append(individual)

# evolve the population
for generation in range(100):
    # evaluate the fitness of each individual
    fitness_scores = [fitness(individual) for individual in population]

    # select the parents for reproduction
    parent1 = population[fitness_scores.index(max(fitness_scores))]
    parent2 = population[fitness_scores.index(sorted(fitness_scores)[-2])]

    # crossover the parents to create the offspring
    crossover_point = random.randint(0, GENE_LENGTH-1)
    offspring = parent1[:crossover_point] + parent2[crossover_point:]

    # mutate the offspring
    for i in range(len(offspring)):
        if random.random() < MUTATION_RATE:
            offspring[i] = 1 - offspring[i]

    # replace a random individual in the population with the offspring
    population[random.randint(0, POPULATION_SIZE-1)] = offspring

    # print the fittest individual of each generation
    fittest_individual = population[fitness_scores.index(max(fitness_scores))]
    print("Generation {}: {}".format(generation, fittest_individual))