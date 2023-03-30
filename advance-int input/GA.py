import random
import math

# define the genetic algorithm parameters
POPULATION_SIZE = 100
MUTATION_RATE = 0.01
TOURNAMENT_SIZE = 5

# define the fitness function
def fitness(x):
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
    crossover_point = random.randint(1, len(bin(parent1))-2) # choose a random crossover point between 1 and the length of the binary representation of parent1 - 1
    mask = int('0b' + '1' * crossover_point, 2) # create a mask to extract the bits up to the crossover point
    offspring1 = (parent1 & mask) | (parent2 & ~mask) # combine the bits of parent1 and parent2 using the mask
    offspring2 = (parent2 & mask) | (parent1 & ~mask) # combine the bits of parent2 and parent1 using the mask
    return offspring1, offspring2

# initialize the population with random integers
population = []
for i in range(POPULATION_SIZE):
    individual = random.randint(1, 1024)
    population.append(individual)

# evolve the population
for generation in range(100):
    # select the parents for reproduction
    parent1 = tournament_selection(population, TOURNAMENT_SIZE)
    parent2 = tournament_selection(population, TOURNAMENT_SIZE)

    # crossover the parents to create the offspring
    offspring1, offspring2 = one_point_crossover(parent1, parent2)

    # mutate the offspring
    if random.random() < MUTATION_RATE:
        offspring1 = offspring1 ^ (1 << random.randint(0, 9)) # flip a random bit in the binary representation of the offspring
    if random.random() < MUTATION_RATE:
        offspring2 = offspring2 ^ (1 << random.randint(0, 9)) # flip a random bit in the binary representation of the offspring

    # replace the two least fit individuals in the population with the offspring
    population.sort(key=fitness, reverse=True)
    population[-1], population[-2] = offspring1, offspring2

    # print the fittest individual of each generation
    fittest_individual = max(population, key=fitness)
    print("Generation {}: {}".format(generation, fittest_individual))
