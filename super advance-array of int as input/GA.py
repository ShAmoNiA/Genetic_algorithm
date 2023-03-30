import random
import math

# define the genetic algorithm parameters
POPULATION_SIZE = 100
MUTATION_RATE = 0.01
MUTATION_RATE_INCREASE = 0.001
TOURNAMENT_SIZE = 5
MIN_CHROMOSOME_LENGTH = 5
MAX_CHROMOSOME_LENGTH = 20
NUM_GENERATIONS = 100

# define the fitness function
def fitness(x):
    y = sum([math.sin(10 * math.pi * xi) / xi + math.log(xi) for xi in x]) # evaluate the function
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
    crossover_point = random.randint(1, len(parent1)-1) # choose a random crossover point between 1 and the length of the parent - 1
    offspring1 = parent1[:crossover_point] + parent2[crossover_point:] # combine the first half of parent1 and the second half of parent2
    offspring2 = parent2[:crossover_point] + parent1[crossover_point:] # combine the first half of parent2 and the second half of parent1
    return offspring1, offspring2

def two_point_crossover(parent1, parent2):
    length = min(len(parent1), len(parent2))
    if length < 3:
        return parent1, parent2
    crossover_points = random.sample(range(1, length), 2) # choose two random crossover points
    crossover_points.sort() # sort the crossover points
    offspring1 = parent1[:crossover_points[0]] + parent2[crossover_points[0]:crossover_points[1]] + parent1[crossover_points[1]:] # combine the three parts
    offspring2 = parent2[:crossover_points[0]] + parent1[crossover_points[0]:crossover_points[1]] + parent2[crossover_points[1]:] # in the same way
    return offspring1, offspring2

# define the mutation operator
def mutation(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.randint(1, 1024) # replace the value at the ith index with a random integer between 1 and 1024
    return individual

# initialize the population with random arrays of integers with variable lengths
population = []
for i in range(POPULATION_SIZE):
    chromosome_length = random.randint(MIN_CHROMOSOME_LENGTH, MAX_CHROMOSOME_LENGTH) # generate a random chromosome length within the specified range
    chromosome = [random.randint(1, 1024) for i in range(chromosome_length)] # generate a random chromosome
    population.append(chromosome)

for generation in range(NUM_GENERATIONS):
    # evaluate the fitness of each individual in the population
    fitness_scores = [fitness(individual) for individual in population]
    # find the fittest individual in the population
    fittest_individual = max(population, key=fitness)
    print(f"Generation {generation}: Best fitness = {fitness(fittest_individual)}")

    # create a new population
    new_population = [fittest_individual] # carry over the fittest individual from the previous generation

    # perform crossover and mutation to generate the rest of the new population
    while len(new_population) < POPULATION_SIZE:
        # select parents using tournament selection
        parent1 = tournament_selection(population, TOURNAMENT_SIZE)
        parent2 = tournament_selection(population, TOURNAMENT_SIZE)
        
        # perform two-point crossover to create two offspring
        offspring1, offspring2 = two_point_crossover(parent1, parent2)
        
        # mutate the offspring with a dynamic mutation rate
        if generation < NUM_GENERATIONS/2:
            mutation_rate = MUTATION_RATE + MUTATION_RATE_INCREASE * generation # increase the mutation rate as the algorithm progresses
        else:
            mutation_rate = MUTATION_RATE # use the maximum mutation rate after the halfway point
        offspring1 = mutation(offspring1, mutation_rate)
        offspring2 = mutation(offspring2, mutation_rate)
        
        # add the offspring to the new population
        new_population.append(offspring1)
        if len(new_population) < POPULATION_SIZE:
            new_population.append(offspring2)

    # set the new population as the current population
    population = new_population
    
fittest_individual = max(population, key=fitness)
print(f"Best individual: {fittest_individual}")
print(f"Fitness score: {fitness(fittest_individual)}")
