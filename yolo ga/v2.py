import random
import threading



# define the fitness function to be maximized
def fitness_function(inputs):
    # replace with your own fitness function
    return sum(inputs)

# define the function to evaluate the fitness of a subset of the population
def evaluate_fitness(population_subset, fitness_scores, start_index):
    for i, individual in enumerate(population_subset):
        fitness_scores[start_index + i] = fitness_function(individual)

# define the genetic algorithm function
def genetic_algorithm(population_size, input_info, generations):
    # unpack the input info into separate lists
    input_ranges = [info[0:2] for info in input_info]
    mutation_rates = [info[2] for info in input_info]
    crossover_rates = [info[3] for info in input_info]

    # initialize the population with random values within the specified ranges
    population = []
    for i in range(population_size):
        individual = [random.uniform(r[0], r[1]) for r in input_ranges]
        population.append(individual)

    # evolve the population for the specified number of generations
    for i in range(generations):
        # evaluate the fitness of each individual in the population in parallel
        fitness_scores = [0] * population_size
        threads = []
        num_threads = 4
        chunk_size = population_size // num_threads
        for j in range(num_threads):
            start_index = j * chunk_size
            end_index = start_index + chunk_size
            if j == num_threads - 1:
                end_index = population_size
            population_subset = population[start_index:end_index]
            thread = threading.Thread(target=evaluate_fitness, args=(population_subset, fitness_scores, start_index))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

        # select the fittest individuals to be parents for the next generation
        parents = []
        for j in range(population_size//2):
            index1 = random.choices(range(population_size), weights=fitness_scores)[0]
            index2 = random.choices(range(population_size), weights=fitness_scores)[0]
            parents.append((population[index1], population[index2]))

        # create offspring by performing crossover and mutation
        offspring = []
        for parent1, parent2 in parents:
            child = []
            for k in range(len(input_ranges)):
                if random.random() < crossover_rates[k]:
                    if random.random() < 0.5:
                        child.append(parent1[k])
                    else:
                        child.append(parent2[k])
                else:
                    child.append(random.uniform(input_ranges[k][0], input_ranges[k][1]))
            # apply mutation to the child
            for k in range(len(input_ranges)):
                if random.random() < mutation_rates[k]:
                    child[k] = random.uniform(input_ranges[k][0], input_ranges[k][1])
            offspring.append(child)

        # replace the old population with the new offspring
        population = offspring

    # evaluate the final fitness scores and return the fittest individual
    fitness_scores = [fitness_function(individual) for individual in population]
    return population[fitness_scores.index(max(fitness_scores))]

#
# define the population size and number of generations
population_size = 100
generations = 50
# define the ranges and mutation/crossover rates for each input
input_info = [(0.0, 1.0, 0.1, 0.5), (10.0, 20.0, 0.05, 0.8), (30.0, 50.0, 0.2, 0.6)]

# run the genetic algorithm
best_individual = genetic_algorithm(population_size, input_info, generations)

# print the results
print("Best individual:", best_individual)
print("Fitness score:", fitness_function(best_individual))
