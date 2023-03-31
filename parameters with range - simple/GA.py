import random

# Define parameter ranges
x_range = (0, 10)
y_range = (0, 5)
z_range = (1, 100)

# Define genetic algorithm parameters
population_size = 50
mutation_rate = 0.1
generations = 100

def generate_individual():
    # Generate a random individual
    x = random.uniform(x_range[0], x_range[1])
    y = random.uniform(y_range[0], y_range[1])
    z = random.uniform(z_range[0], z_range[1])
    return (x, y, z)

def generate_population():
    # Generate a population of random individuals
    population = []
    for i in range(population_size):
        individual = generate_individual()
        population.append(individual)
    return population

def fitness_func(inputs):
    # Define your fitness function here
    # This function should take the inputs and return a fitness score
    x, y, z = inputs
    fitness = x**2 - 2*y + z**3
    return fitness

def selection(population):
    # Select the fittest individuals from the population
    sorted_population = sorted(population, key=lambda x: fitness_func(x), reverse=True)
    selected = sorted_population[:int(0.5*population_size)]
    return selected

def crossover(parent1, parent2):
    # Create a new individual by combining the genes of two parents
    child = []
    for i in range(len(parent1)):
        if random.random() < 0.5:
            child.append(parent1[i])
        else:
            child.append(parent2[i])
    return tuple(child)

def mutation(individual):
    # Mutate an individual by randomly adjusting one of its genes
    new_individual = list(individual)
    gene_to_mutate = random.randint(0, len(individual)-1)
    current_value = new_individual[gene_to_mutate]
    if isinstance(current_value, int):
        new_value = random.randint(x_range[0], x_range[1])
    else:
        new_value = random.uniform(current_value - (current_value*mutation_rate), current_value + (current_value*mutation_rate))
        if new_value < x_range[0]:
            new_value = x_range[0]
        elif new_value > x_range[1]:
            new_value = x_range[1]
    new_individual[gene_to_mutate] = new_value
    return tuple(new_individual)

def evolve(population):
    # Create the next generation of individuals through selection, crossover, and mutation
    selected = selection(population)
    children = []
    for i in range(population_size - len(selected)):
        parent1 = random.choice(selected)
        parent2 = random.choice(selected)
        child = crossover(parent1, parent2)
        if random.random() < mutation_rate:
            child = mutation(child)
        children.append(child)
    selected.extend(children)
    return selected

def genetic_algorithm():
    # Run the genetic algorithm
    population = generate_population()
    for i in range(generations):
        population = evolve(population)
        print("Generation ", i+1, " - Best Fitness: ", fitness_func(population[0]), " - Best Individual: ", population[0])
    print("Final Population: ", population)
    best_individual = max(population, key=lambda x: fitness_func(x))
    print("Best Individual: ", best_individual, " - Fitness: ", fitness_func(best_individual))

if __name__ == '__main__':
    genetic_algorithm()
