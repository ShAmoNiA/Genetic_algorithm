import random
import concurrent.futures

class GeneticAlgorithm:
    def __init__(self, population_size, individual_length, num_generations, input_ranges):
        self.population_size = population_size
        self.individual_length = individual_length
        self.num_generations = num_generations
        self.input_ranges = input_ranges
        
    def generate_individual(self):
        individual = []
        for i in range(self.individual_length):
            lower_bound, upper_bound = self.input_ranges[i]
            individual.append(random.uniform(lower_bound, upper_bound))
        return individual

    def fitness(self, individual):
        return sum(individual)

    def selection(self, population):
        fitnesses = [self.fitness(individual) for individual in population]
        total_fitness = sum(fitnesses)
        probabilities = [fitness / total_fitness for fitness in fitnesses]
        return random.choices(population, probabilities, k=2)

    def crossover(self, parent1, parent2, crossover_rates):
        child1 = parent1[:]
        child2 = parent2[:]
        for i in range(self.individual_length):
            if random.random() < crossover_rates[i]:
                for j in range(self.individual_length):
                    if random.random() < 0.5:
                        child1[j], child2[j] = child2[j], child1[j]
        return child1, child2

    def mutate(self, individual, mutation_rates, mutation_stddev):
        for i in range(self.individual_length):
            lower_bound, upper_bound = self.input_ranges[i]
            if random.random() < mutation_rates[i]:
                current_value = individual[i]
                mutation_value = random.gauss(0, mutation_stddev[i])
                new_value = current_value + mutation_value
                new_value = max(lower_bound, min(upper_bound, new_value))
                individual[i] = new_value

    def run(self, mutation_stddev, crossover_rates, mutation_rates, elitism=True, num_threads=4):
        population = [self.generate_individual() for i in range(self.population_size)]
        best_fitness = 0
        num_generations_without_improvement = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            for generation in range(self.num_generations):
                new_population = []
                if elitism:
                    # Carry over the best individual from the current generation to the next generation
                    best_individual = max(population, key=self.fitness)
                    new_population.append(best_individual)
                    # Add one more individual from the current generation to the next generation
                    # (the second best individual, to avoid excessive elitism)
                    second_best_individual = max([individual for individual in population if individual != best_individual], key=self.fitness)
                    new_population.append(second_best_individual)
                for i in range(int((self.population_size - elitism) / 2)):
                    parent1, parent2 = self.selection(population)
                    crossover_rates = crossover_rates
                    child1, child2 = self.crossover(parent1, parent2, crossover_rates)
                    mutation_rates = mutation_rates
                    executor.submit(self.mutate, child1, mutation_rates, mutation_stddev)
                    executor.submit(self.mutate, child2, mutation_rates, mutation_stddev)
                    new_population.append(child1)
                    new_population.append(child2)
                population = new_population
                best_individual = max(population, key=self.fitness)
                best_fitness_current = self.fitness(best_individual)
                if best_fitness_current <= best_fitness:
                    num_generations_without_improvement += 1
                else:
                    best_fitness = best_fitness_current
                    num_generations_without_improvement = 0
                if num_generations_without_improvement >= 5: # stop if no improvement in 5 generations
                    break
        best_individual = max(population, key=self.fitness)
        return best_individual


# define the size of the population, the length of the individuals, and the number of generations
population_size = 1000
individual_length = 4
num_generations = 5000

# define the list of ranges for each input
input_ranges = [(0.0, 10.0), (-5.0, 5.0), (0.0, 1.0), (1.0, 100.0)]

# run the genetic algorithm and print the result
mutation_stddev = [random.uniform(0.1, 0.9) for i in range(individual_length)]
crossover_rates = [random.uniform(0.1, 0.9) for i in range(individual_length)]
mutation_rates = [random.uniform(0.01, 0.1) for i in range(individual_length)]
ga = GeneticAlgorithm(population_size, individual_length, num_generations, input_ranges) 
result = ga.run(mutation_stddev=mutation_stddev,crossover_rates=crossover_rates,mutation_rates=mutation_rates)
print(result)
