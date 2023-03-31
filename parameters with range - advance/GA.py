import random

class GeneticAlgorithm:
    
    def __init__(self, fitness_func, num_genes, gene_range, pop_size=50, mutation_rate=0.1, crossover_rate=0.9):
        self.fitness_func = fitness_func
        self.num_genes = num_genes
        self.gene_range = gene_range
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = self.initialize_population()

    
    def initialize_population(self):
        population = []
        for i in range(self.pop_size):
            individual = []
            for j in range(self.num_genes):
                gene = random.uniform(self.gene_range[j][0], self.gene_range[j][1])
                individual.append(gene)
            population.append(individual)
        return population

    def calculate_fitness(self, individual):
        return self.fitness_func(individual)

    def evaluate_population(self):
        fitness_scores = []
        for individual in self.population:
            fitness_score = self.calculate_fitness(individual)
            fitness_scores.append(fitness_score)
        return fitness_scores

    def tournament_selection(self, population, fitness_scores):
        selected_parents = []
        for i in range(2):
            tournament = random.sample(range(len(population)), k=5)
            tournament_fitness_scores = [fitness_scores[j] for j in tournament]
            winner = tournament_fitness_scores.index(max(tournament_fitness_scores))
            winner_index = tournament[winner]
            selected_parents.append(population[winner_index])
        return selected_parents

    def uniform_crossover(self, parent1, parent2):
        child = []
        for i in range(self.num_genes):
            if random.random() < 0.5:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        return child

    def arithmetic_crossover(self, parent1, parent2):
        alpha = random.uniform(0, 1)
        child = []
        for i in range(self.num_genes):
            child.append(alpha * parent1[i] + (1 - alpha) * parent2[i])
        return child

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            return self.arithmetic_crossover(parent1, parent2)
        else:
            return self.uniform_crossover(parent1, parent2)

    def mutate(self, individual):
        for i in range(self.num_genes):
            if random.random() < self.mutation_rate:
                individual[i] += random.uniform(-1, 1)
                individual[i] = max(min(individual[i], self.gene_range[i][1]), self.gene_range[i][0])
        return individual

    def get_new_population(self, population, fitness_scores):
        new_population = []
        for i in range(self.pop_size):
            parent1, parent2 = self.tournament_selection(population, fitness_scores)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        return new_population

    def run(self, num_generations):
        for i in range(num_generations):
            fitness_scores = self.evaluate_population()
            best_fitness_score = max(fitness_scores)
            best_individual = self.population[fitness_scores.index(best_fitness_score)]
            print(f"Generation {i}: Best fitness score = {best_fitness_score:.2f}")
            self.population = self.get_new_population(self.population, fitness_scores)
        return best_individual, best_fitness_score

def my_fitness_function(x):
    return -x[0]**2 - x[1]**3 + x[2]**2


gene_ranges = [(-100, -10),(-1, 10),(-10, 20)]
ga = GeneticAlgorithm(fitness_func=my_fitness_function, num_genes=3, gene_range=gene_ranges, pop_size=50, mutation_rate=0.1)
best_individual, best_fitness = ga.run(num_generations=50)
print(f"Best individual: {best_individual}")
print(f"Best fitness: {best_fitness}")
