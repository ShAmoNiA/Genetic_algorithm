import random
import numpy as np

class GeneticAlgorithm:
    
    def __init__(self, fitness_func, num_genes, gene_range, pop_size=50, mutation_rate=0.1, crossover_rate=0.9,extra_prop=None):
        """
        Initialize a new GeneticAlgorithm instance.

        Args:
            fitness_func (function): The fitness function to optimize.
            num_genes (int): The number of genes in each individual.
            gene_range (list): The range of values for each gene, as a list of tuples (min, max).
            pop_size (int): The size of the population. Default is 50.
            mutation_rate (float): The probability of a gene mutation. Default is 0.1.
            crossover_rate (float): The probability of a crossover operation. Default is 0.9.
        """
        self.fitness_func = fitness_func
        self.num_genes = num_genes
        self.gene_range = gene_range
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = self.initialize_population()
        self.extra_prop = extra_prop

    
    def initialize_population(self,initial_values=None):
        """
        Create a new population of individuals with random genes.

        Returns:
            list: The new population.
        """
        population = []
        for _ in range(self.pop_size):
            individual = []
            for j in range(self.num_genes):
                if initial_values is not None:
                    gene = initial_values[j]
                else:
                    gene = random.uniform(self.gene_range[j][0], self.gene_range[j][1])
                individual.append(gene)
            population.append(individual)
        return population


    def calculate_fitness(self, individual):
        """
        Calculate the fitness score of an individual.

        Args:
            individual (list): The individual to evaluate.

        Returns:
            float: The fitness score of the individual.
        """
        return self.fitness_func(individual,self.extra_prop)


    def evaluate_population(self):
        """
        Evaluate the fitness of each individual in the current population.

        Returns:
            list: The fitness scores of the population.
        """
        fitness_scores = []
        for individual in self.population:
            fitness_score = self.calculate_fitness(individual)
            fitness_scores.append(fitness_score)
        return fitness_scores


    def tournament_selection(self, population, fitness_scores, tournament_size=5):
        """
        Perform tournament selection to choose two parents.

        Args:
            population (list): The current population.
            fitness_scores (list): The fitness scores of the population.
            tournament_size (int): The number of individuals to include in each tournament.

        Returns:
            list: The two selected parents.
        """
        selected_parents = []
        for i in range(2):
            tournament = random.sample(range(len(population)), k=tournament_size)
            tournament_fitness_scores = [fitness_scores[j] for j in tournament]
            winner_index = tournament[tournament_fitness_scores.index(max(tournament_fitness_scores))]
            selected_parents.append(population[winner_index])
        return selected_parents



    def uniform_crossover(self, parent1, parent2):
        """
        Perform uniform crossover between two parents.

        Args:
            parent1 (list or tuple): The first parent.
            parent2 (list or tuple): The second parent.

        Returns:
            list: The offspring produced by uniform crossover.
        """
        child = []
        for i in range(self.num_genes):
            if random.random() < 0.5:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        return child


    def arithmetic_crossover(self, parent1, parent2):
        """
        Perform arithmetic crossover between two parents.

        Args:
            parent1 (list or tuple): The first parent.
            parent2 (list or tuple): The second parent.

        Returns:
            list: The offspring produced by arithmetic crossover.
        """
        alpha = random.uniform(0, 1)
        child = []
        for i in range(self.num_genes):
            child.append(alpha * parent1[i] + (1 - alpha) * parent2[i])
        return child

    

    def single_point_crossover(parent1, parent2):
        """
        Perform single-point crossover between two parents.

        Args:
            parent1 (list or tuple): The first parent.
            parent2 (list or tuple): The second parent.

        Returns:
            list: The offspring produced by single-point crossover.

        Raises:
            ValueError: If the parents are of different lengths.
        """
        if len(parent1) != len(parent2):
            raise ValueError("Parents must be of equal length.")

        crossover_point = random.randint(1, len(parent1) - 1)

        offspring = parent1[:crossover_point] + parent2[crossover_point:]

        return offspring


    def multi_point_crossover(parent1, parent2, num_points=2):
        """
        Perform multi-point crossover between two parents.

        Args:
            parent1 (list or tuple): The first parent.
            parent2 (list or tuple): The second parent.
            num_points (int): The number of crossover points to use.

        Returns:
            list: The offspring produced by multi-point crossover.

        Raises:
            ValueError: If the parents are of different lengths or if the number of crossover points is greater than or equal to the length of the parents.
        """
        if len(parent1) != len(parent2):
            raise ValueError("Parents must be of equal length.")

        if num_points >= len(parent1):
            raise ValueError("Number of crossover points must be less than the length of the parents.")

        crossover_points = sorted(random.sample(range(1, len(parent1) - 1), num_points))

        offspring = []
        for i, point in enumerate(crossover_points):
            if i == 0:
                offspring += parent1[:point]
            elif i % 2 == 0:
                offspring += parent1[crossover_points[i-2]:point]
            else:
                offspring += parent2[crossover_points[i-1]:point]
        offspring += parent2[crossover_points[-1]:]

        return offspring


    def crossover(self, parent1, parent2, crossover_probability=0.5, crossover_type="uniform"):
        if not isinstance(parent1, (list, tuple, np.ndarray)) or not isinstance(parent2, (list, tuple, np.ndarray)):
            raise ValueError("Parents must be of type list, tuple or ndarray.")

        if len(parent1) != len(parent2):
            raise ValueError("Parents must be of equal length.")

        if crossover_type not in ["uniform", "arithmetic", "single-point", "multi-point"]:
            raise ValueError("Invalid crossover type.")

        if not 0 <= crossover_probability <= 1:
            raise ValueError("Crossover probability must be between 0 and 1.")

        if crossover_type == "uniform":
            offspring = self.uniform_crossover(parent1, parent2)
        elif crossover_type == "arithmetic":
            offspring = self.arithmetic_crossover(parent1, parent2)
        elif crossover_type == "single-point":
            offspring = self.single_point_crossover(parent1, parent2)
        elif crossover_type == "multi-point":
            offspring = self.multi_point_crossover(parent1, parent2)

        if random.random() < crossover_probability:
            return offspring
        else:
            return parent1 if random.random() < 0.5 else parent2


    def mutate(self, individual):
        """
        Apply mutation to an individual.

        Args:
            individual (list): The individual to mutate.

        Returns:
            The mutated individual.
        """
        for i in range(self.num_genes):
            if random.random() < self.mutation_rate[i]:
                individual[i] += random.uniform(-1, 1)
                individual[i] = max(min(individual[i], self.gene_range[i][1]), self.gene_range[i][0])
        return individual

    def get_new_population(self, population, fitness_scores):
        # create an empty list to store the new population
        new_population = []

        # loop through the population size
        for i in range(self.pop_size):
            # select two parents using tournament selection
            parent1, parent2 = self.tournament_selection(population, fitness_scores)

            # apply crossover to create a new child
            child = self.crossover(parent1, parent2)

            # apply mutation to the child
            child = self.mutate(child)

            # add the child to the new population
            new_population.append(child)

        # return the new population
        return new_population


    def run(self, num_generations, initial_values=None):
        # If initial values are provided, initialize the population with those values
        if initial_values:
            self.population = self.initialize_population(initial_values)
        else:
            self.population = self.initialize_population()
            
        # Loop through the specified number of generations
        for i in range(num_generations):
            # Evaluate the fitness of each individual in the population
            fitness_scores = self.evaluate_population()
            # Get the best fitness score and individual in the population
            best_fitness_score = max(fitness_scores)
            best_individual = self.population[fitness_scores.index(best_fitness_score)]
            # Print the current generation and best fitness score
            print(f"Generation {i}: Best fitness score = {best_fitness_score:.2f}")
            # Generate a new population using tournament selection, crossover, and mutation
            self.population = self.get_new_population(self.population, fitness_scores)
        
        # Return the best individual and its fitness score
        return best_individual, best_fitness_score


