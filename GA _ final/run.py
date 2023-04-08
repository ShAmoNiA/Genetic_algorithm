from genetic_algorithm import genetic_algorithm
import math

import numpy as np

# Define the parameters
population_size = 100
gene_size = 10
num_generations = 100
elitism_rate = 0.1
adaptive_rate = 0.1
coevolution_rate = 0.2
selection_size = 10
mutation_rate = np.full(gene_size, 0.1)
crossover_rate = np.full(gene_size, 0.8)

# Call the genetic_algorithm function
genetic_algorithm(population_size, gene_size, num_generations, elitism_rate, adaptive_rate, coevolution_rate,
                  selection_size, mutation_rate, crossover_rate)
