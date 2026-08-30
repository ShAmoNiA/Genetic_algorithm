"""A bounded, real-valued genetic algorithm that maximizes scalar fitness."""

import concurrent.futures
import math
from numbers import Integral, Real
import random


def _positive_integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def _probability(value, name):
    value = _finite_number(value, name)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


class GeneticAlgorithm:
    """Tournament selection, crossover, and bounded additive mutation.

    ``fitness_func(individual, extra_prop)`` must return a finite real score;
    larger is better. ``initial_values`` is a population of shape
    ``(pop_size, num_genes)``. All genes are real valued, including integer inputs.

    A private RNG isolates the search from global random state. A seed reproduces
    a fresh instance's search for a deterministic objective on the same Python
    version. Threads are opt-in; the objective and extra_prop must be thread-safe.
    """

    def __init__(self, fitness_func, num_genes, gene_range, pop_size=50,
                 mutation_rate=0.1, crossover_rate=0.9, extra_prop=None,
                 initial_values=None, *, seed=None, workers=1,
                 tournament_size=None):
        if not callable(fitness_func):
            raise ValueError("fitness_func must be callable")
        self.num_genes = _positive_integer(num_genes, "num_genes")
        self.pop_size = _positive_integer(pop_size, "pop_size")
        self.workers = _positive_integer(workers, "workers")
        self.tournament_size = _positive_integer(
            min(5, self.pop_size) if tournament_size is None else tournament_size,
            "tournament_size",
        )
        if self.tournament_size > self.pop_size:
            raise ValueError("tournament_size cannot exceed pop_size")
        self.gene_range = self._validate_ranges(gene_range)
        if isinstance(mutation_rate, Real):
            mutation_rate = [mutation_rate] * self.num_genes
        try:
            rates = list(mutation_rate)
        except TypeError as exc:
            raise ValueError("mutation_rate must be a probability or a sequence") from exc
        if len(rates) != self.num_genes:
            raise ValueError("mutation_rate must have one probability per gene")
        self.mutation_rate = tuple(_probability(rate, "mutation_rate") for rate in rates)
        self.crossover_rate = _probability(crossover_rate, "crossover_rate")
        self.fitness_func = fitness_func
        self.extra_prop = extra_prop
        self.seed = seed
        self.rng = random.Random(seed)
        self.history = []
        self.evaluations = 0
        self.population = self.initialize_population(initial_values)

    def _validate_ranges(self, gene_range):
        try:
            ranges = list(gene_range)
        except TypeError as exc:
            raise ValueError("gene_range must contain one pair per gene") from exc
        if len(ranges) != self.num_genes:
            raise ValueError("gene_range must contain one pair per gene")
        validated = []
        for bounds in ranges:
            try:
                lower, upper = bounds
            except (TypeError, ValueError) as exc:
                raise ValueError("each gene range must be a (lower, upper) pair") from exc
            lower = _finite_number(lower, "lower bound")
            upper = _finite_number(upper, "upper bound")
            if lower > upper or not math.isfinite(upper - lower):
                raise ValueError("gene bounds must be ordered with a finite width")
            validated.append((lower, upper))
        return tuple(validated)

    def _individual(self, individual):
        try:
            values = list(individual)
        except TypeError as exc:
            raise ValueError("an individual must be a sequence of real numbers") from exc
        if len(values) != self.num_genes:
            raise ValueError("each individual must have num_genes values")
        values = [_finite_number(value, "gene") for value in values]
        if any(not lower <= value <= upper
               for value, (lower, upper) in zip(values, self.gene_range)):
            raise ValueError("initial values and parents must lie within gene_range")
        return values

    def initialize_population(self, initial_values=None):
        """Return a validated independent copy, or a uniformly sampled population."""
        if initial_values is not None:
            try:
                rows = list(initial_values)
            except TypeError as exc:
                raise ValueError("initial_values must be a population matrix") from exc
            if len(rows) != self.pop_size:
                raise ValueError("initial_values must contain pop_size individuals")
            return [self._individual(row) for row in rows]
        return [[self.rng.uniform(lower, upper) for lower, upper in self.gene_range]
                for _ in range(self.pop_size)]

    def evaluate_individual(self, individual):
        """Evaluate a copy so callback mutation cannot corrupt the population."""
        score = self.fitness_func(list(individual), self.extra_prop)
        return _finite_number(score, "fitness score")

    def calculate_fitness(self, individual):
        """Compatibility alias for evaluate_individual."""
        return self.evaluate_individual(individual)

    def _evaluate_indexed(self, item):
        index, individual = item
        try:
            return self.evaluate_individual(individual)
        except Exception as exc:
            raise RuntimeError(f"Fitness evaluation failed for individual {index}") from exc

    def evaluate_population(self):
        """Return scores in population order; fail on any exception or nonfinite score.

        No cross-generation cache is used: stochastic objectives must be evaluated
        again. A completed call makes exactly one evaluation per individual.
        """
        indexed = enumerate(self.population)
        if self.workers == 1:
            scores = list(map(self._evaluate_indexed, indexed))
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
                scores = list(executor.map(self._evaluate_indexed, indexed))
        self.evaluations += len(scores)
        return scores

    def tournament_selection(self, population, fitness_scores, tournament_size=None):
        """Select two parents using scores aligned with population indices."""
        size = self.tournament_size if tournament_size is None else _positive_integer(
            tournament_size, "tournament_size")
        if len(population) != len(fitness_scores) or not size <= len(population):
            raise ValueError("scores must match the population and tournament must fit")
        parents = []
        for _ in range(2):
            tournament = self.rng.sample(range(len(population)), k=size)
            winner = max(tournament, key=fitness_scores.__getitem__)
            parents.append(population[winner])
        return parents

    def uniform_crossover(self, parent1, parent2):
        first, second = self._individual(parent1), self._individual(parent2)
        return [a if self.rng.random() < 0.5 else b for a, b in zip(first, second)]

    def arithmetic_crossover(self, parent1, parent2):
        first, second = self._individual(parent1), self._individual(parent2)
        alpha = self.rng.random()
        return [max(lower, min(upper, alpha * a + (1 - alpha) * b))
                for a, b, (lower, upper) in zip(first, second, self.gene_range)]

    def single_point_crossover(self, parent1, parent2):
        first, second = self._individual(parent1), self._individual(parent2)
        if self.num_genes == 1:
            return first
        point = self.rng.randint(1, self.num_genes - 1)
        return first[:point] + second[point:]

    def multi_point_crossover(self, parent1, parent2, num_points=2):
        first, second = self._individual(parent1), self._individual(parent2)
        points = _positive_integer(num_points, "num_points", minimum=0)
        if points >= self.num_genes:
            raise ValueError("num_points must be less than num_genes")
        cuts = sorted(self.rng.sample(range(1, self.num_genes), points))
        child = []
        start = 0
        for segment, stop in enumerate(cuts + [self.num_genes]):
            parent = first if segment % 2 == 0 else second
            child.extend(parent[start:stop])
            start = stop
        return child

    def crossover(self, parent1, parent2, crossover_probability=None,
                  crossover_type="uniform"):
        """Return a new child even when crossover is skipped; honor crossover_rate."""
        first, second = self._individual(parent1), self._individual(parent2)
        probability = self.crossover_rate if crossover_probability is None else _probability(
            crossover_probability, "crossover_probability")
        operators = {
            "uniform": self.uniform_crossover,
            "arithmetic": self.arithmetic_crossover,
            "single-point": self.single_point_crossover,
            "multi-point": self.multi_point_crossover,
        }
        if crossover_type not in operators:
            raise ValueError("Invalid crossover type")
        if self.rng.random() >= probability:
            return first if self.rng.random() < 0.5 else second
        if crossover_type == "multi-point":
            return self.multi_point_crossover(first, second, min(2, self.num_genes - 1))
        return operators[crossover_type](first, second)

    def mutate(self, individual):
        """Mutate a child in place using the original +/-1 step and bound clipping."""
        for index, (lower, upper) in enumerate(self.gene_range):
            if self.rng.random() < self.mutation_rate[index]:
                individual[index] = max(lower, min(upper, individual[index] + self.rng.uniform(-1, 1)))
        return individual

    def get_new_population(self, population, fitness_scores):
        """Produce exactly pop_size independent children (no survivor elitism)."""
        children = []
        for _ in range(self.pop_size):
            parents = self.tournament_selection(population, fitness_scores)
            children.append(self.mutate(self.crossover(*parents)))
        return children

    def run(self, num_generations, initial_values=None, *, verbose=True):
        """Evaluate the current population and num_generations offspring populations.

        Return a copy of the best individual observed and its score, including the
        final generation. Zero generations evaluates the current population once.
        Repeated calls continue from the current population/RNG unless an explicit
        replacement initial_values matrix is supplied. History and evaluation count
        describe this call only. Noisy fitness requires separate final validation.
        """
        generations = _positive_integer(num_generations, "num_generations", minimum=0)
        if initial_values is not None:
            self.population = self.initialize_population(initial_values)
        self.history = []
        self.evaluations = 0
        best_individual, best_score = None, -math.inf
        for generation in range(generations + 1):
            scores = self.evaluate_population()
            index = max(range(len(scores)), key=scores.__getitem__)
            if scores[index] > best_score:
                best_individual, best_score = self.population[index][:], scores[index]
            self.history.append({"generation": generation, "evaluations": self.evaluations,
                                 "generation_best": scores[index], "best_fitness": best_score})
            if verbose:
                print(f"Generation {generation}: Best fitness score = {best_score:.6g}")
            if generation < generations:
                self.population = self.get_new_population(self.population, scores)
        return best_individual[:], best_score
