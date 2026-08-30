"""Regression tests for the supported bounded-real implementation."""

import copy
import importlib.util
import math
from pathlib import Path
import random
import threading
import time
import unittest
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "parameters with range - advance" / "GA.py"
SPEC = importlib.util.spec_from_file_location("bounded_ga", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
GeneticAlgorithm = MODULE.GeneticAlgorithm


def objective(values, extra):
    return sum(values)


class GeneticAlgorithmTests(unittest.TestCase):
    def make_ga(self, **changes):
        options = dict(fitness_func=objective, num_genes=3, gene_range=[(-10, 10)] * 3,
                       pop_size=5, seed=42)
        options.update(changes)
        return GeneticAlgorithm(**options)

    def test_initial_population_is_preserved_and_copied(self):
        rows = [[i, i + 1, i + 2] for i in range(5)]
        ga = self.make_ga(initial_values=rows)
        rows[0][0] = 99
        best, score = ga.run(0, verbose=False)
        self.assertEqual(ga.population[0], [0, 1, 2])
        self.assertEqual((best, score), ([4, 5, 6], 15))
        best[0] = 99
        self.assertEqual(ga.population[-1], [4, 5, 6])

    def test_run_accepts_replacement_population(self):
        ga = self.make_ga()
        self.assertEqual(ga.run(0, [[1, 2, 3]] * 5, verbose=False), ([1, 2, 3], 6))
        self.assertEqual(len({id(row) for row in ga.population}), 5)

    def test_parallel_fitness_keeps_population_order(self):
        second_done = threading.Event()

        def delayed(values, extra):
            if values[0] == 0:
                if not second_done.wait(timeout=2):
                    raise TimeoutError("second evaluation never started")
                time.sleep(0.02)
            else:
                second_done.set()
            return values[0]

        ga = self.make_ga(fitness_func=delayed, workers=2,
                          initial_values=[[i, 0, 0] for i in range(5)])
        scores = ga.evaluate_population()
        self.assertEqual(scores, [0, 1, 2, 3, 4])
        self.assertEqual(ga.tournament_selection(ga.population, scores), [[4, 0, 0]] * 2)

    def test_failed_fitness_aborts_with_original_cause(self):
        def broken(values, extra):
            raise LookupError("objective failed")

        for workers in [1, 2]:
            with self.subTest(workers=workers):
                ga = self.make_ga(fitness_func=broken, workers=workers)
                with self.assertRaisesRegex(RuntimeError, "individual 0") as caught:
                    ga.run(1, verbose=False)
                self.assertIsInstance(caught.exception.__cause__, LookupError)
                self.assertEqual(ga.history, [])

    def test_invalid_fitness_is_rejected(self):
        for workers in [1, 2]:
            for score in [math.nan, math.inf, -math.inf, "1", [1], True]:
                with self.subTest(workers=workers, score=score):
                    ga = self.make_ga(fitness_func=lambda x, e: score, workers=workers)
                    with self.assertRaises(RuntimeError) as caught:
                        ga.evaluate_population()
                    self.assertIsInstance(caught.exception.__cause__, ValueError)

    def test_objective_cannot_mutate_population(self):
        extra = {"offset": 5}

        def mutating(values, context):
            values[0] = 100
            return context["offset"]

        ga = self.make_ga(fitness_func=mutating, extra_prop=extra)
        before = copy.deepcopy(ga.population)
        self.assertEqual(ga.evaluate_population(), [5] * 5)
        self.assertEqual(ga.population, before)
        self.assertEqual(ga.calculate_fitness(before[0]), 5)

    def test_scalar_and_vector_mutation_and_fixed_bounds(self):
        for rate in [0.1, [0.1, 0.1, 0.1], 0, 1]:
            with self.subTest(rate=rate):
                ga = self.make_ga(mutation_rate=rate, gene_range=[(-1, 1), (0, 0), (2, 3)])
                for _ in range(20):
                    child = ga.mutate([0, 0, 2.5])
                    self.assertTrue(-1 <= child[0] <= 1)
                    self.assertEqual(child[1], 0)
                    self.assertTrue(2 <= child[2] <= 3)
        ga = self.make_ga(mutation_rate=[0, 1, 0])
        self.assertEqual(ga.mutate([1, 2, 3])[::2], [1, 3])

    def test_no_crossover_does_not_modify_parents(self):
        ga = self.make_ga(crossover_rate=0, mutation_rate=1)
        parents = [[1, 2, 3], [4, 5, 6]]
        before = copy.deepcopy(parents)
        with patch.object(ga, "uniform_crossover", side_effect=AssertionError("should be skipped")):
            child = ga.crossover(*parents)
        self.assertTrue(all(child is not parent for parent in parents))
        ga.mutate(child)
        self.assertEqual(parents, before)

    def test_children_are_independent_and_population_size_is_exact(self):
        for size in [1, 2, 5, 6]:
            ga = self.make_ga(pop_size=size, crossover_rate=0, mutation_rate=1)
            before = copy.deepcopy(ga.population)
            children = ga.get_new_population(ga.population, ga.evaluate_population())
            self.assertEqual(len(children), size)
            self.assertEqual(len({id(child) for child in children}), size)
            self.assertEqual(ga.population, before)

    def test_all_crossover_types_preserve_shape_and_bounds(self):
        for size in [1, 2, 3, 6]:
            ga = self.make_ga(num_genes=size, gene_range=[(0, 1)] * size)
            for kind in ["uniform", "arithmetic", "single-point", "multi-point"]:
                with self.subTest(size=size, kind=kind):
                    child = ga.crossover([0] * size, [1] * size, 1, kind)
                    self.assertEqual(len(child), size)
                    self.assertTrue(all(0 <= value <= 1 for value in child))

    def test_multi_point_alternates_segments_without_duplicate_genes(self):
        ga = self.make_ga(num_genes=6, gene_range=[(0, 20)] * 6)
        with patch.object(ga.rng, "sample", return_value=[1, 3, 5]):
            result = ga.multi_point_crossover(list(range(6)), list(range(10, 16)), 3)
        self.assertEqual(result, [0, 11, 12, 3, 4, 15])
        self.assertEqual(ga.multi_point_crossover([0] * 6, [1] * 6, 0), [0] * 6)
        with self.assertRaises(ValueError):
            ga.multi_point_crossover([0] * 6, [1] * 6, 6)

    def test_final_offspring_are_evaluated(self):
        ga = self.make_ga(initial_values=[[0, 0, 0]] * 5)
        with patch.object(ga, "get_new_population", return_value=[[1, 2, 3]] * 5):
            self.assertEqual(ga.run(1, verbose=False), ([1, 2, 3], 6))
        self.assertEqual(ga.evaluations, 10)

    def test_best_ever_survives_a_worse_generation(self):
        ga = self.make_ga(initial_values=[[1, 2, 3]] * 5)
        with patch.object(ga, "get_new_population", return_value=[[-1, -2, -3]] * 5):
            best, score = ga.run(1, verbose=False)
        self.assertEqual((best, score), ([1, 2, 3], 6))
        self.assertEqual([row["best_fitness"] for row in ga.history], [6, 6])
        self.assertEqual(ga.history[-1]["generation_best"], -6)

    def test_zero_and_negative_fitness_work(self):
        for score in [0, -5]:
            ga = self.make_ga(fitness_func=lambda x, e: score)
            self.assertEqual(ga.run(2, verbose=False)[1], score)

    def test_evaluation_budget_and_history(self):
        calls = []
        ga = self.make_ga(fitness_func=lambda x, e: calls.append(x) or sum(x))
        ga.run(3, verbose=False)
        self.assertEqual(len(calls), 20)
        self.assertEqual(ga.evaluations, 20)
        self.assertEqual([row["evaluations"] for row in ga.history], [5, 10, 15, 20])
        scores = [row["best_fitness"] for row in ga.history]
        self.assertEqual(scores, sorted(scores))
        previous = copy.deepcopy(ga.population)
        ga.run(0, verbose=False)
        self.assertEqual(ga.population, previous)
        self.assertEqual(ga.evaluations, 5)
        self.assertEqual(len(ga.history), 1)

    def test_seed_reproduces_across_worker_counts_without_global_rng_changes(self):
        state = random.getstate()
        first = self.make_ga(workers=1)
        result = first.run(5, verbose=False)
        self.assertEqual(random.getstate(), state)
        for _ in range(30):
            random.random()
        second = self.make_ga(workers=3)
        self.assertEqual(second.run(5, verbose=False), result)
        self.assertEqual(second.population, first.population)
        self.assertEqual(second.history, first.history)
        random.setstate(state)

    def test_invalid_configuration_is_rejected_early(self):
        cases = [dict(num_genes=0), dict(pop_size=0), dict(pop_size=True),
                 dict(workers=0), dict(workers=1.5), dict(tournament_size=6),
                 dict(tournament_size=0), dict(gene_range=[(0, 1)]),
                 dict(gene_range=[(2, 1)] * 3), dict(gene_range=[(0, math.inf)] * 3),
                 dict(gene_range=[(0,)] * 3), dict(mutation_rate=[0.1]),
                 dict(mutation_rate=-1), dict(mutation_rate=math.nan),
                 dict(crossover_rate=2), dict(fitness_func=None),
                 dict(initial_values=[]), dict(initial_values=[1, 2, 3, 4, 5]),
                 dict(initial_values=[[1]] * 5), dict(initial_values=[[99, 0, 0]] * 5),
                 dict(initial_values=[[math.nan, 0, 0]] * 5)]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.make_ga(**case)
        for generations in [-1, 1.5, True]:
            with self.subTest(generations=generations), self.assertRaises(ValueError):
                self.make_ga().run(generations, verbose=False)


if __name__ == "__main__":
    unittest.main()
