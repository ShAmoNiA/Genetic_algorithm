import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("benchmark", ROOT / "benchmarks/compare_random_search.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BenchmarkTests(unittest.TestCase):
    def test_objectives_have_known_optimum_and_sign(self):
        self.assertEqual(MODULE.sphere([0, 0], None), 0)
        self.assertEqual(MODULE.rastrigin([0, 0], None), 0)
        self.assertEqual(MODULE.sphere([1, 2], None), -5)
        self.assertAlmostEqual(MODULE.rastrigin([1, 2], None), -5)

    def test_random_search_budget_and_shared_initial_samples(self):
        samples = []
        score = MODULE.random_search(lambda x, e: samples.append(x) or sum(x), 3, 20, 7)
        self.assertEqual(len(samples), 20)
        self.assertEqual(score, max(map(sum, samples)))
        ga = MODULE.MODULE.GeneticAlgorithm(lambda x, e: sum(x), 3, [(-5.12, 5.12)] * 3,
                                             pop_size=5, seed=7)
        self.assertEqual(samples[:5], ga.population)
        self.assertEqual(MODULE.random_search(lambda x, e: sum(x), 3, 20, 7), score)


if __name__ == "__main__":
    unittest.main()
