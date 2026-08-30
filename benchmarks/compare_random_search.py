"""Exploratory equal-evaluation-budget checks; not evidence of general superiority."""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bounded_ga", ROOT / "parameters with range - advance/GA.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def sphere(values, extra):
    return -sum(value * value for value in values)


def rastrigin(values, extra):
    return -(10 * len(values) + sum(value * value - 10 * math.cos(2 * math.pi * value)
                                  for value in values))


def random_search(fitness, dimensions, budget, seed):
    rng = random.Random(seed)
    return max(fitness([rng.uniform(-5.12, 5.12) for _ in range(dimensions)], None)
               for _ in range(budget))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=30, help="Use seeds 0 through N-1")
    parser.add_argument("--dimensions", type=int, default=5)
    parser.add_argument("--population-size", type=int, default=30)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True, help="New directory for raw CSV and metadata")
    args = parser.parse_args(argv)
    if min(args.seeds, args.dimensions, args.population_size) < 1 or args.generations < 0:
        parser.error("seeds, dimensions and population-size must be positive; generations nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    budget = args.population_size * (args.generations + 1)
    rows = []
    for name, fitness in [("sphere", sphere), ("rastrigin", rastrigin)]:
        for seed in range(args.seeds):
            for algorithm in ["ga", "random_search"]:
                start = time.perf_counter()
                if algorithm == "ga":
                    ga = MODULE.GeneticAlgorithm(
                        fitness, args.dimensions, [(-5.12, 5.12)] * args.dimensions,
                        pop_size=args.population_size, seed=seed, workers=1,
                    )
                    _, score = ga.run(args.generations, verbose=False)
                    evaluations = ga.evaluations
                else:
                    score = random_search(fitness, args.dimensions, budget, seed)
                    evaluations = budget
                rows.append(dict(problem=name, algorithm=algorithm, seed=seed,
                                 dimensions=args.dimensions, evaluations=evaluations,
                                 best_fitness=score, optimality_gap=-score,
                                 elapsed_seconds=time.perf_counter() - start))
    with (args.output / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    metadata = dict(
        python=platform.python_version(), platform=platform.platform(),
        processor=platform.processor(), logical_cpus=os.cpu_count(),
        revision=revision, working_tree_dirty=dirty,
        source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in [ROOT / "parameters with range - advance/GA.py", Path(__file__).resolve()]},
        config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        ga_config=dict(mutation_rate=0.1, crossover_rate=0.9, workers=1,
                       tournament_size=min(5, args.population_size), survivor_elitism=False,
                       mutation_step=1.0, bounds=[-5.12, 5.12]),
        evaluation_budget_per_run=budget,
        limitations="Exploratory unrotated synthetic problems; no tuning, uncertainty intervals, or held-out tasks. "
                     "Elapsed times include search setup and are hardware-specific. Seeds share initial GA-sized samples.",
    )
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for problem in ["sphere", "rastrigin"]:
        for algorithm in ["ga", "random_search"]:
            subset = [row for row in rows if row["problem"] == problem and row["algorithm"] == algorithm]
            print(f"{problem:10} {algorithm:13} median gap={statistics.median(row['optimality_gap'] for row in subset):.6g} "
                  f"median seconds={statistics.median(row['elapsed_seconds'] for row in subset):.6g}")


if __name__ == "__main__":
    main()
