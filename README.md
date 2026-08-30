# Genetic algorithm experiments

This repository explores genetic algorithms for binary, integer, and bounded real-valued optimization. The supported reusable implementation is currently **`parameters with range - advance/GA.py`**, a small real-valued **maximizer** using tournament selection, uniform crossover, and bounded additive mutation. Other variants remain available as historical experiments; several are incomplete or contain known bugs. See [the detailed review](docs/REVIEW.md).

## Install and run

Python 3.10 or newer is required. From the repository root:

```console
python -m pip install .
python -m unittest discover -s tests -v
```

The core module uses only the Python standard library. For the optional YAML configuration example:

```console
python -m pip install ".[examples]"
python "parameters with range - advance/run.py" --seed 42 --generations 50
```

The example reads its YAML file relative to the script, seeds one candidate with those values, and searches all configured genes. It prints JSON including the seed, configuration, evaluation count, score, and named genes. Its fitness is deliberately a **toy sum, not YOLO training or model validation**. The metadata's mutation scales are multiplied by a base probability of 0.1; a zero scale disables mutation for that gene but does not freeze its initial population values or inheritance.

## Use your own objective

```python
from GA import GeneticAlgorithm


def fitness(values, extra_prop):
    # Negate a minimization objective because this implementation maximizes.
    return -sum(value ** 2 for value in values)


ga = GeneticAlgorithm(
    fitness_func=fitness,
    num_genes=3,
    gene_range=[(-5.0, 5.0)] * 3,
    pop_size=50,
    mutation_rate=0.1,        # scalar or one probability per gene
    crossover_rate=0.9,
    seed=42,
    workers=1,
)

best, score = ga.run(num_generations=100, verbose=False)
print(best, score, ga.evaluations)
print(ga.history[-1])
```

To supply initial candidates, pass `initial_values` as a matrix with exactly `pop_size` rows and `num_genes` finite numbers per row. Each value must be within its gene's bounds. The population is copied, and constructor-supplied candidates are preserved by `run()`. Integers are accepted as real values; integer and categorical constraints are **not** enforced.

## Run semantics and reproducibility

- `run(G)` evaluates the initial population and each of `G` offspring populations, including the final one: exactly `pop_size * (G + 1)` successful fitness calls. `run(0)` evaluates the current population once.
- The returned `(best_individual, score)` is an independent copy of the best observation over the whole run. This is best-so-far reporting, not survivor elitism: the population itself can lose its best candidate.
- Repeated `run()` calls continue from the current population and RNG state. To replay an experiment, create a fresh instance with the same seed and inputs. History and evaluation counts reset per call.
- Each instance owns its RNG. Search reproducibility assumes the same Python version, deterministic fitness, and unchanged inputs. A GA seed does not seed an external trainer, NumPy, a GPU, or the fitness function's own randomness.
- `workers=1` is the default. Opt into threads with `workers=N` only when the objective and shared `extra_prop` are thread-safe. Ordered evaluation keeps scores aligned even when tasks finish out of order. Threads need not accelerate inexpensive or Python CPU-bound objectives.
- Fitness exceptions and nonfinite/nonscalar values abort the run with the candidate index and original exception as the cause. Failed evaluations are never silently dropped. Evaluation counts describe completed populations; a failed threaded call may already have executed other callbacks.
- `ga.history` records generation number, cumulative evaluations, generation-best score, and best-so-far score. No cross-generation cache is used. With noisy objectives, the best observed score is optimistic; independently re-evaluate the selected candidate.

The legacy public methods remain available. `crossover()` honors the configured rate unless explicitly overridden, and always returns a new list. Uniform, arithmetic, single-point, and multi-point crossover are supported; short chromosomes safely reduce the default number of cut points. Explicit invalid cut counts raise `ValueError`. Mutation retains the original additive step in `[-1, 1]`, clipped to each gene's bounds. Width-aware mutation, elitism, checkpointing, and richer result objects are future work.

Behavior changes from the original: a scalar mutation rate now works; malformed configuration fails early; `run()` no longer discards constructor initialization; final offspring are scored; default evaluation is serial; skipped crossover does not alias parents; randomness is local rather than controlled by `random.seed()`.

## Exploratory benchmarks

```console
python benchmarks/compare_random_search.py --seeds 30 --output benchmark-results
```

Use a new output directory for each run. This compares the GA with uniform random search on 5-dimensional Sphere and Rastrigin over `[-5.12, 5.12]`, using seeds 0–29 and 1,230 objective evaluations per algorithm per seed by default. Both minimize through negated fitness; the known optimum is zero. The first population-sized samples match across methods for each seed.

The harness saves raw per-run CSV data and JSON metadata with configuration, Python/OS/CPU information, Git revision, dirty-state flag, and source hashes. Its console summary reports median optimality gap and elapsed time. It does **not** establish general superiority: it uses two unrotated synthetic problems, a single configuration, no confidence intervals, and no held-out task suite. Follow [the research protocol and priorities](docs/REVIEW.md#research-and-benchmark-methodology) before making research claims.

## Repository map

| Path | Status |
| --- | --- |
| `parameters with range - advance/` | Supported reusable real-valued GA and portable toy YAML example |
| `simple/`, `advance-binary input/`, `advance-int input/` | Historical binary/integer examples; not covered by the supported-core tests |
| `parameters with range - simple/` | Historical bounded-value prototype; known mutation-bound bug |
| `super advance-array of int as input/` | Historical variable-length chromosome experiment |
| `yolo ga/` | Historical threaded prototypes; toy fitness, known concurrency/selection issues |
| `GA _ final/` | Incomplete: source module is empty and runner cannot import its function |
| `GA without for/` | Indented integration fragment, not a standalone Python program |
| `advance-array of int as input` | Unresolved Git submodule entry with no `.gitmodules` mapping |
| `tests/`, `benchmarks/`, `docs/` | Regression suite, exploratory harness, and review |

CI installs the package, tests the supported implementation and example, and smoke-tests the installed module on Python 3.10/3.13 on Linux and Windows. Historical fragments are deliberately excluded; a passing supported-core test suite does not imply every old script is runnable.

The original root MIT license and all historical files are retained. The YAML configuration has its own Ultralytics GPL-3.0 attribution header; review source provenance and redistribution obligations before publishing a distribution containing third-party material.
