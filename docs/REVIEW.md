# Repository review and first improvement round

Reviewed 30 August 2026, against original commit `9bc9bdde3106afdb0cbffad86496d7d5c135efcd`.

## Assessment and scope

This is a collection of educational GA experiments, not yet a validated optimization library or research artifact. Its core intent—exploring selection, crossover, mutation, and bounded hyperparameter search—is worth preserving. Correctness and reproducibility take priority over adding more advanced operators or concurrency.

The inspection covered all 12 tracked Python source files, the README, YAML configuration, license headers, and Git tree/history. The array-input directory is a Git submodule entry without a mapping, so its contents were unavailable. The two tracked Python 3.10 bytecode files were inventoried but not treated as maintainable source. The repository originally had no tests, CI, packaging metadata, benchmark harness, raw results, or environment specification.

Static review was supplemented by isolated reproductions of selected failures. Historical examples with import-time execution were inspected through their function/class definitions instead of launching long experiments. No YOLO training, GPU workload, or remote data transfer was run.

## Priorities by impact

| Priority | Weakness and consequence | First-round status | Next action |
| --- | --- | --- | --- |
| P1: result integrity | Fitness/individual misalignment, skipped failures, parent aliasing, malformed initialization, unevaluated final population | Fixed in reusable bounded-real GA | Preserve regressions before extending operators |
| P1: basic execution | Default scalar mutation crashes; point crossovers fail; configuration example is machine-specific and collapses bounds | Fixed in supported path | Keep historical scripts explicitly marked unsupported |
| P1: other variants | Mutation races, incorrect bounds, undefined objective domains, invalid roulette weights | Documented, unchanged | Repair one variant at a time behind tests, or explicitly retire it with maintainer approval |
| P2: repeatability | Global RNGs, silent behavior assumptions, missing fitness budgets and experiment metadata | Local seed, history, counters, benchmark metadata added | Add objective/trainer seeds, checkpoint state, environment locks |
| P2: evaluation evidence | No baselines, repetitions, uncertainty, held-out tasks, or recorded results | Small equal-budget, multi-seed harness added | Expand methodology before making performance claims |
| P2: usability | README command points to nonexistent root file; inconsistent representation and operators; no installable entry point | README, portable CLI, minimal package, tests and CI added | Move supported code into a namespaced package with compatibility imports |
| P2: scaling | Excessive reevaluation in older variants; thread overhead; fixed absolute mutation steps across very different scales | Serial default and single score per candidate per population in supported path | Profile realistic objectives; compare normalized mutation and batch evaluation |
| P3: repository hygiene | Empty final source, broken Git submodule entry, tracked bytecode, ambiguous third-party provenance | Inventoried; originals retained | Recover intended sources/mapping, remove generated artifacts in a separate approved cleanup |

P1 means results may be wrong or a documented/basic execution path fails; P2 is needed for dependable experimentation; P3 improves maintainability without directly fixing a computed result.

## Concrete findings in the original source

Line references below refer to the original commit, not the repaired files.

### Reusable bounded-real implementation

Source: [parameters with range - advance/GA.py](https://github.com/ShAmoNiA/Genetic_algorithm/blob/9bc9bdde3106afdb0cbffad86496d7d5c135efcd/parameters%20with%20range%20-%20advance/GA.py).

1. **Scores are collected in completion order and then used as population indices** (`evaluate_population`, lines 87–99; selection, lines 115–118). A delayed objective produced scores `[3, 4, 2, 1, 0]` for individuals with values `[0, 1, 2, 3, 4]`. This can select the wrong parents and report a chromosome that does not have the returned score. Failed evaluations are printed and omitted, further shifting or shortening that list.
2. **Initialization constructs nested genes** (lines 30–32, 45–53). With a population matrix, it uses `initial_values[j]` for every row instead of indexing the candidate and gene. Two supplied rows `[[0, 1], [1, 0]]` become two copies of the entire nested matrix. `run()` also discards constructor initialization when no replacement is passed.
3. **The default mutation setting crashes** (`mutate`): `mutation_rate=0.1` is indexed as `mutation_rate[i]`. Ranges, dimensions, probabilities, tournament size, and score finiteness are not validated consistently.
4. **Skipped crossover returns a parent object directly**. In-place mutation then changes a candidate whose score is already cached, and multiple offspring can share state. The configured `crossover_rate` is never used by the default generation path, which takes a separate default probability of 0.5.
5. **Point crossover methods omit `self`**. Calling the single-point option through an instance raises `TypeError`. Multi-point crossover excludes the last valid cut boundary, duplicates earlier sections for some cut counts, and always appends the second parent's tail regardless of segment parity.
6. **Run results omit the final offspring population and forget previous bests**. Zero generations raises `UnboundLocalError`; returned best references can be changed by later in-place mutation. These defects undermine both correctness and comparisons by generation count.

The repair preserves tournament selection, maximization, real-valued chromosomes, uniform crossover as the default, and the original additive `[-1, 1]` mutation with clipping. It intentionally does not add survivor elitism or claim the mutation distribution is ideal.

### Configuration runner and incomplete paths

- [Advanced runner](https://github.com/ShAmoNiA/Genetic_algorithm/blob/9bc9bdde3106afdb0cbffad86496d7d5c135efcd/parameters%20with%20range%20-%20advance/run.py), lines 43–57: the YAML path is hard-coded to `F:\project\...`; `upper_limit = lower_limit = ...` overwrites both arrays with upper bounds, collapsing the search space. `num_genes=3` conflicts with full metadata arrays, and the supplied initialization has the matrix bug above. The revised demo aligns every YAML key with its bounds and mutation probability and clearly labels its sum objective as a toy.
- [GA _ final](https://github.com/ShAmoNiA/Genetic_algorithm/tree/9bc9bdde3106afdb0cbffad86496d7d5c135efcd/GA%20_%20final): `genetic_algorithm.py` is empty while `run.py` imports a function from it. The tracked bytecode is not a substitute for the missing source. The runner's elitism/adaptation/co-evolution parameters therefore do not establish implemented algorithm features.
- [GA without for/GA.py](https://github.com/ShAmoNiA/Genetic_algorithm/blob/9bc9bdde3106afdb0cbffad86496d7d5c135efcd/GA%20without%20for/GA.py): compilation fails with `IndentationError` at line 2. It depends on undeclared training context (`opt`, `train`, callbacks, device, logging helpers), repeats the collapsed-bound assignment at line 53, then overrides intended metadata with ten generic `[0,1]` genes. It is an integration fragment, not an executable vectorized GA.
- `advance-array of int as input` is a Git tree entry of mode `160000` pointing at `2dcb5c9...`, but `.gitmodules` is absent. `git submodule status` fails. The intended nested repository must be identified before changing this entry.

### Historical algorithm variants

- [parameters with range - simple/GA.py](https://github.com/ShAmoNiA/Genetic_algorithm/blob/9bc9bdde3106afdb0cbffad86496d7d5c135efcd/parameters%20with%20range%20-%20simple/GA.py), lines 49–64: all mutations use `x_range`, even for `y` and `z`. A forced mutation of `z=100` produced `z=10`. The proportionate step also makes zero-valued genes immobile without another variation source. Generation logging inspects a selected survivor before considering new offspring, so it can lag the true generation best.
- [yolo ga/v1.py](https://github.com/ShAmoNiA/Genetic_algorithm/blob/9bc9bdde3106afdb0cbffad86496d7d5c135efcd/yolo%20ga/v1.py), lines 60–74: a population of identical individuals leaves no distinct second-best candidate and raises `ValueError`. Mutation futures are submitted without awaiting completion before selection/evaluation; genes can change while being scored, and worker exceptions are never retrieved. Population-size arithmetic is wrong for odd sizes. The initial best score of zero also distorts early stopping for negative objectives.
- In that same variant, each parent selection recomputes fitness for the whole population, giving roughly quadratic fitness-evaluation growth in population size per generation. Crossover nests a full gene loop inside another, making its advertised per-gene probability difficult to interpret and adding quadratic gene work. Threads mutate cheap values rather than evaluate the expensive objective.
- [yolo ga/v2.py](https://github.com/ShAmoNiA/Genetic_algorithm/blob/9bc9bdde3106afdb0cbffad86496d7d5c135efcd/yolo%20ga/v2.py), lines 48–74: `N//2` parent pairs each produce only one child, so the population drops to half the requested size after the first generation while score buffers and loop bounds retain the original size. This changes the experiment's effective population/budget. Raw roulette weights fail for zero totals or negative fitness; the all-zero case was reproduced. Thread exceptions can leave placeholder zeros in the score array. The no-crossover branch draws fresh random genes instead of inheriting an unchanged parent.
- [advance-binary input/GA.py](https://github.com/ShAmoNiA/Genetic_algorithm/blob/9bc9bdde3106afdb0cbffad86496d7d5c135efcd/advance-binary%20input/GA.py), lines 11–13: an all-zero chromosome decodes to zero, outside the objective's division/log domain. [advance-int input/GA.py](https://github.com/ShAmoNiA/Genetic_algorithm/blob/9bc9bdde3106afdb0cbffad86496d7d5c135efcd/advance-int%20input/GA.py), lines 28–33: bit crossover can produce zero or exceed the initial integer interval; for parents 1 and 1024 with cut 1, children are 1025 and 0. Initialization bounds alone do not enforce valid offspring.
- [simple/GA.py](https://github.com/ShAmoNiA/Genetic_algorithm/blob/9bc9bdde3106afdb0cbffad86496d7d5c135efcd/simple/GA.py), lines 24–25 and 40: tied scores can select the same candidate twice through `list.index`; reporting uses fitness scores from before replacing an individual, which can misidentify the best candidate. Replacing one candidate per iteration also makes its “generation” incomparable to full-population replacement.
- [super advance-array of int as input/GA.py](https://github.com/ShAmoNiA/Genetic_algorithm/blob/9bc9bdde3106afdb0cbffad86496d7d5c135efcd/super%20advance-array%20of%20int%20as%20input/GA.py): a sum of mostly positive per-gene scores rewards longer chromosomes. This confounds solution quality with length unless length itself is part of the objective. The two-point operator preserves parent lengths; it does not meaningfully explore new lengths. Mutation resets to the base rate halfway through despite the comment saying “maximum.”

Several historical examples optimize `sin(10*pi*x)/x + log(x)` for positive integers. Mathematically the sine term is zero on integers, leaving essentially a monotonic log objective; floating-point residuals do not make this a convincing multimodal benchmark. Many examples run experiments at import time, making reuse and automated testing awkward.

## Performance, packaging, and code quality

The supported implementation now evaluates each candidate once per population, keeps aligned scores for tournament selection, and defaults to serial execution. Ordered executor mapping follows the [Python concurrent.futures contract](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Executor.map). Threads remain available for suitable objectives, but recreating their pool each population has overhead. There is no claimed speedup for Python CPU-bound work. Profile realistic objective costs before adding process pools, persistent executors, vectorization, or caches; caching noisy fitness changes experimental meaning.

The fixed absolute mutation step is poorly matched to ranges from `0.001` to `45`, and clipping concentrates probability at boundaries. Compare a configurable step proportional to range width and log-domain representations for positive scale parameters. Treat zero-width bounds as fixed genes. Add integer/categorical specifications before presenting the real-valued core as general hyperparameter optimization.

Minimal packaging installs the supported module as `GA`, preserving its existing import. This avoids moving historical sources now, but the generic module name can collide with other projects; a proper namespace with a compatibility shim is a later improvement. PyYAML is an optional example dependency, and NumPy is no longer needed for the core. Exact environment locks, a broader Python matrix, release procedures, type annotations, and a stable public result schema remain future work.

The root MIT license is retained, and the YAML file separately identifies Ultralytics and GPL-3.0. This is a provenance discrepancy to resolve before redistribution, not a conclusion about legal compatibility. No license header or historical artifact was removed or relicensed. `.gitignore` prevents new generated files; existing tracked bytecode remains tracked.

## Research and benchmark methodology

The original repository does not contain evidence that any variant outperforms another optimizer. A printed best score on a single easy objective is a demo. Counts of “generations” are especially misleading because some variants replace one or two candidates, others half or all of the population, and some reevaluate scores repeatedly.

The first-round harness provides a limited starting point: Sphere and Rastrigin, five dimensions, bounds `[-5.12,5.12]`, seeds 0–29, population 30, 40 replacement generations, and **1,230 actual objective calls per method per seed**. Random search receives the same budget and shares the initial population-sized sample prefix. Raw records include seed, objective, dimensions, score, optimality gap, evaluation count, and elapsed time. Metadata captures configuration, environment, Git state, and source hashes. No settings were tuned using these results.

Observed on the local Python 3.13.5 / Windows run (smaller gap is better):

| Problem | GA median gap | Random-search median gap |
| --- | ---: | ---: |
| Sphere | 0.002435 | 2.390928 |
| Rastrigin | 0.716026 | 24.162369 |

The GA used more wall-clock time despite these better objective values. These observations apply only to the specified two-problem experiment and do not establish statistical significance, general superiority, or improvement over the original buggy implementation. Raw timing measurements belong to the recorded machine and run.

Before making research claims:

1. **Define the problem contract.** Record objective direction, bounds, dimensionality, gene types, constraints, feasible-solution handling, known optimum or target, and the stopping budget. Use evaluation counts and separately report elapsed time and compute resources.
2. **Use relevant baselines and tasks.** Start with random search, a simple local optimizer, and established evolutionary methods such as differential evolution or CMA-ES where appropriate. Use unimodal, multimodal, ill-conditioned, rotated, constrained, and noisy tasks across dimensions. The [COCO project](https://numbbo.github.io/coco-doc/) provides a structured benchmarking route. [Bergstra and Bengio's random-search study](https://www.jmlr.org/papers/v13/bergstra12a.html) motivates random search as a meaningful hyperparameter baseline, not a guarantee about this repository.
3. **Separate development from evaluation.** Choose operator settings and stopping criteria on development tasks. Reserve distinct tasks or instances for evaluation. Predeclare seeds and budgets; do not keep rerunning until a favorable seed appears.
4. **Report distributions.** Retain every run and failure. Show median and dispersion, paired per-seed differences where justified, bootstrap confidence intervals, success rates, and convergence against evaluation count. State interval assumptions and account for multiple comparisons; do not infer significance from two medians.
5. **Ablate design choices.** Compare fixed versus range-scaled mutation, crossover types, survivor elitism, tournament pressure, initialization, and serial/threaded evaluation while changing one factor at a time and holding actual evaluations constant. Include setup costs for wall-time comparisons and separate expensive-objective time from algorithm overhead.
6. **For YOLO, define an actual training protocol.** Record dataset/version/splits, architecture and pretrained weights, trainer revision, epochs or early-stopping rules, augmentation, hardware, numerical precision, and training seeds. Search only on training/validation data; use the test set for final assessment. Re-evaluate selected configurations across independent training seeds. Prevent parallel candidates from sharing mutable training state or oversubscribing one GPU.
7. **Make runs replayable.** Save machine-readable configuration, dependency versions or locks, code revision, dataset identity, objective seed, RNG state for resumptions, candidate history, and failure policy. A search RNG seed alone cannot reproduce an external stochastic trainer.

## Validation and intentional limits

The first-round suite has 21 tests covering score ordering, exception propagation, nonfinite scores, callback isolation, initialization shape/copying, scalar/vector mutation, crossover rates and segment integrity, parent/child isolation, small and odd population sizes, final-generation scoring, best-so-far retention, seeded serial/thread agreement, evaluation budgets, benchmark objective/budget sanity, and portable reproducible CLI behavior.

Locally verified on Windows with Python 3.13: clean-environment wheel build/install, optional YAML example, dependency consistency, 21 passing tests, and the 120-run exploratory benchmark (two problems, two algorithms, 30 seeds). The benchmark consumes 147,600 objective calls total. CI is configured for Python 3.10 and 3.13 on Linux and Windows; its remote outcome must be checked separately. No other historical variant is claimed fixed, tested end-to-end, or production-ready.
