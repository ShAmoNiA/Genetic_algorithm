"""Portable configuration demo; the toy fitness does not train or assess YOLO."""

import argparse
import json
from pathlib import Path

from GA import GeneticAlgorithm

meta = {
    'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
    'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
    'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
    'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
    'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
    'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
    'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
    'box': (1, 0.02, 0.2),  # box loss gain
    'cls': (1, 0.2, 4.0),  # cls loss gain
    'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
    'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
    'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
    'iou_t': (0, 0.1, 0.7),  # IoU training threshold
    'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
    'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
    'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
    ##---------
    'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
    'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
    'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
    'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
    'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
    'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
    'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
    'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
    'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
    'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
    'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
    'mixup': (1, 0.0, 1.0),  # image mixup (probability)
    'copy_paste': (1, 0.0, 1.0)}  # segment copy-paste (probability)


def my_fitness_function(values, extra_prop):
    """A toy bounded sum, useful for checking wiring only (larger is better)."""
    return sum(values)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--population-size", type=int, default=50)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--hyp", type=Path,
                        default=Path(__file__).with_name("hyp.scratch-low.yaml"))
    args = parser.parse_args(argv)
    # Only this configuration example needs PyYAML; importing GA does not.
    import yaml

    with args.hyp.open(encoding="utf-8") as stream:
        hyp = yaml.safe_load(stream)
    if not isinstance(hyp, dict) or not hyp or set(hyp) - set(meta):
        raise ValueError("hyp must be a nonempty mapping of known hyperparameter names")
    names = list(hyp)
    bounds = [(meta[name][1], meta[name][2]) for name in names]
    # Metadata entries are mutation scales, not probabilities. In this toy
    # example a base probability of 0.1 is multiplied by each scale.
    rates = [0.1 * meta[name][0] for name in names]
    ga = GeneticAlgorithm(
        my_fitness_function, len(names), bounds, pop_size=args.population_size,
        mutation_rate=rates, seed=args.seed, workers=args.workers,
    )
    initial_population = [row[:] for row in ga.population]
    initial_population[0] = [hyp[name] for name in names]
    best, score = ga.run(args.generations, initial_population, verbose=False)
    print(json.dumps({
        "objective": "toy_sum_not_yolo_training", "seed": args.seed,
        "generations": args.generations, "population_size": args.population_size,
        "workers": args.workers, "evaluations": ga.evaluations,
        "best_fitness": score, "best_individual": dict(zip(names, best)),
    }, indent=2))


if __name__ == "__main__":
    main()
