from GA import GeneticAlgorithm
import yaml
import numpy as np 

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
        


def my_fitness_function(x,extra_prop):
    print(extra_prop)
    return -x[0]**2 - x[1]**3 + x[2]**2

with open("F:\project\Genetic_algorithm\parameters with range - advance\hyp.scratch-low.yaml", errors='ignore') as f:
    hyp = yaml.safe_load(f)
mutation_rate = np.array([meta[k][0] for k in hyp.keys()])
lower_limit = np.array([meta[k][1] for k in hyp.keys()])
upper_limit = lower_limit = np.array([meta[k][2] for k in hyp.keys()])
initial_values = list(hyp.values())
gene_ranges = []
for i in range(len(upper_limit)):
    gene_ranges.append((lower_limit[i],upper_limit[i]))
    

print(hyp)
# ga = GeneticAlgorithm(fitness_func=my_fitness_function, num_genes=3, gene_range=gene_ranges, pop_size=50, mutation_rate=mutation_rate, extra_prop = ["a","b"])
# best_individual, best_fitness = ga.run(num_generations=500,initial_values=initial_values)
# print(f"Best individual: {best_individual}")
# print(f"Best fitness: {best_fitness}")
