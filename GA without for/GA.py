      # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
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

        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3
        if opt.noautoanchor:
            del hyp['anchors'], meta['anchors']
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'
        if opt.bucket:
            # download evolve.csv if exists
            subprocess.run([
                'gsutil',
                'cp',
                f'gs://{opt.bucket}/evolve.csv',
                str(evolve_csv),])

        mutation_rate = np.array([meta[k][0] for k in hyp.keys()])
        lower_limit = np.array([meta[k][1] for k in hyp.keys()])
        upper_limit = lower_limit = np.array([meta[k][2] for k in hyp.keys()])
        initial_solution = list(hyp.values())
        sol_ranges = []
        for i in range(len(upper_limit)):
            sol_ranges.append((lower_limit[i],upper_limit[i]))

        num_parents = 10
        num_generations = 100
        mutation_rate = 0.1
        sol_per_pop = 20
        num_genes = 10
        sol_range = [(0.0, 1.0)] * num_genes

        # Define initial solutions
        solutions = []
        for i in range(sol_per_pop):
            sol = []
            for j in range(num_genes):
                sol.append(random.uniform(sol_range[j][0], sol_range[j][1]))
            solutions.append(sol)

        # Define genetic algorithm
        best_solution = None
        for generation in range(num_generations):
            # Calculate fitness of each solution
            for solution in solutions:
                # Update hyperparameters
                keys = list(hyp.keys())
                for i in range(len(solution)):
                    hyp[keys[i]] = solution[i]
                # Train and get fitness
                results = train(hyp.copy(), opt, device, callbacks)
                callbacks = Callbacks()
                # Write mutation results
                keys = ('metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', 'val/box_loss',
                        'val/obj_loss', 'val/cls_loss')
                print_mutation(keys, results, hyp.copy(), save_dir, opt.bucket)
                fitness = 0.9*results[2]+0.1*results[3] # (0.9 * 'metrics/mAP_0.5') + (0.1 * 'metrics/mAP_0.5:0.95')
                # Save fitness in solution object
                solution.append(fitness)

            # Sort solutions by fitness
            solutions.sort(key=lambda x: x[-1], reverse=True)

            # Print the best solution in the current generation
            print("Generation:", generation + 1, "Best Fitness:", solutions[0][-1], "Best Solution:", solutions[0][:-1])

            # Update the best solution
            if best_solution is None or solutions[0][-1] > best_solution[-1]:
                best_solution = solutions[0]

            # Create a list to store new generation children
            children = []

            # Create new generation
            for i in range(sol_per_pop):
                # Select parents
                parents = random.sample(solutions[:num_parents], 2)

                # Create child from parents
                child = []
                for j in range(num_genes):
                    if random.random() < 0.5:
                        child.append(parents[0][j])
                    else:
                        child.append(parents[1][j])

                # Mutate child
                for j in range(num_genes):
                    if random.random() < mutation_rate:
                        child[j] = random.uniform(sol_range[j][0], sol_range[j][1])

                # Add child to new generation
                children.append(child)

            # Set current solutions to new generation
            solutions = children

        # Print the best solution found by the genetic algorithm
        print("Best Solution Found:", best_solution[:-1])
  
        # Plot results
        plot_evolve(evolve_csv)
        LOGGER.info(f'Hyperparameter evolution finished {opt.evolve} generations\n'
                    f"Results saved to {colorstr('bold', save_dir)}\n"
                    f'Usage example: $ python train.py --hyp {evolve_yaml}')