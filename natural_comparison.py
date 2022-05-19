import argparse
import os
import time
import json
import random
import numpy as np
import datasets
import models
import yaml
import torch
from torch.utils.data import DataLoader
from time import strftime
from copy import deepcopy
from continual import MetaModel, EWC, GEM, AGEM, SGEM, NGEM, Independent
from utils import color_codes, time_to_string


"""
> Arguments
"""


def parse_inputs():
    parser = argparse.ArgumentParser(
        description='Train models with incremental learning approaches and '
                    'test them to obtain timeseries metrics of simple'
                    'overlap concepts.'
    )

    # Mode selector
    parser.add_argument(
        '-i', '--input-config',
        dest='config', default='/data/IncrementalLearning/activity_dual.yml',
        help='Path to the file with the configuration for the experiment.'
    )
    options = vars(parser.parse_args())

    return options


"""
> Data functions
"""


def load_datasets(experiment_config):
    d_tr, d_te = torch.load(experiment_config['path'])
    return d_tr, d_te


"""
> Network functions
"""


def train(
    config, seed, net, training, validation, model_name, epochs, patience, verbose=0
):
    """

    :param config:
    :param seed:
    :param net:
    :param training:
    :param validation:
    :param model_name:
    :param epochs:
    :param patience:
    :param verbose:
    """
    # Init
    path = config['model_path']

    try:
        net.load_model(os.path.join(path, model_name))
    except IOError:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)

        if verbose > 1:
            print('Preparing the training datasets / dataloaders')

        # Training
        if verbose > 1:
            print('< Training dataset >')
        dtrain, ltrain = training
        train_dataset = config['training'](dtrain, ltrain)

        if verbose > 1:
            print('Dataloader creation <with validation>')
        train_loader = DataLoader(
            train_dataset, config['train_batch'], True, num_workers=8
        )

        # Validation (training cases)
        if verbose > 1:
            print('< Validation dataset >')
        dval, lval = validation
        val_dataset = config['validation'](dval, lval)

        if verbose > 1:
            print('Dataloader creation <val>')
        val_loader = DataLoader(
            val_dataset, config['test_batch'], num_workers=8
        )

        if verbose > 1:
            print(
                'Training / validation samples samples = '
                '{:02d} / {:02d}'.format(
                    len(train_dataset), len(val_dataset)
                )
            )

        net.fit(
            train_loader, val_loader, epochs=epochs, patience=patience
        )
        net.save_model(os.path.join(path, model_name))


def test(config, net, testing, task, n_classes, verbose=0):
    # Init
    matrix = np.zeros((n_classes, n_classes))
    dataset = config['validation'](testing[0], testing[1])
    test_loader = DataLoader(
        dataset, config['test_batch'], num_workers=32
    )
    test_start = time.time()

    for batch_i, (x, y) in enumerate(test_loader):
        tests = len(test_loader) - batch_i
        test_elapsed = time.time() - test_start
        test_eta = tests * test_elapsed / (batch_i + 1)
        if verbose > 0:
            print(
                '\033[KTesting batch ({:d}/{:d}) {:} ETA {:}'.format(
                    batch_i + 1, len(test_loader),
                    time_to_string(test_elapsed),
                    time_to_string(test_eta),
                ), end='\r'
            )
        prediction = net.inference(
            x.cpu().numpy(), nonbatched=False, task=task)
        predicted = np.argmax(prediction, axis=1)
        target = y.cpu().numpy()
        for t_i, p_i in zip(target, predicted):
            matrix[t_i, p_i] += 1

    return matrix


def update_results(
    config, net, seed, step, training, validation, testing, results, n_classes,
    verbose=0
):
    seed = str(seed)
    for t_i, (tr_i, val_i, tst_i) in enumerate(zip(training, validation, testing)):
        tr_matrix = test(config, net, tr_i, t_i, n_classes, verbose)
        val_matrix = test(config, net, val_i, t_i, n_classes, verbose)
        tst_matrix = test(config, net, tst_i, t_i, n_classes, verbose)
        if isinstance(results, list):
            for results_i in results:
                results_i[seed]['training'][step, t_i, ...] = tr_matrix
                results_i[seed]['validation'][step, t_i, ...] = val_matrix
                results_i[seed]['testing'][step, t_i, ...] = tst_matrix
        else:
            results[seed]['training'][step, t_i, ...] = tr_matrix
            results[seed]['validation'][step, t_i, ...] = val_matrix
            results[seed]['testing'][step, t_i, ...] = tst_matrix


def empty_confusion_matrix(n_tasks, n_classes):
    return np.zeros((n_tasks + 2, n_tasks, n_classes, n_classes))


def save_results(config, json_name, results):
    path = config['json_path']
    json_file = os.path.join(path, json_name)
    with open(json_file, 'w') as testing_json:
        json.dump(results, testing_json)


"""
> Dummy main function
"""


def main(verbose=2):
    # Init
    c = color_codes()
    options = parse_inputs()
    with open(options['config'], 'r') as stream:
        try:
            config = yaml.load(stream, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
    val_split = config['val_split']
    model_path = config['model_path']
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    json_path = config['json_path']
    if not os.path.isdir(json_path):
        os.mkdir(json_path)
    model_base = os.path.splitext(os.path.basename(options['config']))[0]

    seeds = config['seeds']
    epochs = config['epochs']

    print(
        '{:}[{:}] {:}<Incremental learning framework>{:}'.format(
            c['c'], strftime("%H:%M:%S"), c['y'], c['nc']
        )
    )

    # We want a common starting point
    d_tr, d_te = load_datasets(config)
    n_tasks = len(d_tr)
    n_classes = max(d_tr[-1][0][-1], d_te[-1][0][-1])

    # We also need dictionaries for the training tasks so we can track their
    # evolution. The main difference here, is that we need different
    # dictionaries for each task (or batch). These might be defined later and
    # we will fill these dictionaries accordingly when that happens.
    naive_results = {
        str(seed): {
            'training': empty_confusion_matrix(n_tasks, n_classes),
            'validation': empty_confusion_matrix(n_tasks, n_classes),
            'testing': empty_confusion_matrix(n_tasks, n_classes),
        }
        for seed in seeds
    }
    ewc_results = deepcopy(naive_results)
    gem_results = deepcopy(naive_results)
    agem_results = deepcopy(naive_results)
    sgem_results = deepcopy(naive_results)
    ngem_results = deepcopy(naive_results)
    ind_results = deepcopy(naive_results)
    all_methods = ['naive', 'ewc', 'gem', 'agem', 'sgem', 'ngem', 'ind']
    all_results = [
        naive_results, ewc_results, gem_results, agem_results, sgem_results,
        ngem_results, ind_results
    ]

    # Main loop with all the seeds
    for test_n, seed in enumerate(seeds):
        print(
            '{:}[{:}] {:}Starting cross-validation (model: {:}){:}'
            ' (seed {:d}){:}'.format(
                c['clr'] + c['c'], strftime("%H:%M:%S"), c['g'], model_base,
                c['nc'] + c['y'], seed, c['nc']
            )
        )
        # Network init (random weights)
        np.random.seed(seed)
        torch.manual_seed(seed)
        net = config['network'](n_outputs=n_classes)
        starting_model = os.path.join(
            model_path,
            '{:}-start.s{:05d}.pt'.format(model_base, seed)
        )
        net.save_model(starting_model)
        n_param = sum(
            p.numel() for p in net.parameters() if p.requires_grad
        )

        if verbose > 0:
            print(
                '{:}Testing initial weights{:} - {:02d}/{:02d} '
                '({:} parameters)'.format(
                    c['clr'] + c['c'], c['nc'], test_n + 1,
                    len(config['seeds']), c['b'] + str(n_param) + c['nc']
                )
            )

        # Training
        # Here we'll do the training / validation split...
        # Training and testing split
        # We account for a validation set or the lack of it. The reason for
        # this is that we want to measure forgetting and that is easier to
        # measure if we only focus on the training set and leave the testing
        # set as an independent generalisation test.
        if val_split > 0:
            training_tasks = [
                (x[int(len(x) * val_split):], y[int(len(x) * val_split):])
                for _, x, y in d_tr
            ]
            validation_tasks = [
                (x[:int(len(x) * val_split)], y[:int(len(x) * val_split)])
                for _, x, y in d_tr
            ]
        else:
            training_tasks = validation_tasks = [(x, y) for _, x, y in d_tr]
        testing_tasks = [(x, y) for _, x, y in d_te]

        # Baseline model (full continuum access)
        try:
            lr = config['lr']
        except KeyError:
            lr = 1e-3
        net = config['network'](n_outputs=n_classes, lr=lr)
        net.load_model(starting_model)
        training_set = (
            torch.cat([x for x, _ in training_tasks]),
            torch.cat([y for _, y in training_tasks])
        )
        validation_set = (
            torch.cat([x for x, _ in validation_tasks]),
            torch.cat([y for _, y in validation_tasks])
        )
        model_name = os.path.join(
            model_path,
            '{:}-bl.s{:05d}.pt'.format(
                model_base, seed
            )
        )
        print(
            '{:}Starting baseline{:} - {:02d}/{:02d} '
            '({:} parameters)'.format(
                c['clr'] + c['c'], c['nc'],
                test_n + 1, len(config['seeds']),
                c['b'] + str(n_param) + c['nc']
            )
        )
        # Init results
        update_results(
            config, net, seed, 1, training_tasks, validation_tasks, testing_tasks,
            all_results, n_classes, 2
        )
        # Baseline (all data) training and results
        train(
            config, seed, net, training_set, validation_set,
            model_name, epochs * 10, 10, 2
        )
        update_results(
            config, net, seed,  0, training_tasks, validation_tasks, testing_tasks,
            all_results, n_classes, 2
        )

        # Naive approach. We just partition the data and update the model
        # with each new batch without caring about previous samples
        net = MetaModel(
            config['network'](n_outputs=n_classes, lr=lr), best=False
        )
        net.model.load_model(starting_model)

        # Independent approach. We have a duplicate model for each task.
        # We also use the previously learned blocks for future tasks. This is
        # called finetune in the original repo. Here we use it by default for
        # simplicity. Might add the option later.
        ind_net = Independent(
            config['network'](n_outputs=n_classes, lr=lr), best=False,
            n_tasks=n_tasks
        )
        for net_i in ind_net.model:
            net_i.load_model(starting_model)

        # EWC approach. We use a penalty term / regularization loss
        # to ensure previous data isn't forgotten.
        try:
            ewc_weight = config['ewc_weight']
        except KeyError:
            ewc_weight = 1
        try:
            ewc_binary = config['ewc_binary']
        except KeyError:
            ewc_binary = False

        ewc_net = EWC(
            config['network'](n_outputs=n_classes, lr=lr), best=False,
            ewc_weight=ewc_weight, ewc_binary=ewc_binary
        )
        ewc_net.model.load_model(starting_model)

        # GEM approaches. We group all the GEM-related approaches here for
        # simplicity. All parameters should be shared for a fair comparison.
        try:
            gem_weight = config['gem_weight']
        except KeyError:
            gem_weight = 0.5
        try:
            gem_memories = config['gem_memories']
        except KeyError:
            gem_memories = 256

        gem_net = GEM(
            config['network'](n_outputs=n_classes, lr=lr), best=False,
            n_memories=gem_memories, memory_strength=gem_weight,
            n_tasks=n_tasks, n_classes=n_classes, split=True
        )
        gem_net.model.load_model(starting_model)
        agem_net = AGEM(
            config['network'](n_outputs=n_classes, lr=lr), best=False,
            n_memories=gem_memories, memory_strength=gem_weight,
            n_tasks=n_tasks, n_classes=n_classes, split=True
        )
        agem_net.model.load_model(starting_model)
        sgem_net = SGEM(
            config['network'](n_outputs=n_classes, lr=lr), best=False,
            n_memories=gem_memories, memory_strength=gem_weight,
            n_tasks=n_tasks, n_classes=n_classes, split=True
        )
        sgem_net.model.load_model(starting_model)
        ngem_net = NGEM(
            config['network'](n_outputs=n_classes, lr=lr), best=False,
            n_memories=gem_memories, memory_strength=gem_weight,
            n_tasks=n_tasks, n_classes=n_classes, split=True
        )
        ngem_net.model.load_model(starting_model)

        for t_i, (training_set, validation_set) in enumerate(
                zip(training_tasks, validation_tasks)
        ):
            # < NAIVE >
            print(
                '{:}Starting task - naive {:02d}/{:02d}{:} - {:02d}/{:02d} '
                '({:} parameters)'.format(
                    c['clr'] + c['c'], t_i + 1, n_tasks, c['nc'],
                    test_n + 1, len(config['seeds']),
                    c['b'] + str(n_param) + c['nc']
                )
            )

            # We train the naive model on the current task
            model_name = os.path.join(
                model_path,
                '{:}-naive-t{:02d}.s{:05d}.pt'.format(
                    model_base, t_i, seed
                )
            )
            train(
                config, seed, net, training_set, validation_set,
                model_name, epochs, epochs, 2
            )
            net.reset_optimiser()
            update_results(
                config, net, seed,  t_i, training_tasks, validation_tasks,
                testing_tasks, naive_results, n_classes, 2
            )

            # < Independent >
            print(
                '{:}Starting task - Independent {:02d}/{:02d}{:} - '
                '{:02d}/{:02d} ({:} parameters)'.format(
                    c['clr'] + c['c'], t_i + 1, n_tasks, c['nc'],
                    test_n + 1, len(config['seeds']),
                    c['b'] + str(n_param) + c['nc']
                )
            )

            # We train the naive model on the current task
            model_name = os.path.join(
                model_path,
                '{:}-ind-t{:02d}.s{:05d}.pt'.format(
                    model_base, t_i, seed
                )
            )
            train(
                config, seed, ind_net, training_set, validation_set,
                model_name, epochs, epochs, 2
            )
            update_results(
                config, ind_net, seed, t_i, training_tasks, validation_tasks,
                testing_tasks, ind_results, n_classes, 2
            )

            # < EWC >
            print(
                '{:}Starting task - EWC {:02d}/{:02d}{:} - {:02d}/{:02d} '
                '({:} parameters)'.format(
                    c['clr'] + c['c'], t_i + 1, n_tasks, c['nc'],
                    test_n + 1, len(config['seeds']),
                    c['b'] + str(n_param) + c['nc']
                )
            )

            # We train the naive model on the current task
            model_name = os.path.join(
                model_path,
                '{:}-ewc-t{:02d}.s{:05d}.pt'.format(
                    model_base, t_i, seed
                )
            )
            train(
                config, seed, ewc_net, training_set, validation_set,
                model_name, epochs, epochs, 2
            )
            ewc_net.reset_optimiser()
            update_results(
                config, ewc_net, seed, t_i, training_tasks, validation_tasks,
                testing_tasks, ewc_results, n_classes, 2
            )

            # < GEM >
            # Original GEM
            print(
                '{:}Starting task - GEM {:02d}/{:02d}{:} - {:02d}/{:02d} '
                '({:} parameters)'.format(
                    c['clr'] + c['c'], t_i + 1, n_tasks, c['nc'],
                    test_n + 1, len(config['seeds']),
                    c['b'] + str(n_param) + c['nc']
                )
            )

            # We train the naive model on the current task
            model_name = os.path.join(
                model_path,
                '{:}-gem-t{:02d}.s{:05d}.pt'.format(
                    model_base, t_i, seed
                )
            )
            train(
                config, seed, gem_net, training_set, validation_set,
                model_name, epochs, epochs, 2
            )
            gem_net.reset_optimiser()
            update_results(
                config, gem_net, seed, t_i, training_tasks, validation_tasks,
                testing_tasks, gem_results, n_classes, 2
            )

            # Average GEM
            print(
                '{:}Starting task - AGEM {:02d}/{:02d}{:} - {:02d}/{:02d} '
                '({:} parameters)'.format(
                    c['clr'] + c['c'], t_i + 1, n_tasks, c['nc'],
                    test_n + 1, len(config['seeds']),
                    c['b'] + str(n_param) + c['nc']
                )
            )

            # We train the naive model on the current task
            model_name = os.path.join(
                model_path,
                '{:}-agem-t{:02d}.s{:05d}.pt'.format(
                    model_base, t_i, seed
                )
            )
            train(
                config, seed, agem_net, training_set, validation_set,
                model_name, epochs, epochs, 2
            )
            agem_net.reset_optimiser()
            update_results(
                config, agem_net, seed, t_i, training_tasks, validation_tasks,
                testing_tasks, agem_results, n_classes, 2
            )

            # Stochastic GEM
            print(
                '{:}Starting task - SGEM {:02d}/{:02d}{:} - {:02d}/{:02d} '
                '({:} parameters)'.format(
                    c['clr'] + c['c'], t_i + 1, n_tasks, c['nc'],
                    test_n + 1, len(config['seeds']),
                    c['b'] + str(n_param) + c['nc']
                )
            )

            # We train the naive model on the current task
            model_name = os.path.join(
                model_path,
                '{:}-sgem-t{:02d}.s{:05d}.pt'.format(
                    model_base, t_i, seed
                )
            )
            train(
                config, seed, sgem_net, training_set, validation_set,
                model_name, epochs, epochs, 2
            )
            sgem_net.reset_optimiser()
            update_results(
                config, sgem_net, seed, t_i, training_tasks, validation_tasks,
                testing_tasks, sgem_results, n_classes, 2
            )

            # PCA-based GEM
            print(
                '{:}Starting task - NGEM {:02d}/{:02d}{:} - {:02d}/{:02d} '
                '({:} parameters)'.format(
                    c['clr'] + c['c'], t_i + 1, n_tasks, c['nc'],
                    test_n + 1, len(config['seeds']),
                    c['b'] + str(n_param) + c['nc']
                )
            )

            # We train the naive model on the current task
            model_name = os.path.join(
                model_path,
                '{:}-ngem-t{:02d}.s{:05d}.pt'.format(
                    model_base, t_i, seed
                )
            )
            train(
                config, seed, ngem_net, training_set, validation_set,
                model_name, epochs, epochs, 2
            )
            ngem_net.reset_optimiser()
            update_results(
                config, ngem_net, seed, t_i, training_tasks, validation_tasks,
                testing_tasks, ngem_results, n_classes, 2
            )

    for results_i, results_name in zip(all_results, all_methods):
        save_results(
            config, '{:}-{:}_results.json'.format(model_base, results_name),
            results_i
        )


if __name__ == '__main__':
    torch.backends.cudnn.enabled = False
    main()
