import argparse
import os
import time
import json
import random
import numpy as np
import importlib
import yaml
import torch
from torch.nn import ModuleList
from torch.utils.data import DataLoader
from time import strftime
from copy import deepcopy
from scipy.special import softmax
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


def load_label_data(config, normal_class='normal', disease_class='mixed'):
    json_file = config['labels']
    with open(json_file, 'r') as labels_json:
        labels_dict = json.load(labels_json)

    sorted_classes, task_subs = tuple(zip(*[
        (k[0], list(k[1].keys())) for k in sorted(
            labels_dict['train'].items(),
            key=lambda k_dict: len(k_dict[1]),
        )
        if k[0] != normal_class
    ]))
    classes = [normal_class] + list(sorted_classes)
    class_range = np.arange(0, len(classes), dtype=int)
    class_labels = [(class_range == k).astype(int) for k in class_range]

    train_tasks = []
    train_labels = []
    n_normal = 0
    normal_train = list(labels_dict['train'][normal_class].keys())
    for k, subs in enumerate(task_subs):
        n_subs = len(subs)
        task = list(subs) + normal_train[n_normal:n_normal + n_subs]
        n_normal += n_subs
        train_tasks.append(task)
        train_labels.append(
            [class_labels[0]] * n_subs + [class_labels[k + 1]] * n_subs,
        )

    normal_test = list(labels_dict['test'][normal_class].keys())
    disease_test = list(labels_dict['test'][disease_class].keys())
    test_normal_labels = [class_labels[0]] * len(normal_test)
    test_disesase_labels = [
        np.logical_or.reduce(
            [
                [k == k_sub for k in classes]
                for k_sub in study_data['clinical']],
            axis=0
        ).astype(int)
        for study, study_data in labels_dict['test'][disease_class].items()
    ]
    test_labels = test_normal_labels + test_disesase_labels
    test_tasks = normal_test + disease_test

    train_data = train_tasks, train_labels
    test_data = test_tasks, test_labels

    return [normal_class] + list(sorted_classes), train_data, test_data


"""
> Network functions
"""


def process_net(
    config, net, model_name, seed, fold,training_set, validation_set,
    training_tasks, validation_tasks, testing_tasks, mixed_tasks,
    task, offset1, offset2, epochs, n_classes, results
):
    net.to(net.device)
    train(
        config, seed, net, training_set, validation_set,
        model_name, epochs, epochs, task, offset1, offset2, 2
    )
    net.reset_optimiser()
    update_results(
        config, net, seed, task + 2, fold, training_tasks, validation_tasks,
        testing_tasks, mixed_tasks, results, n_classes, 2
    )
    net.to(torch.device('cpu'))


def train(
    config, seed, net, training, validation, model_name, epochs, patience, task,
    offset1, offset2, verbose=0
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
    :param task:
    :param offset1:
    :param offset2:
    :param verbose:
    """
    # Init
    path = config['model_path']
    d_path = config['path']
    image_name = config['image']

    try:
        net.load_model(os.path.join(path, model_name))
    except IOError:
        datasets = importlib.import_module('datasets')
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
        train_dataset = getattr(datasets, config['training'])(
            d_path, image_name, dtrain, ltrain
        )

        if verbose > 1:
            print('Dataloader creation <with validation>')
        train_loader = DataLoader(
            train_dataset, config['train_batch'], True, num_workers=8,
            drop_last=True
        )

        # Validation (training cases)
        if verbose > 1:
            print('< Validation dataset >')
        dval, lval = validation
        val_dataset = getattr(datasets, config['validation'])(
            d_path, image_name, dval, lval
        )

        if verbose > 1:
            print('Dataloader creation <val>')
        val_loader = DataLoader(
            val_dataset, config['test_batch'], num_workers=8,
            drop_last=True
        )

        if verbose > 1:
            print(
                'Training / validation samples = '
                '{:02d} / {:02d}'.format(
                    len(train_dataset), len(val_dataset)
                )
            )

        if task is None:
            net.fit(train_loader, val_loader, epochs=epochs, patience=patience)
        else:
            net.fit(
                train_loader, val_loader, offset1=offset1, offset2=offset2,
                epochs=epochs, patience=patience, task=task
            )
        net.save_model(os.path.join(path, model_name))


def test(config, net, testing, task, n_classes, verbose=0):
    # Init
    d_path = config['path']
    image_name = config['image']
    matrix = np.zeros((n_classes, 2, 2))
    datasets = importlib.import_module('datasets')
    dataset = getattr(datasets, config['validation'])(
        d_path, image_name, testing[0], testing[1]
    )
    test_loader = DataLoader(
        dataset, config['test_batch'], num_workers=8
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
            x.cpu().numpy(), nonbatched=False, task=task
        )
        target = y.cpu().numpy()

        for k in range(n_classes):
            if target[k]:
                # Positive class example
                if prediction[k]:
                    # TP
                    matrix[k, 1, 1] += 1
                else:
                    # FN
                    matrix[k, 1, 0] += 1
            else:
                # Negative class example
                if prediction[k]:
                    # FP
                    matrix[k, 0, 1] += 1
                else:
                    # TN
                    matrix[k, 0, 0] += 1
    return matrix


def update_results(
    config, net, seed, step, fold, training, validation, testing, mixed_testing,
    results, n_classes, verbose=0
):
    seed = str(seed)
    test_start = time.time()

    multitst_matrix = test(
        config, net, mixed_testing, None, n_classes, verbose
    )
    try:
        if fold is None:
            for fold_i in range(len(results[seed]['testing'])):
                results[seed]['testing'][fold_i][step, ...] += multitst_matrix
        else:
            results[seed]['testing'][fold][step, ...] += multitst_matrix
    except KeyError:
        for results_i in results.values():
            if fold is None:
                for fold_i in range(len(results_i[seed]['testing'])):
                    results_i[seed]['testing'][fold_i][step, ...] += multitst_matrix
            else:
                results_i[seed]['testing'][fold][step, ...] += multitst_matrix

    for t_i, (tr_i, val_i, tst_i) in enumerate(zip(training, validation, testing)):
        tr_matrix = test(
            config, net, tr_i, t_i, n_classes, verbose
        )
        val_matrix = test(
            config, net, val_i, t_i, n_classes, verbose
        )
        tst_matrix = test(
            config, net, tst_i, t_i, n_classes, verbose
        )

        try:
            results[seed]['training'][step, ...] += tr_matrix
            results[seed]['validation'][step, ...] += val_matrix
            results[seed]['xval'][step, ...] += tst_matrix
        except KeyError:
            for results_i in results.values():
                results_i[seed]['training'][step, ...] += tr_matrix
                results_i[seed]['validation'][step, ...] += val_matrix
                results_i[seed]['xval'][step, ...] += tst_matrix

    test_elapsed = time.time() - test_start
    if verbose > 0:
        print('\033[KTesting finished {:}'.format(time_to_string(test_elapsed)))


def empty_confusion_matrix(n_tasks, n_classes):
    return np.zeros((n_tasks + 2, n_classes, 2, 2))


def save_results(config, json_name, results):
    path = config['json_path']
    json_file = os.path.join(path, json_name)
    results_tmp = deepcopy(results)
    for meta_name, r_meta in results.items():
        for seed, r_seed in r_meta.items():
            for name, r_numpy in r_seed.items():
                results_tmp[meta_name][seed][name] = r_numpy.tolist()

    with open(json_file, 'w') as testing_json:
        json.dump(results_tmp, testing_json)


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
    n_folds = config['folds']
    epochs = config['epochs']

    try:
        memories = config['memories']
    except KeyError:
        memories = 1000

    print(
        '{:}[{:}] {:}<Incremental learning framework>{:}'.format(
            c['c'], strftime("%H:%M:%S"), c['y'], c['nc']
        )
    )

    models = importlib.import_module('models')
    network = getattr(models, config['network'])
    meta = importlib.import_module('continual')
    memory = importlib.import_module('memory')

    # We want a common starting point
    classes, d_tr, d_te = load_label_data(config)
    train_tasks, train_labels = d_tr
    n_tasks = len(train_tasks)
    n_classes = len(classes)
    nc_per_task = 2

    # We also need dictionaries for the training tasks, so we can track their
    # evolution. The main difference here, is that we need different
    # dictionaries for each task (or batch). These might be defined later, and
    # we will fill these dictionaries accordingly when that happens.
    all_results = {}
    base_results = {
        str(seed): {
            'training': empty_confusion_matrix(n_tasks, n_classes),
            'xval': empty_confusion_matrix(n_tasks, n_classes),
            'validation': empty_confusion_matrix(n_tasks, n_classes),
            'testing': [
                empty_confusion_matrix(n_tasks, n_classes)
                for _ in range(n_folds)
            ],
        }
        for seed in seeds
    }

    for model in config['metamodels']:
        meta_name = model[0]
        all_results[meta_name] = deepcopy(base_results)

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
        net = network(n_outputs=n_classes)
        all_metas = {}
        starting_model = os.path.join(
            model_path,
            '{:}-start.s{:05d}.pt'.format(model_base, seed)
        )
        net.save_model(starting_model)
        n_param = sum(
            p.numel() for p in net.parameters() if p.requires_grad
        )

        shuffled_tasks = []
        shuffled_labels = []
        for t_list, lab_list in zip(train_tasks, train_labels):
            shuffled_indices = np.random.permutation(len(t_list))
            shuffled_tasks.append(np.array(t_list)[shuffled_indices].tolist())
            shuffled_labels.append(np.array(lab_list)[shuffled_indices].tolist())

        for i in range(n_folds):
            train_tasks_xval = []
            train_labels_xval = []
            test_tasks_xval = []
            test_labels_xval = []
            for t_list, lab_list in zip(shuffled_tasks, shuffled_labels):
                task_ini = len(t_list) * i // n_folds
                task_end = len(t_list) * (i + 1) // n_folds
                train_tasks_xval.append(
                    t_list[task_end:] + t_list[task_ini:]
                )
                train_labels_xval.append(
                    lab_list[task_end:] + lab_list[task_ini:]
                )
                test_tasks_xval.extend(
                    t_list[task_ini:task_end]
                )
                test_labels_xval.extend(
                    lab_list[task_ini:task_end]
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
                    for x, y in zip(train_tasks_xval, train_labels_xval)
                ]
                validation_tasks = [
                    (x[:int(len(x) * val_split)], y[:int(len(x) * val_split)])
                    for x, y in zip(train_tasks_xval, train_labels_xval)
                ]
            else:
                training_tasks = validation_tasks = [
                    (x, y) for x, y in zip(train_tasks_xval, train_labels_xval)
                ]
            testing_tasks = [
                (x, y) for x, y in zip(test_tasks_xval, test_labels_xval)
            ]

            # Baseline model (full continuum access)
            try:
                lr = config['lr']
            except KeyError:
                lr = 1e-3
            net = network(
                n_outputs=n_classes, lr=lr
            )
            net.load_model(starting_model)
            training_set = (
                [study for x, _ in training_tasks for study in x],
                [label for _, y in training_tasks for label in y]
            )
            validation_set = (
                [study for x, _ in validation_tasks for study in x],
                [label for _, y in validation_tasks for label in y]
            )
            model_name = os.path.join(
                model_path,
                '{:}-bl.s{:05d}.f{:02d}.pt'.format(
                    model_base, seed, i
                )
            )

            # Init results
            if verbose > 0:
                print(
                    '{:}Testing initial weights{:} - {:02d}/{:02d} '
                    '({:} parameters)'.format(
                        c['clr'] + c['c'], c['nc'], test_n + 1,
                        len(config['seeds']), c['b'] + str(n_param) + c['nc']
                    )
                )

            update_results(
                config, net, seed, 1, i, training_tasks, validation_tasks,
                testing_tasks, d_te, all_results, n_classes, 2
            )
            print(
                '{:}Starting baseline{:} [fold {:02d}/{:02d}]{:} - '
                '{:02d}/{:02d} ({:} parameters)'.format(
                    c['clr'] + c['c'], c['nc'] + c['b'],
                    i + 1, n_folds, c['nc'],
                    test_n + 1, len(config['seeds']),
                    c['b'] + str(n_param) + c['nc']
                )
            )
            # Baseline (all data) training and results
            train(
                config, seed, net, training_set, validation_set,
                model_name, epochs * n_tasks, n_tasks, None, None, None, 2
            )
            update_results(
                config, net, seed, 0, None, training_tasks, validation_tasks,
                testing_tasks, d_te, all_results, n_classes, 2
            )

            for model in config['metamodels']:
                try:
                    meta_name, meta_class, memory_class, extra_params = model
                except ValueError:
                    meta_name, meta_class, memory_class = model
                    extra_params = {}

                meta_model = getattr(meta, meta_class)

                try:
                    manager = getattr(memory, memory_class)
                    memory_manager = manager(memories, n_classes, n_tasks)
                except TypeError:
                    memory_manager = None

                all_metas[meta_name] = meta_model(
                    network(
                        n_outputs=n_classes, lr=lr
                    ), False, memory_manager, n_classes, n_tasks, **extra_params
                )
                try:
                    if isinstance(all_metas[meta_name].model, ModuleList):
                        for model_i in all_metas[meta_name].model:
                            model_i.load_model(starting_model)
                    else:
                        all_metas[meta_name].model.load_model(starting_model)
                except AttributeError:
                    pass
                all_metas[meta_name].to(torch.device('cpu'))
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            for t_i, (training_set, validation_set) in enumerate(
                    zip(training_tasks, validation_tasks)
            ):

                offset1 = 0
                offset2 = t_i + 1

                for meta_name, results_i in all_results.items():
                    print(
                        '{:}Starting task  {:}[fold {:02d}/{:02d}]{:} - '
                        '{:} {:02d}/{:02d}{:} - {:02d}/{:02d} '
                        '({:} parameters)'.format(
                            c['clr'] + c['c'], c['nc'] + c['b'],
                            i + 1, n_folds, c['nc'] + c['c'],
                            c['nc'] + c['y'] + meta_name + c['nc'] + c['c'],
                            t_i + 1, n_tasks, c['nc'],
                            test_n + 1, len(config['seeds']),
                            c['b'] + str(n_param) + c['nc']
                        )
                    )

                    # We train the naive model on the current task
                    model_name = os.path.join(
                        model_path,
                        '{:}-{:}-t{:02d}.s{:05d}.f{:02d}.pt'.format(
                            model_base, meta_name, t_i, seed, i,
                        )
                    )
                    process_net(
                        config, all_metas[meta_name], model_name, seed, i,
                        training_set, validation_set,
                        training_tasks, validation_tasks, testing_tasks, d_te,
                        t_i, offset1, offset2, epochs, n_classes, results_i
                    )

    save_results(config, '{:}_results.json'.format(model_base), all_results)


if __name__ == '__main__':
    torch.backends.cudnn.enabled = False
    torch.autograd.set_detect_anomaly(True)
    main()
