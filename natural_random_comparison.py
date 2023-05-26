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
    parser.add_argument(
        '-s', '--seed-index',
        dest='seed_idx', type=int,
        help='Index of the random seed. By default, number of seeds are taken '
             'from the config file and all are run. Setting a seed index '
             'helps running parallel processes.'
    )
    parser.add_argument(
        '-m', '--method-index',
        dest='method_idx', type=int,
        help='Index of the method to run in the config. By default all the '
             'methods from the config file are run. Setting a method index '
             'helps running parallel processes.'
    )
    parser.add_argument(
        '-n', '--no-color',
        dest='no_color', default=False, action='store_true',
        help='Whether to print with colors (bad for log files).'
    )
    options = vars(parser.parse_args())

    return options


"""
> Data functions
"""


def load_datasets(experiment_config):
    data_path = experiment_config['path']
    tmp_path = experiment_config['tmp_path']
    if os.path.exists(data_path):
        d_tr, d_te = torch.load(data_path)
    else:
        data_packages = data_path.split('.')
        datasets = importlib.import_module('.'.join(data_packages[:-1]))
        d_tr = getattr(datasets, data_packages[-1])(
            tmp_path, train=True, download=True
        )
        d_te = getattr(datasets, data_packages[-1])(
            tmp_path, train=False, download=True
        )

    s_tr = count_samples(d_tr)
    s_te = count_samples(d_te)

    return d_tr, d_te, s_tr, s_te


def count_samples(dataset):
    all_classes = np.unique([y for _, y in dataset]).tolist()
    samples = np.zeros(len(all_classes))
    for x, y in dataset:
        samples[y] += 1

    return int(np.max(samples))


def split_dataset(dataset, tasks):
    n_tasks = len(tasks)
    tr_images = [[] for _ in range(n_tasks)]
    tr_labels = [[] for _ in range(n_tasks)]
    for x, y in dataset:
        task_index = np.where([np.isin(y, t_i).tolist() for t_i in tasks])[0][0]
        x_numpy = np.moveaxis(
            np.array(x.getdata(), dtype=np.float32), 0, 1
        ) / 255
        tr_images[task_index].append(torch.from_numpy(x_numpy.flatten()))
        tr_labels[task_index].append(torch.tensor(y))
    task_split = [
        (torch.tensor(k_i), torch.stack(im_i), torch.stack(lab_i))
        for k_i, im_i, lab_i in zip(tasks, tr_images, tr_labels)
    ]

    return task_split


def split_data(d_tr, d_te, classes, randomise=True):
    all_classes = np.unique([y for _, y in d_tr]).tolist()
    if randomise:
        all_classes = np.random.permutation(all_classes).tolist()
    n_tasks = len(all_classes) // classes
    if len(all_classes) % classes == 0:
        tasks = [
            tuple(all_classes[i * classes:i * classes + classes])
            for i in range(n_tasks)
        ]

        tasks_tr = split_dataset(d_tr, tasks)
        tasks_te = split_dataset(d_te, tasks)

    else:
        tasks_tr = []
        tasks_te = []

    return tasks_tr, tasks_te, all_classes


"""
> Network functions
"""


def train(
    config, seed, net, training, model_name, epochs, patience, task,
    verbose=0
):
    """

    :param config:
    :param seed:
    :param net:
    :param training:
    :param model_name:
    :param epochs:
    :param patience:
    :param task:
    :param verbose:
    """
    # Init
    path = config['model_path']

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
        dmask, dtrain, ltrain = training
        train_dataset = getattr(datasets, config['training'])(dtrain, ltrain)

        if verbose > 1:
            print('Dataloader creation')
        train_loader = DataLoader(
            train_dataset, config['train_batch'], True, num_workers=8,
            drop_last=True
        )
        val_loader = DataLoader(
            train_dataset, config['train_batch'], num_workers=8,
            drop_last=True
        )

        if verbose > 1:
            print('Training samples = {:02d}'.format(len(train_dataset)))

        try:
            net.fit(
                train_loader, val_loader, task_mask=dmask.to(net.device),
                epochs=epochs, patience=patience, task=task,
                verbose=(not config['no_color'])
            )
        except TypeError:
            net.fit(
                train_loader, val_loader, epochs=epochs, patience=patience,
                verbose=(not config['no_color'])
            )
        net.save_model(os.path.join(path, model_name))


def test(config, net, testing, task, n_classes, verbose=0):
    # Init
    matrix = np.zeros((n_classes, n_classes))
    task_matrix = np.zeros((n_classes, n_classes))
    datasets = importlib.import_module('datasets')
    task_mask = testing[0]
    dataset = getattr(datasets, config['validation'])(testing[1], testing[2])
    test_loader = DataLoader(
        dataset, config['test_batch'], num_workers=32
    )
    test_start = time.time()
    accuracy_list = []
    task_accuracy_list = []
    class_list = []
    for batch_i, (x, y) in enumerate(test_loader):
        tests = len(test_loader) - batch_i
        test_elapsed = time.time() - test_start
        test_eta = tests * test_elapsed / (batch_i + 1)
        if verbose > 0 and config['no_color']:
            print(
                '\033[KTesting batch ({:d}/{:d}) {:} ETA {:}'.format(
                    batch_i + 1, len(test_loader),
                    time_to_string(test_elapsed),
                    time_to_string(test_eta),
                ), end='\r'
            )
        prediction, pred_task = net.inference(
            x.cpu().numpy(), nonbatched=False, task=task
        )

        predicted = np.argmax(prediction, axis=1)
        task_predicted = task_mask[np.argmax(
            prediction[:, task_mask], axis=1
        )].cpu().numpy()
        target = y.cpu().numpy()

        accuracy_list.append(target == predicted)
        task_accuracy_list.append(target == task_predicted)
        class_list.append(target)

        for t_i, p_i, tp_i in zip(
            target, predicted, task_predicted
        ):
            matrix[t_i, p_i] += 1
            task_matrix[t_i, tp_i] += 1

    accuracy = np.concatenate(accuracy_list)
    task_accuracy = np.concatenate(task_accuracy_list)
    classes = np.concatenate(class_list)

    return matrix, task_matrix, accuracy, task_accuracy, classes


def update_results(
    config, net, seed, epoch, total_epochs, nc_per_task, step,
    training, testing, results, n_classes, verbose=0
):
    def _update_results(results_dict, e_indx):
        if epoch < e_indx:
            results_dict[seed][k]['training'][e_indx][
                step, epoch, ...
            ] += tr_matrix
            results_dict[seed][k]['testing'][e_indx][
                step, epoch, ...
            ] += tst_matrix
            results_dict[seed][k]['task_training'][e_indx][
                step, epoch, ...
            ] += ttr_matrix
            results_dict[seed][k]['task_testing'][e_indx][
                step, epoch, ...
            ] += ttst_matrix
            if step == 0:
                for tr_k in np.unique(tr_classes):
                    results_dict[seed][k]['init_accuracy_training'][
                        0, 0, tr_k, :
                    ] = tr_acc[tr_classes == tr_k]
                for tst_k in np.unique(tst_classes):
                    results_dict[seed][k]['init_accuracy_testing'][
                        0, 0, tst_k, :
                    ] = tst_acc[tst_classes == tst_k]
            elif step == 1:
                for tr_k in np.unique(tr_classes):
                    results_dict[seed][k]['base_accuracy_training'][
                        0, epoch, tr_k, :
                    ] = tr_acc[tr_classes == tr_k]
                for tst_k in np.unique(tst_classes):
                    results_dict[seed][k]['base_accuracy_testing'][
                        0, epoch, tst_k, :
                    ] = tst_acc[tst_classes == tst_k]
            else:
                for tr_k in np.unique(tr_classes):
                    results_dict[seed][k]['accuracy_training'][e_indx][
                        step - 2, epoch + 1, tr_k, :
                    ] = tr_acc[tr_classes == tr_k]
                    results_dict[seed][k]['task_accuracy_training'][e_indx][
                        step - 2, epoch + 1, tr_k, :
                    ] = ttr_acc[tr_classes == tr_k]
                for tst_k in np.unique(tst_classes):
                    results_dict[seed][k]['accuracy_testing'][e_indx][
                        step - 2, epoch + 1, tst_k, :
                    ] = tst_acc[tst_classes == tst_k]
                    results_dict[seed][k]['task_accuracy_testing'][e_indx][
                        step - 2, epoch + 1, tst_k, :
                    ] = ttst_acc[tst_classes == tst_k]

    seed = str(seed)
    k = str(nc_per_task)
    test_start = time.time()
    for t_i, (tr_i, tst_i) in enumerate(
            zip(training, testing)
    ):
        tr_matrix, ttr_matrix, tr_acc, ttr_acc, tr_classes = test(
            config, net, tr_i, t_i, n_classes, verbose
        )
        tst_matrix, ttst_matrix, tst_acc, ttst_acc, tst_classes = test(
            config, net, tst_i, t_i, n_classes, verbose
        )
        if step > 1:
            try:
                _update_results(results, total_epochs)

            except KeyError:
                for results_i in results.values():
                    _update_results(results_i, total_epochs)
        else:
            for n_e in range(total_epochs):
                try:
                    _update_results(results, n_e)

                except KeyError:
                    for results_i in results.values():
                        _update_results(results_i, n_e)

    test_elapsed = time.time() - test_start
    if verbose > 0:
        if config['no_color']:
            print(
                'Testing finished {:}'.format(
                    time_to_string(test_elapsed)
                )
            )
        else:
            print(
                '\033[KTesting finished {:}'.format(
                    time_to_string(test_elapsed)
                )
            )


def empty_confusion_matrix(n_tasks, n_epochs, n_classes):
    return np.zeros((n_tasks + 2, n_epochs, n_classes, n_classes))


def empty_model_accuracies(n_tasks, n_epochs, n_classes, n_samples):
    return np.zeros((n_tasks, n_epochs, n_classes, n_samples))


def save_results(config, json_name, results):
    path = config['json_path']
    json_file = os.path.join(path, json_name)
    results_tmp = deepcopy(results)
    for incr_name, r_incr in results.items():
        for seed, r_seed in r_incr.items():
            for nc_x_task, r_nc in r_seed.items():
                for name, r_numpy in r_nc.items():
                    print(name, r_numpy)
                    if isinstance(r_numpy, np.ndarray):
                        results_tmp[incr_name][seed][name] = r_numpy.tolist()
                    elif isinstance(r_numpy, dict):
                        for loss, r_array in r_numpy.items():
                            if isinstance(r_array, np.ndarray):
                                r = r_array.tolist()
                                results_tmp[incr_name][seed][name][loss] = r
                    elif isinstance(r_numpy, list):
                        results_tmp[incr_name][seed][name] = [
                            npy_i.tolist() if isinstance(npy_i, np.ndarray)
                            else npy_i for npy_i in r_numpy
                        ]

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
    model_path = config['model_path']
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    json_path = config['json_path']
    if not os.path.isdir(json_path):
        os.mkdir(json_path)

    master_seed = config['master_seed']
    np.random.seed(master_seed)

    s_idx = options['seed_idx']
    if s_idx is None:
        n_seeds = config['seeds']
        seed_suffix = ''
    else:
        n_seeds = options['seed_idx'] + 1
    seeds = np.random.randint(0, 100000, n_seeds)
    if s_idx is not None:
        seeds = seeds[-1:]
        seed_suffix = '_s{:05d}'.format(seeds[0])

    epochs = config['epochs']
    config['no_color'] = options['no_color']

    try:
        pretrained = config['pretrained']
    except KeyError:
        pretrained = False
    try:
        memories = config['memories']
    except KeyError:
        memories = 1000

    if config['no_color']:
        print(
            '[{:}] <Incremental learning framework>'.format(
                strftime("%H:%M:%S")
            )
        )
    else:
        print(
            '{:}[{:}] {:}<Incremental learning framework>{:}'.format(
                c['c'], strftime("%H:%M:%S"), c['y'], c['nc']
            )
        )

    models = importlib.import_module('models')
    network = getattr(models, config['network'])
    incr = importlib.import_module('continual_masked')
    memory = importlib.import_module('memory')

    # We want a common starting point
    d_tr, d_te, s_tr, s_te = load_datasets(config)
    try:
        class_list = config['classes_task']
    except KeyError:
        class_list = [2]
    n_classes = len(np.unique([y for _, y in d_tr]))
    try:
        randomise = config['randomise_split']
    except KeyError:
        randomise = False

    # We also need dictionaries for the training tasks, so we can track their
    # evolution. The main difference here, is that we need different
    # dictionaries for each task (or batch). These might be defined later, and
    # we will fill these dictionaries accordingly when that happens.
    all_results = {}
    base_results = {
        str(seed): {
            str(nc_per_task): {
                    'tasks': None,
                    'training': [
                        empty_confusion_matrix(
                            n_classes // nc_per_task, n_e + 1, n_classes
                        )
                        for n_e in range(epochs)
                    ],
                    'testing': [
                        empty_confusion_matrix(
                            n_classes // nc_per_task, n_e + 1, n_classes
                        )
                        for n_e in range(epochs)
                    ],
                    'task_training': [
                        empty_confusion_matrix(
                            n_classes // nc_per_task, n_e + 1, n_classes
                        )
                        for n_e in range(epochs)
                    ],
                    'task_testing': [
                        empty_confusion_matrix(
                            n_classes // nc_per_task, n_e + 1, n_classes
                        )
                        for n_e in range(epochs)
                    ],
                    'init_accuracy_training': empty_model_accuracies(
                        1, 1, n_classes, s_tr
                    ),
                    'base_accuracy_training': empty_model_accuracies(
                        1, epochs, n_classes, s_tr
                    ),
                    'accuracy_training': [
                        empty_model_accuracies(
                            n_classes // nc_per_task, n_e + 1, n_classes, s_tr
                        )
                        for n_e in range(epochs)
                    ],
                    'init_accuracy_testing': empty_model_accuracies(
                        1, 1, n_classes, s_te
                    ),
                    'base_accuracy_testing': empty_model_accuracies(
                        1, epochs, n_classes, s_te
                    ),
                    'accuracy_testing': [
                        empty_model_accuracies(
                            n_classes // nc_per_task, n_e + 1, n_classes, s_te
                        )
                        for n_e in range(epochs)
                    ],
                    'task_accuracy_training': [
                        empty_model_accuracies(
                            n_classes // nc_per_task, n_e + 1, n_classes, s_tr
                        )
                        for n_e in range(epochs)
                    ],
                    'task_accuracy_testing': [
                        empty_model_accuracies(
                            n_classes // nc_per_task, n_e + 1, n_classes, s_te
                        )
                        for n_e in range(epochs)
                    ],
                }
            for nc_per_task in class_list
        }
        for seed in seeds
    }

    incremental_list = config['incremental']
    m_idx = options['method_idx']
    if m_idx is None:
        m_suffix = ''
    else:
        incremental_list = incremental_list[m_idx:m_idx + 1]
        m_suffix = '_{:}'.format(incremental_list[0][0])

    for model in incremental_list:
        incr_name = model[0]
        all_results[incr_name] = deepcopy(base_results)

    config_base = os.path.splitext(os.path.basename(options['config']))[0]
    model_base = config_base + m_suffix + seed_suffix

    # Main loop with all the seeds
    for test_n, seed in enumerate(seeds):
        if config['no_color']:
            print(
                '[{:}] Starting cross-validation (model: {:})'
                ' (seed {:d})'.format(
                    strftime("%H:%M:%S"), model_base, seed
                )
            )
        else:
            print(
                '{:}[{:}] {:}Starting cross-validation (model: {:}){:}'
                ' (seed {:d}){:}'.format(
                    c['clr'] + c['c'], strftime("%H:%M:%S"), c['g'], model_base,
                    c['nc'] + c['y'], seed, c['nc']
                )
            )

        for k_i, nc_per_task in enumerate(class_list):
            n_tasks = n_classes // nc_per_task
            # Network init (random weights)
            np.random.seed(seed)
            torch.manual_seed(seed)
            net = network(n_outputs=n_classes, pretrained=pretrained)
            alltraining_tasks, testing_tasks, task_list = split_data(
                d_tr, d_te, nc_per_task, randomise=randomise
            )
            all_incr = [{} for _ in range(epochs)]
            starting_model = os.path.join(
                model_path,
                '{:}-start.s{:05d}.c{:02d}.pt'.format(
                    model_base, seed, nc_per_task
                )
            )
            net.save_model(starting_model)
            n_param = sum(
                p.numel() for p in net.parameters() if p.requires_grad
            )

            if verbose > 0:
                if config['no_color']:
                    print(
                        'Testing initial weights - {:02d}/{:02d} '
                        '[{:02d}/{:02d}] ({:} parameters)'.format(
                            k_i + 1, len(class_list), test_n + 1,
                            len(seeds), str(n_param)
                        )
                    )
                else:
                    print(
                        '{:}Testing initial weights{:} - {:02d}/{:02d} '
                        '[{:02d}/{:02d}] ({:} parameters)'.format(
                            c['clr'] + c['c'], c['nc'],
                            k_i + 1, len(class_list), test_n + 1,
                            len(seeds), c['b'] + str(n_param) + c['nc']
                        )
                    )

            # Training
            # Here we'll do the training / validation split...
            # Training and testing split
            # We account for a validation set or the lack of it. The reason for
            # this is that we want to measure forgetting and that is easier to
            # measure if we only focus on the training set and leave the testing
            # set as an independent generalisation test.
            training_tasks = alltraining_tasks

            # Baseline model (full continuum access)
            try:
                lr = config['lr']
            except KeyError:
                lr = 1e-3
            net = network(
                n_outputs=n_classes, lr=lr, pretrained=pretrained
            )
            net.load_model(starting_model)
            training_set = (
                torch.from_numpy(np.array(range(n_classes))),
                torch.cat([x for _, x, _ in training_tasks]),
                torch.cat([y for _, _, y in training_tasks])
            )

            # Init results
            update_results(
                config, net, seed, 0, epochs, nc_per_task, 1,
                training_tasks, testing_tasks,
                all_results, n_classes, 2
            )
            if config['no_color']:
                print(
                    'Starting baseline - {:02d}/{:02d} '
                    '({:} parameters)'.format(
                        test_n + 1, len(seeds), str(n_param)
                    )
                )
            else:
                print(
                    '{:}Starting baseline{:} - {:02d}/{:02d} '
                    '({:} parameters)'.format(
                        c['clr'] + c['c'], c['nc'],
                        test_n + 1, len(seeds),
                        c['b'] + str(n_param) + c['nc']
                    )
                )
            # Baseline (all data) training and results
            verbose = 0
            for epoch in range(epochs):
                model_name = os.path.join(
                    model_path,
                    '{:}-bl.s{:05d}.c{:02d}.e{:02d}.pt'.format(
                        model_base, seed, nc_per_task, epoch
                    )
                )
                train(
                    config, seed, net, training_set,
                    model_name, 1, n_tasks, 2
                )
                update_results(
                    config, net, seed, epoch, epochs, nc_per_task, 0,
                    training_tasks, testing_tasks, all_results, n_classes, 2
                )

            for model in incremental_list:
                results_i = all_results[model[0]][str(seed)][str(nc_per_task)]
                results_i['tasks'] = task_list
                try:
                    incr_name, incr_class, memory_class, extra_params = model
                except ValueError:
                    incr_name, incr_class, memory_class = model
                    extra_params = {}

                incr_model = getattr(incr, incr_class)

                try:
                    manager = getattr(memory, memory_class)
                    memory_manager = manager(memories, n_classes, n_tasks)
                except TypeError:
                    memory_manager = None

                for n_e in range(epochs):
                    all_incr[n_e][incr_name] = incr_model(
                        network(
                            n_outputs=n_classes, lr=lr, pretrained=pretrained
                        ), False, memory_manager, n_classes, n_tasks, **extra_params
                    )
                    try:
                        if isinstance(all_incr[n_e][incr_name].model, ModuleList):
                            for model_i in all_incr[n_e][incr_name].model:
                                model_i.load_model(starting_model)
                        else:
                            all_incr[n_e][incr_name].model.load_model(starting_model)
                    except AttributeError:
                        pass
                    all_incr[n_e][incr_name].to(torch.device('cpu'))
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

            for t_i, training_set in enumerate(training_tasks):
                for incr_name, results_i in all_results.items():
                    if config['no_color']:
                        print(
                            'Starting task - {:} {:02d}/{:02d} - {:02d}/{:02d} '
                            '({:} parameters) [{:d} classes per task]'.format(
                                incr_name, t_i + 1, n_tasks,
                                test_n + 1, len(seeds),
                                str(n_param), nc_per_task
                            )
                        )
                    else:
                        print(
                            '{:}Starting task - {:} {:02d}/{:02d}{:} -'
                            ' {:02d}/{:02d} ({:} parameters) '
                            '[{:d} classes per task]'.format(
                                c['clr'] + c['c'],
                                c['nc'] + c['y'] + incr_name + c['nc'] + c['c'],
                                t_i + 1, n_tasks, c['nc'],
                                test_n + 1, len(seeds),
                                c['b'] + str(n_param) + c['nc'],
                                nc_per_task
                            )
                        )
                    # We train the incremental model on the current task
                    for n_e in range(epochs):
                        net = all_incr[n_e][incr_name]
                        net.to(net.device)
                        for epoch in range(n_e):
                            model_name = os.path.join(
                                model_path,
                                '{:}-{:}-t{:02d}.s{:05d}.c{:02d}'
                                '.e{:02d}.te{:02d}.pt'.format(
                                    model_base, incr_name, t_i, seed, nc_per_task,
                                    epoch, n_e
                                )
                            )
                            train(
                                config, seed, net, training_set,
                                model_name, 1, 1, t_i, 2
                            )
                            update_results(
                                config, net, seed, epoch, n_e, nc_per_task,
                                t_i + 2, training_tasks, testing_tasks,
                                results_i, n_classes, 2
                            )
                    net.reset_optimiser()
                    net.to(torch.device('cpu'))

            for model in incremental_list:
                incr_name = model[0]
                results_i = all_results[incr_name][str(seed)][str(nc_per_task)]
                for incr_m in all_incr:
                    train_log = incr_m[incr_name].train_log
                    val_log = incr_m[incr_name].val_log
                    if isinstance(train_log, torch.Tensor):
                        results_i['train-log'] = train_log.numpy().tolist()
                    else:
                        results_i['train-log'] = train_log
                    if isinstance(val_log, torch.Tensor):
                        results_i['val-log'] = val_log.numpy().tolist()
                    else:
                        results_i['val-log'] = val_log

    save_results(config, '{:}_results.json'.format(model_base), all_results)


if __name__ == '__main__':
    torch.backends.cudnn.enabled = False
    torch.autograd.set_detect_anomaly(True)
    main()
