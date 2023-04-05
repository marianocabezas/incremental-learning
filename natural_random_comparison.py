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
    options = vars(parser.parse_args())

    return options


"""
> Data functions
"""


def load_datasets(experiment_config):
    data_path = experiment_config['path']
    if os.path.exists(data_path):
        d_tr, d_te = torch.load(data_path)
    else:
        data_packages = data_path.split('.')
        datasets = importlib.import_module('.'.join(data_packages[:-1]))
        d_tr = getattr(datasets, data_packages[-1])(
            '/tmp', train=True, download=True
        )
        d_te = getattr(datasets, data_packages[-1])(
            '/tmp', train=False, download=True
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


def process_net(
    config, net, model_name, seed, nc_per_task, training_set,
    training_tasks, testing_tasks,
    task, epochs, n_classes, results
):
    net.to(net.device)
    for epoch in range(epochs):
        train(
            config, seed, net, training_set,
            model_name, 1, 1, task, 2
        )
        update_results(
            config, net, seed, epoch + 1, nc_per_task, task + 2, training_tasks,
            testing_tasks, results, n_classes, 2
        )

    net.reset_optimiser()
    net.to(torch.device('cpu'))


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
                epochs=epochs, patience=patience, task=task
            )
        except TypeError:
            net.fit(train_loader, val_loader, epochs=epochs, patience=patience)
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
        if verbose > 0:
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
        print(task_mask)
        task_predicted = task_mask[np.argmax(
            prediction[:, task_mask], axis=1
        )]
        target = y.cpu().numpy()

        accuracy_list.append(target == predicted)
        task_accuracy_list.append(target == task_predicted)
        class_list.append(target)

        for t_i, p_i, tp_i in zip(
            target, predicted, task_predicted
        ):
            matrix[t_i, p_i] += 1
            task_matrix[t_i, tp_i] += 1

    print([a.shape for a in accuracy_list])
    accuracy = np.concatenate(accuracy_list)
    print([a.shape for a in task_accuracy_list])
    task_accuracy = np.concatenate(task_accuracy_list)
    print([a.shape for a in class_list])
    classes = np.concatenate(class_list)

    return matrix, task_matrix, accuracy, task_accuracy, classes


def update_results(
    config, net, seed, epoch, nc_per_task, step, training, testing,
    results, n_classes, verbose=0
):
    def _update_results(results_dict):
        results_dict[seed][k]['training'][step, ...] += tr_matrix
        results_dict[seed][k]['testing'][step, ...] += tst_matrix
        results_dict[seed][k]['task_training'][step, ...] += ttr_matrix
        results_dict[seed][k]['task_testing'][step, ...] += ttst_matrix
        if step == 0:
            n_steps = len(results_dict[seed][k]['accuracy_training'])
            for tr_k in np.unique(tr_classes):
                results_dict[seed][k]['accuracy_training'][
                    :, epoch, tr_k, :
                ] = np.repeat(
                    np.expand_dims(tr_acc[tr_classes == tr_k], axis=0),
                    n_steps, axis=0
                )
                results_dict[seed][k]['task_accuracy_training'][
                    :, epoch, tr_k, :
                ] = np.repeat(
                    np.expand_dims(ttr_acc[tr_classes == tr_k], axis=0),
                    n_steps, axis=0
                )
            for tst_k in np.unique(tr_classes):
                results_dict[seed][k]['accuracy_testing'][
                    :, epoch, tst_k, :
                ] = np.repeat(
                    np.expand_dims(tst_acc[tst_classes == tst_k], axis=0),
                    n_steps, axis=0
                )
                results_dict[seed][k]['task_accuracy_testing'][
                    :, epoch, tst_k, :
                ] = np.repeat(
                    np.expand_dims(ttst_acc[tst_classes == tst_k], axis=0),
                    n_steps, axis=0
                )
        elif step > 1:
            for tr_k in np.unique(tr_classes):
                results_dict[seed][k]['accuracy_training'][
                    step, epoch, tr_k, :
                ] = tr_acc[tr_classes == tr_k]
                results_dict[seed][k]['task_accuracy_training'][
                    step, epoch, tr_k, :
                ] = ttr_acc[tr_classes == tr_k]
            for tst_k in np.unique(tr_classes):
                results_dict[seed][k]['accuracy_testing'][
                    step, epoch, tst_k, :
                ] = tst_acc[tst_classes == tst_k]
                results_dict[seed][k]['task_accuracy_testing'][
                    step, epoch, tst_k, :
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
        try:
            _update_results(results)

        except KeyError:
            for results_i in results.values():
                _update_results(results_i)

    test_elapsed = time.time() - test_start
    if verbose > 0:
        print('\033[KTesting finished {:}'.format(time_to_string(test_elapsed)))


def empty_confusion_matrix(n_tasks, n_classes):
    return np.zeros((n_tasks + 2, n_classes, n_classes))


def empty_model_accuracies(n_tasks, n_epochs, n_classes, n_samples):
    return np.zeros((n_tasks, n_epochs, n_classes, n_samples))


def save_results(config, json_name, results):
    path = config['json_path']
    json_file = os.path.join(path, json_name)
    results_tmp = deepcopy(results)
    for incr_name, r_incr in results.items():
        for seed, r_seed in r_incr.items():
            for name, r_numpy in r_seed.items():
                if isinstance(r_numpy, np.ndarray):
                    results_tmp[incr_name][seed][name] = r_numpy.tolist()
                elif isinstance(r_numpy, dict):
                    for loss, r_array in r_numpy.items():
                        if isinstance(r_array, np.ndarray):
                            r = r_array.tolist()
                            results_tmp[incr_name][seed][name][loss] = r

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
    model_base = os.path.splitext(os.path.basename(options['config']))[0]

    seeds = config['seeds']
    epochs = config['epochs']

    try:
        pretrained = config['pretrained']
    except KeyError:
        pretrained = False
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
                    'training': empty_confusion_matrix(
                        n_classes // nc_per_task, n_classes
                    ),
                    'testing': empty_confusion_matrix(
                        n_classes // nc_per_task, n_classes
                    ),
                    'task_training': empty_confusion_matrix(
                        n_classes // nc_per_task, n_classes
                    ),
                    'task_testing': empty_confusion_matrix(
                        n_classes // nc_per_task, n_classes
                    ),
                    'accuracy_training': empty_model_accuracies(
                        n_classes // nc_per_task, epochs, n_classes, s_tr
                    ),
                    'accuracy_testing': empty_model_accuracies(
                        n_classes // nc_per_task, epochs, n_classes, s_te
                    ),
                    'task_accuracy_training': empty_model_accuracies(
                        n_classes // nc_per_task, epochs, n_classes, s_tr
                    ),
                    'task_accuracy_testing': empty_model_accuracies(
                        n_classes // nc_per_task, epochs, n_classes, s_te
                    )
                }
            for nc_per_task in class_list
        }
        for seed in seeds
    }

    for model in config['incremental']:
        incr_name = model[0]
        all_results[incr_name] = deepcopy(base_results)

    # Main loop with all the seeds
    for test_n, seed in enumerate(seeds):
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
            all_incr = {}
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
                    '[{:02d}/{:02d}] ({:} parameters)'.format(
                        c['clr'] + c['c'], c['nc'], test_n + 1,
                        k_i, len(class_list),
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
            model_name = os.path.join(
                model_path,
                '{:}-bl.s{:05d}.pt'.format(
                    model_base, seed
                )
            )

            # Init results
            update_results(
                config, net, seed, 0, nc_per_task, 1, training_tasks, testing_tasks,
                all_results, n_classes, 2
            )
            print(
                '{:}Starting baseline{:} - {:02d}/{:02d} '
                '({:} parameters)'.format(
                    c['clr'] + c['c'], c['nc'],
                    test_n + 1, len(config['seeds']),
                    c['b'] + str(n_param) + c['nc']
                )
            )
            # Baseline (all data) training and results
            verbose = 0
            for epoch in range(epochs):
                train(
                    config, seed, net, training_set,
                    model_name, epochs * n_tasks, n_tasks, 2
                )
                update_results(
                    config, net, seed, epoch + 1,  nc_per_task, 0, training_tasks, testing_tasks,
                    all_results, n_classes, 2
                )

            for model in config['incremental']:
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

                all_incr[incr_name] = incr_model(
                    network(
                        n_outputs=n_classes, lr=lr, pretrained=pretrained
                    ), False, memory_manager, n_classes, n_tasks, **extra_params
                )
                try:
                    if isinstance(all_incr[incr_name].model, ModuleList):
                        for model_i in all_incr[incr_name].model:
                            model_i.load_model(starting_model)
                    else:
                        all_incr[incr_name].model.load_model(starting_model)
                except AttributeError:
                    pass
                all_incr[incr_name].to(torch.device('cpu'))
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            for t_i, training_set in enumerate(training_tasks):

                for incr_name, results_i in all_results.items():
                    print(
                        '{:}Starting task - {:} {:02d}/{:02d}{:} - {:02d}/{:02d} '
                        '({:} parameters)'.format(
                            c['clr'] + c['c'],
                            c['nc'] + c['y'] + incr_name + c['nc'] + c['c'],
                            t_i + 1, n_tasks, c['nc'],
                            test_n + 1, len(config['seeds']),
                            c['b'] + str(n_param) + c['nc']
                        )
                    )

                    # We train the naive model on the current task
                    model_name = os.path.join(
                        model_path,
                        '{:}-{:}-t{:02d}.s{:05d}.pt'.format(
                            model_base, incr_name, t_i, seed
                        )
                    )
                    process_net(
                        config, all_incr[incr_name], model_name, seed,
                        nc_per_task, training_set, training_tasks, testing_tasks,
                        t_i, epochs, n_classes, results_i
                    )

            for model in config['incremental']:
                incr_name = model[0]
                results_i = all_results[incr_name][str(seed)][str(nc_per_task)]
                train_log = all_incr[incr_name].train_log
                val_log = all_incr[incr_name].val_log
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
