import argparse
import os
import time
import json
import nibabel
import numpy as np
import pandas as pd
import datasets
import models
import yaml
import torch
from functools import partial
from torch.utils.data import DataLoader
from time import strftime
from copy import deepcopy
from skimage.measure import label as bwlabeln
from continual import EWC, GEM, AGEM, SGEM, NGEM
from utils import find_file, get_mask, get_normalised_image, get_bb
from utils import color_codes, time_to_string, remove_small_regions

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


def train(config, net, training, validation, model_name, verbose=0):
    """

    :param config:
    :param net:
    :param training:
    :param validation:
    :param model_name:
    :param verbose:
    """
    # Init
    path = config['model_path']
    epochs = config['epochs']
    patience = config['patience']

    try:
        net.load_model(os.path.join(path, model_name))
    except IOError:

        if verbose > 1:
            print('Preparing the training datasets / dataloaders')

        # Training
        if verbose > 1:
            print('< Training dataset >')
        dtrain, ltrain, rtrain = get_data(config, training)
        if 'train_patch' in config and 'train_overlap' in config:
            train_dataset = config['training'](
                dtrain, ltrain, rtrain, patch_size=config['train_patch'],
                overlap=config['train_overlap']
            )
        elif 'train_patch' in config:
            train_dataset = config['training'](
                dtrain, ltrain, rtrain, patch_size=config['train_patch']
            )
        else:
            train_dataset = config['training'](dtrain, ltrain, rtrain)

        if verbose > 1:
            print('Dataloader creation <with validation>')
        train_loader = DataLoader(
            train_dataset, config['train_batch'], True, num_workers=8
        )

        # Validation (training cases)
        if verbose > 1:
            print('< Validation dataset >')
        if training == validation:
            dval, lval, rval = dtrain, ltrain, rtrain
        else:
            dval, lval, rval = get_data(config, validation)
        if 'test_patch' in config and 'test_overlap' in config:
            val_dataset = config['validation'](
                dval, lval, rval, patch_size=config['test_patch'],
                overlap=config['train_overlap']
            )
        elif 'test_patch' in config:
            val_dataset = config['validation'](
                dval, lval, rval, patch_size=config['test_patch']
            )
        else:
            val_dataset = config['validation'](dval, lval, rval)

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


def test(
        config, seed, net, testing_results, testing_subjects,
        verbose=0
):
    # Init
    test_start = time.time()
    dataset = config['dataset'](
        testing_subjects[1], testing_subjects[2]
    )
    test_loader = DataLoader(
        dataset, config['test_batch'], num_workers=32
    )

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for batch_i, (x, y) in enumerate(test_loader):
        prediction = net.inference(x.cpu().numpy(), nonbatched=False) > 0.5
        target = y.cpu().numpy().astype(bool)
        no_target = np.logical_not(target)
        no_prediction = np.logical_not(prediction)

        tp += int(np.sum(np.logical_and(target, prediction)))
        tn += int(np.sum(np.logical_and(target, prediction)))
        fp += int(np.sum(np.logical_and(no_target, prediction)))
        fn += int(np.sum(np.logical_and(target, no_prediction)))

    results = {
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
    }
    for r_key, r_value in results.items():
        testing_results[subject][str(seed)][r_key].append(
            r_value
        )


def test_tasks(config, net, base_name, task_results, verbose=0):
    # Init
    options = parse_inputs()
    mask_base = os.path.splitext(os.path.basename(options['config']))[0]

    test_start = time.time()
    n_subjects = sum(len(task) for task in task_results)
    sub_i = 0
    for task_i, task_list in enumerate(task_results):
        mask_name = '{:}-{:}.tb{:02d}.nii.gz'.format(
            mask_base, base_name, task_i
        )
        type_dict = {
            'class': test_images_class,
            'seg': partial(test_images_seg, mask_name=mask_name),
            'patch': test_images_patch
        }
        type_tag = config['type']

        for subject in task_list.keys():
            tests = n_subjects - sub_i
            test_elapsed = time.time() - test_start
            test_eta = tests * test_elapsed / (sub_i + 1)
            if config['multisession']:
                sessions = list(task_list[subject].keys())
                for sess_j, session in enumerate(sessions):
                    if verbose:
                        print(
                            '\033[KTesting subject {:} [{:}] - '
                            'task {:02d}/{:02d} ({:d}/{:d} - {:d}/{:d}) '
                            '{:} ETA {:}'.format(
                                subject, session,
                                task_i, len(task_results),
                                sub_i + 1, n_subjects,
                                sess_j + 1, len(sessions),
                                time_to_string(test_elapsed),
                                time_to_string(test_eta),
                            ), end='\r'
                        )
                    results = type_dict[type_tag](
                        config=config, net=net, subject=subject, session=session
                    )
                    for r_key, r_value in results.items():
                        task_list[subject][session][r_key].append(
                            r_value
                        )
            else:
                if verbose > 0:
                    print(
                        '\033[KTesting subject {:} - '
                        'task {:02d}/{:02d} ({:d}/{:d}) '
                        '{:} ETA {:}'.format(
                            subject, task_i, len(task_results),
                            sub_i + 1, n_subjects,
                            time_to_string(test_elapsed),
                            time_to_string(test_eta),
                        ), end='\r'
                    )
                results = type_dict[type_tag](
                    config=config, net=net, subject=subject
                )
                for r_key, r_value in results.items():
                    task_list[subject][r_key].append(
                        r_value
                    )
            sub_i += 1


def empty_results_dict_seg():
    results_dict = {
        'TPV': [],
        'TNV': [],
        'FPV': [],
        'FNV': [],
        'TPR': [],
        'FPR': [],
        'FNR': [],
        'GTR': [],
        'R': [],
    }

    return results_dict


def empty_results_dict_class():
    results_dict = {
        'TP': [],
        'TN': [],
        'FP': [],
        'FN': [],
    }

    return results_dict


def empty_task_results(config, tasks):
    type_dict = {
        'class': empty_results_dict_class,
        'seg': empty_results_dict_seg,
        'patch': empty_results_dict_class
    }
    type_tag = config['type']
    if config['multisession']:
        results = [
            {
                subject['subject']: {
                    session: type_dict[type_tag]()
                    for session in subject['sessions']
                }
                for subject in task_list
            }
            for task_list in tasks
        ]
    else:
        results = [
            {
                subject: type_dict[type_tag]()
                for subject in task_list
            }
            for task_list in tasks
        ]

    return results


def empty_test_results(config, subjects):
    type_dict = {
        'class': empty_results_dict_class,
        'seg': empty_results_dict_seg,
        'patch': empty_results_dict_class
    }
    type_tag = config['type']
    seeds = config['seeds']
    if config['multisession']:
        results = {
            subject['subject']: {
                session: {
                    str(seed): type_dict[type_tag]()
                    for seed in seeds
                }
                for session in subject['sessions']
            }
            for t_list in subjects.values() for subject in t_list
        }
    else:
        results = {
            subject: {
                str(seed): type_dict[type_tag]()
                for seed in seeds
            }
            for t_list in subjects.values() for subject in t_list
        }

    return results


def get_test_results(
        config, seed, net, results, data
):
    path = config['json_path']
    json_file = find_file(json_name, path)
    if json_file is None:
        json_file = os.path.join(path, json_name)
        test(
            config, seed, net, results,
            data, verbose=1
        )

        with open(json_file, 'w') as testing_json:
            json.dump(results, testing_json)
    else:
        with open(json_file, 'r') as testing_json:
            results = json.load(testing_json)

    return results


def get_task_results(
        config, json_name, base_name, net, results
):
    path = config['json_path']
    json_file = find_file(json_name, path)
    if json_file is None:
        json_file = os.path.join(path, json_name)
        test_tasks(
            config, net, base_name, results, verbose=1
        )

        with open(json_file, 'w') as testing_json:
            json.dump(results, testing_json)
    else:
        with open(json_file, 'r') as testing_json:
            results = json.load(testing_json)

    return results


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
    n_folds = config['folds']
    val_split = config['val_split']
    model_path = config['model_path']
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    json_path = config['json_path']
    if not os.path.isdir(json_path):
        os.mkdir(json_path)
    model_base = os.path.splitext(os.path.basename(options['config']))[0]

    seeds = config['seeds']

    print(
        '{:}[{:}] {:}<Incremental learning framework>{:}'.format(
            c['c'], strftime("%H:%M:%S"), c['y'], c['nc']
        )
    )

    # We want a common starting point
    d_tr, d_te = load_datasets(config)

    # We prepare the dictionaries that will hold the relevant segmentation and
    # detection measures. That includes all positive combinations of positives
    # and negatives. Most relevant metrics like DSC come from there.
    baseline_testing = empty_test_results(config, d_te)
    naive_testing = empty_test_results(config, d_te)
    ewc_testing = empty_test_results(config, d_te)
    gem_testing = empty_test_results(config, d_te)
    agem_testing = empty_test_results(config, d_te)
    sgem_testing = empty_test_results(config, d_te)
    ngem_testing = empty_test_results(config, d_te)
    ind_testing = empty_test_results(config, d_te)
    init_testing = empty_test_results(config, d_te)

    # We also need dictionaries for the training tasks so we can track their
    # evolution. The main difference here, is that we need different
    # dictionaries for each task (or batch). These might be defined later and
    # we will fill these dictionaries accordingly when that happens.
    baseline_training = {
        str(seed): {
            'training': [],
            'validation': []
        }
        for seed in seeds
    }
    naive_training = deepcopy(baseline_training)
    ewc_training = deepcopy(baseline_training)
    gem_training = deepcopy(baseline_training)
    agem_training = deepcopy(baseline_training)
    sgem_training = deepcopy(baseline_training)
    ngem_training = deepcopy(baseline_training)
    ind_training = deepcopy(baseline_training)

    # Main loop with all the seeds
    for test_n, master_seed in enumerate(seeds):
        print(
            '{:}[{:}] {:}Starting cross-validation (model: {:}){:}'
            ' (seed {:d}){:}'.format(
                c['clr'] + c['c'], strftime("%H:%M:%S"), c['g'], model_base,
                c['nc'] + c['y'], master_seed, c['nc']
            )
        )
        # Network init (random weights)
        np.random.seed(master_seed)
        torch.manual_seed(master_seed)
        seed = np.random.randint(1e5)
        net = config['network'](n_outputs=d_tr[-1][0][-1])
        starting_model = os.path.join(
            model_path,
            '{:}-start.s{:05d}.pt'.format(model_base, master_seed)
        )
        net.save_model(starting_model)
        n_param = sum(
            p.numel() for p in net.parameters() if p.requires_grad
        )

        print(
            '{:}Testing initial weights{:} - {:02d}/{:02d} '
            '({:} parameters)'.format(
                c['clr'] + c['c'], c['nc'], test_n + 1, len(config['seeds']),
                c['b'] + str(n_param) + c['nc']
            )
        )
        # We will save the initial results pre-training
        # all_subjects = [p for t_list in subjects.values() for p in t_list]
        init_testing = get_test_results(
            config, master_seed, 'init', net,
            init_testing, d_te
        )

        # Training
        # Here we'll do the training / validation split...
        # Training and testing split
        training_validation = [
            [p for p in t['list'][t['end']:] + t['list'][:t['ini']]]
            for t in subjects_fold.values()
        ]
        if len(training_validation) == 1 or config['shuffling']:
            shuffled_subjects = np.array([
                sub for subs in training_validation for sub in subs
            ])
            training_validation = [
                array.tolist()
                for array in np.array_split(
                    shuffled_subjects,
                    len(shuffled_subjects) // config['task_size']
                )
            ]

        # We account for a validation set or the lack of it. The reason for
        # this is that we want to measure forgetting and that is easier to
        # measure if we only focus on the training set and leave the testing
        # set as an independent generalisation test.
        if val_split > 0:
            training_tasks = [
                [p for p in p_list[int(len(p_list) * val_split):]]
                for p_list in training_validation
            ]
            validation_tasks = [
                [p for p in p_list[:int(len(p_list) * val_split)]]
                for p_list in training_validation
            ]
            fold_val_baseline = empty_task_results(config, validation_tasks)
            fold_val_naive = empty_task_results(config, validation_tasks)
            fold_val_ewc = empty_task_results(config, validation_tasks)
            # fold_val_ewcplus = empty_task_results(config, validation_tasks)
        else:
            training_tasks = validation_tasks = training_validation
            fold_val_baseline = None
            fold_val_naive = None
            fold_val_ewc = None
            # fold_val_ewcplus = None
        tr_baseline = empty_task_results(config, training_tasks)
        tr_naive = empty_task_results(config, training_tasks)
        tr_ewc = empty_task_results(config, training_tasks)
        # fold_tr_ewcplus = empty_task_results(config, training_tasks)

        # Testing set for the current fold
        testing_set = [
            p for t in subjects_fold.values()
            for p in t['list'][t['ini']:t['end']]
        ]

        # Baseline model (full continuum access)
        net = config['network'](
            conv_filters=config['filters'],
            n_images=n_images
        )
        net.load_model(starting_model)

        # We test with the initial model to know the starting point for all
        # tasks
        tr_baseline = get_task_results(
            config, json_name, 'baseline-train.init', net,
            tr_baseline
        )
        fold_tr_naive = get_task_results(
            config, json_name, 'naive-train.init', net, fold_tr_naive
        )
        fold_tr_ewc = get_task_results(
            config, json_name, 'ewc-train.init', net, fold_tr_ewc
        )
        if fold_val_baseline is not None:
            json_name = '{:}-baseline-init_validation.s{:d}.json'.format(
                model_base, seed
            )
            fold_val_baseline = get_task_results(
                config, json_name, 'baseline-val.init', net,
                fold_val_baseline
            )
        if fold_val_naive is not None:
            json_name = '{:}-naive-init_validation.s{:d}.json'.format(
                model_base, seed
            )
            fold_val_naive = get_task_results(
                config, json_name, 'naive-val.init', net, fold_val_naive
            )
        if fold_val_ewc is not None:
            json_name = '{:}-ewc-init_validation.s{:d}.json'.format(
                model_base, seed
            )
            fold_val_ewc = get_task_results(
                config, json_name, 'ewc-val.init', net, fold_val_ewc
            )

        training_set = [
            p for p_list in training_tasks for p in p_list
        ]
        validation_set = [
            p for p_list in validation_tasks for p in p_list
        ]
        model_name = os.path.join(
            model_path,
            '{:}-bl.n{:d}.s{:05d}.pt'.format(
                model_base, i, seed
            )
        )
        print(
            '{:}Starting baseline fold {:} - {:02d}/{:02d} '
            '({:} parameters)'.format(
                c['clr'] + c['c'], c['g'] + str(i) + c['nc'],
                test_n + 1, len(config['seeds']),
                c['b'] + str(n_param) + c['nc']
            )
        )
        train(config, net, training_set, validation_set, model_name, 2)

        # Testing for the baseline. We want to reduce repeating the same
        # experiments to save time if the algorithm crashes.
        json_name = '{:}-baseline_testing.f{:d}.s{:d}.json'.format(
            model_base, i, seed
        )
        baseline_testing = get_test_results(
            config, seed, json_name, 'baseline', net,
            baseline_testing, testing_set
        )

        json_name = '{:}-baseline-training.f{:d}.s{:d}.json'.format(
            model_base, i, seed
        )
        fold_tr_baseline = get_task_results(
            config, json_name, 'baseline-train.f{:d}'.format(i), net,
            fold_tr_baseline
        )
        if fold_val_baseline is not None:
            json_name = '{:}-baseline-validation.f{:d}.s{:d}.json'.format(
                model_base, i, seed
            )
            fold_val_baseline = get_task_results(
                config, json_name, 'baseline-val.f{:d}'.format(i), net,
                fold_val_baseline
            )

        # Naive approach. We just partition the data and update the model
        # with each new batch without caring about previous samples
        net = config['network'](
            conv_filters=config['filters'],
            n_images=n_images
        )
        net.load_model(starting_model)

        # EWC approach. We use a penalty term / regularization loss
        # to ensure previous data isn't forgotten.
        try:
            ewc_weight = config['ewc_weight']
        except KeyError:
            ewc_weight = 1e6
        try:
            ewc_binary = config['ewc_binary']
        except KeyError:
            ewc_binary = True

        ewc_net = EWC(
            config['network'](
                conv_filters=config['filters'],
                n_images=n_images
            ),
            ewc_weight, ewc_binary
        )
        ewc_net.model.load_model(starting_model)

        # EWC approach. We use a penalty term / regularization loss
        # to ensure previous data isn't forgotten.
        # try:
        #     ewc_alpha = config['ewc_alpha']
        # except KeyError:
        #     ewc_alpha = 0.9
        #
        # ewcplus_net = models.MetaModel(
        #     config['network'](
        #         conv_filters=config['filters'],
        #         n_images=n_images
        #     ),
        #     ewc_weight, ewc_binary, ewc_alpha
        # )
        # ewcplus_net.model.load_model(starting_model)

        for ti, (training_set, validation_set) in enumerate(
                zip(training_tasks, validation_tasks)
        ):
            # < NAIVE >
            print(
                '{:}Starting task - naive {:02d} fold {:} - {:02d}/{:02d} '
                '({:} parameters)'.format(
                    c['clr'] + c['c'], ti + 1, c['g'] + str(i) + c['nc'],
                    test_n + 1, len(config['seeds']),
                    c['b'] + str(n_param) + c['nc']
                )
            )

            # We train the naive model on the current task
            model_name = os.path.join(
                model_path,
                '{:}-t{:02d}.n{:d}.s{:05d}.pt'.format(
                    model_base, ti, i, seed
                )
            )
            train(config, net, training_set, validation_set, model_name, 2)
            net.reset_optimiser()

            # Then we test it against all the datasets and tasks
            json_name = '{:}-naive_test.' \
                        'f{:d}.s{:d}.t{:02d}.json'.format(
                model_base, i, seed, ti
            )
            naive_testing = get_test_results(
                config, seed, json_name, 'naive-test.t{:02d}'.format(ti),
                net, naive_testing, testing_set
            )

            json_name = '{:}-naive-training.' \
                        'f{:d}.s{:d}.t{:02d}.json'.format(
                model_base, i, seed, ti
            )
            fold_tr_naive = get_task_results(
                config, json_name, 'naive-train.f{:d}.t{:02d}'.format(
                    i, ti
                ), net, fold_tr_naive
            )
            if fold_val_naive is not None:
                json_name = '{:}-naive-validation.' \
                            'f{:d}.s{:d}.t{:02d}.json'.format(
                    model_base, i, seed, ti
                )
                fold_val_naive = get_task_results(
                    config, json_name, 'naive-val.f{:d}.t{:02d}'.format(
                        i, ti
                    ),
                    net, fold_val_naive
                )

            # < EWC >
            print(
                '{:}Starting task - EWC {:02d} fold {:} - {:02d}/{:02d} '
                '({:} parameters)'.format(
                    c['clr'] + c['c'], ti + 1, c['g'] + str(i) + c['nc'],
                    test_n + 1, len(config['seeds']),
                    c['b'] + str(n_param) + c['nc']
                )
            )

            # We train the EWC model on the current task
            model_name = os.path.join(
                model_path,
                '{:}-ewc-t{:02d}.n{:d}.s{:05d}.pt'.format(
                    model_base, ti, i, seed
                )
            )
            train(
                config, ewc_net, training_set, validation_set,
                model_name, 2
            )
            ewc_net.reset_optimiser()

            # Then we test it against all the datasets and tasks
            json_name = '{:}-ewc_test.f{:d}.s{:d}.t{:02d}.json'.format(
                model_base, i, seed, ti
            )
            ewc_testing = get_test_results(
                config, seed, json_name, 'ewc-test.t{:02d}'.format(ti),
                ewc_net, ewc_testing, testing_set
            )

            json_name = '{:}-ewc-training.' \
                        'f{:d}.s{:d}.t{:02d}.json'.format(
                model_base, i, seed, ti
            )
            fold_tr_ewc = get_task_results(
                config, json_name, 'ewc-train.f{:d}.t{:02d}'.format(
                    i, ti
                ), ewc_net, fold_tr_ewc
            )
            if fold_val_naive is not None:
                json_name = '{:}-ewc-validation.' \
                            'f{:d}.s{:d}.t{:02d}.json'.format(
                    model_base, i, seed, ti
                )
                fold_val_ewc = get_task_results(
                    config, json_name, 'ewc-val.f{:d}.t{:02d}'.format(
                        i, ti
                    ),
                    ewc_net, fold_val_ewc
                )

            print(
                '{:}Starting task - EWC++ {:02d} fold {:} - {:02d}/{:02d} '
                '({:} parameters)'.format(
                    c['clr'] + c['c'], ti + 1, c['g'] + str(i) + c['nc'],
                    test_n + 1, len(config['seeds']),
                    c['b'] + str(n_param) + c['nc']
                )
            )

            # < EWC++ >
            # We train the EWC model on the current task
            # model_name = os.path.join(
            #     model_path,
            #     '{:}-ewcpp-t{:02d}.n{:d}.s{:05d}.pt'.format(
            #         model_base, ti, i, seed
            #     )
            # )
            # train(
            #     config, ewcplus_net, training_set, validation_set,
            #     model_name, 2
            # )
            # ewcplus_net.reset_optimiser()
            #
            # # Then we test it against all the datasets and tasks
            # json_name = '{:}-ewcpp_test.f{:d}.s{:d}.t{:02d}.json'.format(
            #     model_base, i, seed, ti
            # )
            # ewcplus_testing = get_test_results(
            #     config, seed, json_name, 'ewcpp-test.t{:02d}'.format(ti),
            #     ewcplus_net, ewcplus_testing, testing_set
            # )
            #
            # json_name = '{:}-ewcpp-training.' \
            #             'f{:d}.s{:d}.t{:02d}.json'.format(
            #                 model_base, i, seed, ti
            #             )
            # fold_tr_ewcplus = get_task_results(
            #     config, json_name, 'ewcpp-train.f{:d}.t{:02d}'.format(
            #         i, ti
            #     ), ewcplus_net, fold_tr_ewcplus
            # )
            # if fold_val_naive is not None:
            #     json_name = '{:}-ewcpp-validation.' \
            #                 'f{:d}.s{:d}.t{:02d}.json'.format(
            #                     model_base, i, seed, ti
            #                 )
            #     fold_val_ewcplus = get_task_results(
            #         config, json_name, 'ewcpp-val.f{:d}.t{:02d}'.format(
            #             i, ti
            #         ),
            #         ewcplus_net, fold_val_ewcplus
            #     )

        # Now it's time to push the results
        baseline_training[str(seed)]['training'].append(fold_tr_baseline)
        naive_training[str(seed)]['training'].append(fold_tr_naive)
        ewc_training[str(seed)]['training'].append(fold_tr_ewc)
        # ewcplus_training[str(seed)]['training'].append(fold_tr_ewcplus)
        if val_split > 0:
            baseline_training[str(seed)]['validation'].append(
                fold_val_baseline
            )
            naive_training[str(seed)]['validation'].append(
                fold_val_naive
            )
            ewc_training[str(seed)]['validation'].append(
                fold_val_ewc
            )
            # ewcplus_training[str(seed)]['validation'].append(
            #     fold_val_ewcplus
            # )

    save_results(
        config, '{:}-baseline_testing.json'.format(model_base),
        baseline_testing
    )
    save_results(
        config, '{:}-baseline_training.json'.format(model_base),
        baseline_training
    )
    save_results(
        config, '{:}-naive_testing.json'.format(model_base),
        naive_testing
    )
    save_results(
        config, '{:}-naive_training.json'.format(model_base),
        naive_training
    )
    save_results(
        config, '{:}-ewc_testing.json'.format(model_base),
        ewc_testing
    )
    save_results(
        config, '{:}-ewc_training.json'.format(model_base),
        ewc_training
    )
    # save_results(
    #     config, '{:}-ewcpp_testing.json'.format(model_base),
    #     ewcplus_testing
    # )
    # save_results(
    #     config, '{:}-ewcpp_training.json'.format(model_base),
    #     ewcplus_training
    # )


if __name__ == '__main__':
    main()
