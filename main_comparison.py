import argparse
import os
import numpy as np
import nibabel as nib
import time
import datasets
import models
import yaml
import torch
from torch.utils.data import DataLoader
from time import strftime
from utils import find_file, get_mask, get_normalised_image
from utils import color_codes, time_to_string


"""
> Arguments
"""


def parse_inputs():
    parser = argparse.ArgumentParser(
        description='Do cross-calidation of the ALS/healthy subjects using '
                    'pre-training, attention gates and other stuff.'
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


def get_subjects(experiment_config):
    d_path = experiment_config['path']
    multitask = False
    task = 'Continuum'
    tasks = [task]
    try:
        if experiment_config['tasks'] is not None:
            multitask = True
            tasks = experiment_config['tasks']
    except KeyError:
        pass

    subjects = sorted([
        patient for patient in os.listdir(d_path)
        if os.path.isdir(os.path.join(d_path, patient))
    ])
    subject_dicts = {
        task: [] for task in tasks
    }
    for pi, p in enumerate(subjects):
        if multitask:
            for task in tasks:
                if task in p:
                    break
        p_path = os.path.join(d_path, p)
        if experiment_config['multisession']:
            sessions = [
                session for session in os.listdir(p_path)
                if os.path.isdir(os.path.join(p_path, session))
            ]
            patient_dict = {
                'subject': p,
                'sessions': sessions
            }
            subject_dicts[task].append(patient_dict)
        else:
            subject_dicts[task].append(p)
    return subject_dicts


def load_image_list(path, image_list, roi):
    images = [
        get_normalised_image(os.path.join(path, image), roi, masked=True)
        for image in image_list
    ]

    return np.stack(images).astype(np.float16)


def get_data(experiment_config, subject_list):
    d_path = experiment_config['path']
    load_start = time.time()

    subjects = []
    labels = []
    rois = []
    for pi, p in enumerate(subject_list):
        loads = len(subject_list) - pi
        load_elapsed = time.time() - load_start
        load_eta = loads * load_elapsed / (pi + 1)
        if experiment_config['multisession']:
            p_path = os.path.join(d_path, p['subject'])
            sessions = p['sessions']
            for si, session in enumerate(sessions):
                print(
                    '\033[KLoading subject {:} [{:}] ({:d}/{:d} - {:d}/{:d}) '
                    '{:} ETA {:}'.format(
                        p['subject'], session, pi + 1, len(subject_list),
                        si + 1, len(sessions),
                        time_to_string(load_elapsed),
                        time_to_string(load_eta),
                    ), end='\r'
                )
                s_path = os.path.join(p_path, session)
                roi = get_mask(find_file(experiment_config['roi'], s_path))
                rois.append(roi)
                labels.append(
                    get_mask(find_file(experiment_config['labels'], s_path))
                )
                if isinstance(experiment_config['files'], tuple):
                    images = tuple(
                        load_image_list(s_path, file_i, roi)
                        for file_i in experiment_config['files']
                    )
                else:
                    images = load_image_list(
                        s_path, experiment_config['files'], roi
                    )
                subjects.append(images)
        else:
            p_path = os.path.join(d_path, p)
            print(
                '\033[KLoading subject {:} ({:d}/{:d}) '
                '{:} ETA {:}'.format(
                    p, pi + 1, len(subject_list),
                    time_to_string(load_elapsed),
                    time_to_string(load_eta),
                ), end='\r'
            )
            roi = get_mask(find_file(experiment_config['roi'], p_path))
            rois.append(roi)
            labels.append(
                get_mask(find_file(experiment_config['labels'], p_path))
            )
            if isinstance(experiment_config['files'], tuple):
                images = tuple(
                    load_image_list(p_path, file_i, roi)
                    for file_i in experiment_config['files']
                )
            else:
                images = load_image_list(
                    p_path, experiment_config['files'], roi
                )
            subjects.append(images)
    print('\033[K', end='\r')
    return subjects, labels, rois


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
    c = color_codes()
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
                dtrain, ltrain, rtrain, patch_size=config['train_batch'],
                overlap=config['train_overlap']
            )
        elif 'train_patch' in config:
            train_dataset = config['training'](
                dtrain, ltrain, rtrain , patch_size=config['train_batch']
            )
        else:
            train_dataset = config['training'](dtrain, ltrain, rtrain)

        if verbose > 1:
            print('Dataloader creation <with validation>')
        train_loader = DataLoader(
            train_dataset, config['train_batch'], True, num_workers=32
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
                dval, lval, rval, patch_size=config['train_batch'],
                overlap=config['train_overlap']
            )
        elif 'test_patch' in config:
            val_dataset = config['validation'](
                dval, lval, rval, patch_size=config['train_batch']
            )
        else:
            val_dataset = config['validation'](dval, lval, rval)

        if verbose > 1:
            print('Dataloader creation <val>')
        val_loader = DataLoader(
            val_dataset, config['test_batch'], num_workers=32
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


def test(config, seed, net, testing, training, validation=None, verbose=0):
    # Init
    c = color_codes()
    options = parse_inputs()
    masks_path = config['masks_path']
    mask_base = os.path.splitext(os.path.basename(options['config']))[0]

    print(testing)
    print(training)
    if validation is not None:
        print(validation)


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
    masks_path = config['masks_path']
    if not os.path.isdir(masks_path):
        os.mkdir(masks_path)
    model_base = os.path.splitext(os.path.basename(options['config']))[0]

    seeds = config['seeds']

    print(
        '{:}[{:}] {:}<Incremental learning framework>{:}'.format(
            c['c'], strftime("%H:%M:%S"), c['y'], c['nc']
        )
    )

    # We want a common starting point
    subjects = get_subjects(config)
    testing_results = {
        subject: {
            str(seed): {
                'TPV': [],
                'TNV': [],
                'FPV': [],
                'FNV': [],
                'TPR': [],
                'FPR': [],
                'FNR': [],
            }
            for seed in seeds
        }
        for subject in subjects
    }
    if isinstance(config['files'], tuple):
        n_images = len(config['files'][0])
    else:
        n_images = len(config['files'])

    for test_n, seed in enumerate(seeds):
        print(
            '{:}[{:}] {:}Starting cross-validation (model: {:}){:}'
            ' (seed {:d}){:}'.format(
                c['c'], strftime("%H:%M:%S"), c['g'], model_base,
                c['nc'] + c['y'], seed, c['nc']
            )
        )
        np.random.seed(seed)
        torch.manual_seed(seed)
        net = config['network'](
            conv_filters=config['filters'],
            n_images=n_images
        )
        starting_model = os.path.join(
            model_path,
            '{:}-start.s{:05d}.pt'.format(model_base, seed)
        )
        net.save_model(starting_model)
        n_param = sum(
            p.numel() for p in net.parameters() if p.requires_grad
        )
        for i in range(n_folds):
            subjects_fold = {
                t_key: {
                    'list': t_list,
                    'ini': len(t_list) * i // n_folds,
                    'end': len(t_list) * (i + 1) // n_folds
                }
                for t_key, t_list in subjects.items()
            }
            # Training
            # Here we'll do the training / validation / testing split...
            # Training and testing split
            training_validation = [
                [p for p in t['list'][t['end']:] + t['list'][:t['ini']]]
                for t in subjects_fold.values()
            ]
            if len(training_validation) == 1:
                shuffled_subjects = np.random.permutation(
                    training_validation[0]
                )
                training_validation = [
                    array.tolist()
                    for array in np.array_split(
                        shuffled_subjects,
                        len(shuffled_subjects) // config['task_size']
                    )
                ]
            if val_split > 0:
                training_tasks = [
                    [p for p in p_list[int(len(p_list) * val_split):]]
                    for p_list in training_validation
                ]
                validation_tasks = [
                    [p for p in p_list[:int(len(p_list) * val_split)]]
                    for p_list in training_validation
                ]
            else:
                training_tasks = validation_tasks = training_validation
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
                    c['c'], c['g'] + str(i) + c['nc'],
                    test_n + 1, len(config['seeds']),
                    c['b'] + str(n_param) + c['nc']
                )
            )
            train(config, net, training_set, validation_set, model_name, 2)

            net = config['network'](
                conv_filters=config['filters'],
                n_images=n_images
            )
            net.load_model(starting_model)

            if val_split > 0:
                test(
                    config, seed, net, testing_results, training_tasks,
                    validation_tasks
                )
            else:
                test(config, seed, net, testing_results, training_tasks)

            for ti, (training_set, validation_set) in enumerate(
                zip(training_tasks, validation_tasks)
            ):
                print(
                    '{:}Starting task {:02d} fold {:} - {:02d}/{:02d} '
                    '({:} parameters)'.format(
                        c['c'], ti + 1, c['g'] + str(i) + c['nc'],
                        test_n + 1, len(config['seeds']),
                        c['b'] + str(n_param) + c['nc']
                    )
                )
                model_name = os.path.join(
                    model_path,
                    '{:}-t{:02d}.n{:d}.s{:05d}.pt'.format(
                        model_base, ti, i, seed
                    )
                )
                train(config, net, training_set, validation_set, model_name, 2)
                net = config['network'](
                    conv_filters=config['filters'],
                    n_images=n_images
                )
                net.load_model(model_name)


if __name__ == '__main__':
    main()
