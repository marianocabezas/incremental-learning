import argparse
import os
import time
import yaml
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from time import strftime
import datasets
import models
from medical_comparison import get_images_class, get_images_seg
from utils import find_file, get_mask, get_normalised_image
from utils import color_codes, time_to_string

"""
> Arguments
"""


def parse_inputs():
    parser = argparse.ArgumentParser(
        description='Train a model and then cluster the images according to '
                    'their model-based features.'
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
    # Init
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

    # Subject listing
    subjects = sorted([
        patient for patient in os.listdir(d_path)
        if os.path.isdir(os.path.join(d_path, patient))
    ])
    subject_dicts = {
        task: [] for task in tasks
    }
    # Task sorting
    for pi, p in enumerate(subjects):
        if multitask:
            task_found = False
            for task in tasks:
                task_found = task in p
                if task_found:
                    break
        else:
            task_found = True

        if task_found:
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
        get_normalised_image(find_file(image, path), roi)
        for image in image_list
    ]

    return np.stack(images).astype(np.float16)


def get_data(experiment_config, subject_list):
    # Init
    load_start = time.time()
    type_dict = {
        'class': get_images_class,
        'seg': get_images_seg,
        'patch': get_images_seg
    }
    type_tag = experiment_config['type']
    subjects = []
    labels = []
    rois = []

    # Loading
    for pi, p in enumerate(subject_list):
        loads = len(subject_list) - pi
        load_elapsed = time.time() - load_start
        load_eta = loads * load_elapsed / (pi + 1)
        if experiment_config['multisession']:
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
                roi, label, images = type_dict[type_tag](
                    experiment_config, p['subject'], session
                )
                rois.append(roi)
                labels.append(label)
                subjects.append(images)
        else:
            print(
                '\033[KLoading subject {:} ({:d}/{:d}) '
                '{:} ETA {:}'.format(
                    p, pi + 1, len(subject_list),
                    time_to_string(load_elapsed),
                    time_to_string(load_eta),
                ), end='\r'
            )
            roi, label, images = type_dict[type_tag](
                experiment_config, p
            )
            rois.append(roi)
            labels.append(label)
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
    path = config['model_path']
    epochs = 10
    # epochs = config['epochs']
    patience = epochs
    # patience = config['patience']

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

    print(
        '{:}[{:}] {:}<Data clustering framework>{:}'.format(
            c['c'], strftime("%H:%M:%S"), c['y'], c['nc']
        )
    )

    # We want a common starting point
    subjects = get_subjects(config)

    if isinstance(config['files'], tuple):
        n_images = len(config['files'][0])
    elif isinstance(config['files'], list):
        n_images = len(config['files'])
    else:
        n_images = 1

    # Main loop with all the seeds
    for test_n, seed in enumerate(seeds):
        print(
            '{:}[{:}] {:}Starting training (model: {:}){:}'
            ' (seed {:d}){:}'.format(
                c['clr'] + c['c'], strftime("%H:%M:%S"), c['g'], model_base,
                c['nc'] + c['y'], seed, c['nc']
            )
        )
        # Network init (random weights)
        np.random.seed(seed)
        torch.manual_seed(seed)
        net = config['network'](
            conv_filters=config['filters'],
            n_images=n_images
        )
        n_param = sum(
            p.numel() for p in net.parameters() if p.requires_grad
        )
        # We will save the initial results pre-training
        shuffled_subjects = [
            [
                p for p in np.random.permutation([
                    sub for sub in t_list
                ]).tolist()
            ]
            for t_list in subjects.values()
        ]

        # Training
        val_split = 0.25

        # We account for a validation set or the lack of it. The reason for
        # this is that we want to measure forgetting and that is easier to
        # measure if we only focus on the training set and leave the testing
        # set as an independent generalisation test.
        training_set = [
            p for p_list in shuffled_subjects
            for p in p_list[int(len(p_list) * val_split):]
        ]
        validation_set = [
            p for p_list in shuffled_subjects
            for p in p_list[:int(len(p_list) * val_split)]
        ]

        # Model with good image features
        net = config['network'](
            conv_filters=config['filters'],
            n_images=n_images
        )
        model_name = os.path.join(
            model_path,
            '{:}-data.s{:05d}.pt'.format(
                model_base, seed
            )
        )
        print(
            '{:}Starting data embedding{:} - {:02d}/{:02d} '
            '({:} parameters)'.format(
                c['clr'] + c['c'], c['nc'], test_n + 1, len(config['seeds']),
                c['b'] + str(n_param) + c['nc']
            )
        )
        train(config, net, training_set, validation_set, model_name, 2)

        all_set = [
            p for p_list in shuffled_subjects for p in p_list
        ]


if __name__ == '__main__':
    main()
