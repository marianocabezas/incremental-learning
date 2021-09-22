import argparse
import os
import json
import nibabel
import numpy as np
import time
import datasets
import models
import yaml
import torch
from torch.utils.data import DataLoader
from time import strftime
from copy import deepcopy
from skimage.measure import label as bwlabeln
from utils import find_file, get_mask, get_normalised_image, get_bb
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


def get_images(experiment_config, subject, session=None):
    d_path = experiment_config['path']
    p_path = os.path.join(d_path, subject)
    if session is not None:
        p_path = os.path.join(p_path, session)
    roi = get_mask(find_file(experiment_config['roi'], p_path))
    label = get_mask(find_file(experiment_config['labels'], p_path))
    if isinstance(experiment_config['files'], tuple):
        images = tuple(
            load_image_list(p_path, file_i, roi)
            for file_i in experiment_config['files']
        )
    else:
        images = load_image_list(
            p_path, experiment_config['files'], roi
        )
    return roi, label, images


def get_data(experiment_config, subject_list):
    load_start = time.time()

    subjects = []
    labels = []
    rois = []
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
                roi, label, images = get_images(
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
            roi, label, images = get_images(
                experiment_config, p['subject']
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


def test_images(config, mask_name, net, subject, session=None):
    masks_path = config['masks_path']
    d_path = os.path.join(config['path'], subject)
    p_path = os.path.join(masks_path, subject)
    if not os.path.isdir(p_path):
        os.mkdir(p_path)
    if session is not None:
        p_path = os.path.join(p_path, session)
        d_path = os.path.join(d_path, session)
        if not os.path.isdir(p_path):
            os.mkdir(p_path)

    roi, label, images = get_images(
        config, subject, session
    )
    bb = get_bb(roi, 2)

    prediction_file = find_file(mask_name, p_path)
    if prediction_file is None:
        prediction_file = os.path.join(p_path, mask_name)
        segmentation = np.zeros_like(label)
        none_slice = (slice(None, None),)
        # Patch "extraction".
        if isinstance(images, tuple):
            data = tuple(
                data_i[none_slice + bb].astype(np.float32)
                for data_i in images
            )
        else:
            data = images[none_slice + bb].astype(np.float32)
        prediction = net.inference(data) > 0.5
        segmentation[bb] = prediction
        segmentation[np.logical_not(roi)] = 0
        ref_nii = nibabel.load(os.path.join(d_path, config['labels']))
        segmentation_nii = nibabel.Nifti1Image(
            segmentation, ref_nii.get_qform(), ref_nii.header
        )
        segmentation_nii.to_filename(prediction_file)
    else:
        segmentation = nibabel.load(prediction_file).get_fdata()
        prediction = segmentation[bb].astype(bool)

    target = label[bb].astype(bool)
    no_target = np.logical_not(target)
    target_regions, gtr = bwlabeln(target, return_num=True)
    no_prediction = np.logical_not(prediction)
    prediction_regions, r = bwlabeln(prediction, return_num=True)
    true_positive = np.logical_and(target, prediction)
    false_positive = np.logical_and(no_target, prediction)

    results = {
        'TPV': np.sum(true_positive),
        'TNV': np.sum(np.logical_and(no_target, no_prediction)),
        'FPV': np.sum(false_positive),
        'FNV': np.sum(np.logical_and(target, np.logical_not(prediction))),
        'TPR': len(np.unique(target_regions[true_positive])),
        'FPR': len(np.unique(prediction_regions[false_positive])),
        'GTR': gtr,
        'R': r
    }
    return results


def test(config, seed, net, testing, training, validation=None, verbose=0):
    # Init
    options = parse_inputs()
    mask_base = os.path.splitext(os.path.basename(options['config']))[0]

    testing_subjects = list(testing.keys())
    test_start = time.time()
    mask_name = '{:}-test.s{:05d}.nii.gz'.format(mask_base, seed)
    for sub_i, subject in enumerate(testing_subjects):
        tests = len(testing_subjects) - sub_i
        test_elapsed = time.time() - test_start
        test_eta = tests * test_elapsed / (sub_i + 1)
        if config['multisession']:
            sessions = list(testing[subject].keys())
            for sess_j, session in enumerate(sessions):
                if verbose:
                    print(
                        '\033[KTesting subject {:} [{:}]'
                        ' ({:d}/{:d} - {:d}/{:d}) {:} ETA {:}'.format(
                            subject, session, sub_i + 1, len(testing_subjects),
                            sess_j + 1, len(sessions),
                            time_to_string(test_elapsed),
                            time_to_string(test_eta),
                        ), end='\r'
                    )
                results = test_images(config, mask_name, net, subject, session)
                for r_key, r_value in results.items():
                    testing[subject][session][str(seed)][r_key].append(r_value)
        else:
            if verbose > 0:
                print(
                    '\033[KTesting subject {:} ({:d}/{:d}) {:} ETA {:}'.format(
                        subject, sub_i + 1, len(testing_subjects),
                        time_to_string(test_elapsed),
                        time_to_string(test_eta),
                    ), end='\r'
                )
            results = test_images(config, mask_name, net, subject)
            for r_key, r_value in results.items():
                testing[subject][str(seed)][r_key].append(r_value)

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
    if config['multisession']:
        baseline_testing = {
            subject['subject']: {
                session: {
                    str(seed): {
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
                    for seed in seeds
                }
            }
            for t_list in subjects.values() for subject in t_list
            for session in subject['sessions']
        }
    else:
        baseline_testing = {
            subject: {
                str(seed): {
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
                for seed in seeds
            }
            for t_list in subjects.values() for subject in t_list
        }
    naive_testing = deepcopy(baseline_testing)

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
                    config, seed, net, baseline_testing, training_tasks,
                    validation_tasks, verbose=1
                )
            else:
                test(
                    config, seed, net, baseline_testing, training_tasks,
                    verbose=1
                )

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
                if val_split > 0:
                    test(
                        config, seed, net, naive_testing, training_tasks,
                        validation_tasks, verbose=1
                    )
                else:
                    test(
                        config, seed, net, naive_testing, training_tasks,
                        verbose=1
                    )
            json_name = '{:}-baseline_testing.f{:d}.s{:d}.jsom'.format(
                model_base, i, seed
            )
            with open(os.path.join(masks_path, json_name), 'w') as testing_json:
                json.dump(baseline_testing, testing_json)
            json_name = '{:}-naive_testing.f{:d}.s{:d}.jsom'.format(
                model_base, i, seed
            )
            with open(os.path.join(masks_path, json_name), 'w') as testing_json:
                json.dump(naive_testing, testing_json)

    json_name = '{:}-baseline_testing.jsom'.format(model_base)
    with open(os.path.join(masks_path, json_name), 'w') as testing_json:
        json.dump(baseline_testing, testing_json)
    json_name = '{:}-naive_testing.jsom'.format(model_base)
    with open(os.path.join(masks_path, json_name), 'w') as testing_json:
        json.dump(naive_testing, testing_json)

    # Read data from file:
    # data = json.load(open("file_name.json"))


if __name__ == '__main__':
    main()
