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

    subjects = [
        patient for patient in os.listdir(d_path)
        if os.path.isdir(os.path.join(d_path, patient))
    ]
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
        get_normalised_image(os.path.join(path, image), roi)
        for image in image_list
    ]

    return np.stack(images)


def get_data(experiment_config, subject_list):
    d_path = experiment_config['path']
    load_start = time.time()

    subjects = []
    labels = []
    rois = []
    for pi, p in enumerate(subject_list):
        p_path = os.path.join(d_path, p)
        loads = len(subject_list) - pi
        load_elapsed = time.time() - load_start
        load_eta = loads * load_elapsed / (pi + 1)
        if experiment_config['multisession']:
            sessions = [
                session for session in os.listdir(p_path)
                if os.path.isdir(os.path.join(p_path, session))
            ]
            for si, session in enumerate(sessions):
                print(
                    'Loading subject {:} [{:}] ({:d}/{:d} - {:d}/{:d}) '
                    '{:} ETA {:}'.format(
                        p, session, pi + 1, len(subject_list),
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
            print(
                'Loading subject {:} ({:d}/{:d}) '
                '{:} ETA {:}'.format(
                    p, pi + 1, len(subjects),
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
    path = config['output_path']
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
            val_dataset, config['batch_test'], True, num_workers=32
        )

        net.fit(
            train_loader, val_loader, epochs=epochs, patience=patience
        )
        net.save_model(os.path.join(path, model_name))


# def test(net, testing, path, model, verbose=0):
#     c = color_codes()
#     options = parse_inputs()
#     labels = options['labels']
#     confusion_matrix = np.array([[0, 0], [0, 0]])
#
#     for subject in testing:
#         if verbose > 1:
#             print(
#                 '{:}Testing subject {:^24} [{:}] - {:}'.format(
#                     c['c'], c['g'] + subject['name'] + c['nc'],
#                     c['y'] + subject['class'] + c['nc'],
#                     c['lgy'] + model + c['nc']
#                 ), end='\r'
#             )
#         p_path = os.path.join(path, subject['class'].upper(), subject['name'])
#         o_path = os.path.join(p_path, 'Predictions')
#         if not os.path.exists(o_path):
#             os.mkdir(o_path)
#
#         pr = net.classify(
#             np.expand_dims(subject['dwi'], axis=0),
#             np.expand_dims(subject['brain'], axis=0)
#         )
#
#         true_label = subject['label']
#         pred_label = int(pr > 0.5)
#         label_name = labels[pred_label]
#         true_name = labels[true_label]
#
#         if verbose > 1:
#             tp_c = c['g'] if true_label == pred_label else c['r']
#             print(
#                 '{:}Testing subject (CEL) {:^24} [{:} - {:}] - {:}'.format(
#                     c['c'], c['g'] + subject['name'] + c['nc'],
#                     c['y'] + subject['class'] + c['nc'],
#                     tp_c + label_name + c['nc'],
#                     c['lgy'] + model + c['nc']
#                 )
#             )
#
#         confusion_matrix[true_label, pred_label] += 1
#
#         # We need to clean previous files (labels might have changed)
#         for file in os.listdir(o_path):
#             if file.startswith(model) and file.endswith('.txt'):
#                 os.remove(os.path.join(o_path, file))
#
#         pr_name = os.path.join(o_path, '{:}_{:}-{:}.txt'.format(
#             model, label_name, true_name
#         ))
#
#         with open(pr_name, 'w') as f:
#             if pred_label:
#                 f.write(str(pr))
#             else:
#                 f.write(str(1 - pr))
#
#     n = np.sum(confusion_matrix)
#     n_ctl = np.sum(confusion_matrix[0, :])
#     n_als = np.sum(confusion_matrix[1, :])
#     tp_ctl = 100 * confusion_matrix[0, 0]
#     tp_als = 100 * confusion_matrix[1, 1]
#     intersection = 100 * (confusion_matrix[0, 0] + confusion_matrix[1, 1])
#     tpf_ctl = tp_ctl / n_ctl
#     tpf_als = tp_als / n_als
#
#     print('TPF (CTL) =', tpf_ctl)
#     print('TPF (ALS) =', tpf_als)
#     print('Accuracy =', intersection / n)
#     print('Balanced accuracy =', (tpf_ctl + tpf_als) / 2)


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

    print(
        '{:}[{:}] {:}<Incremental learning framework>{:}'.format(
            c['c'], strftime("%H:%M:%S"), c['y'], c['nc']
        )
    )

    # We want a common starting point
    subjects = get_subjects(config)
    if isinstance(config['files'], tuple):
        n_images = len(config['files'][0])
    else:
        n_images = len(config['files'])
    net = config['network'](
        conv_filters=config['filters'],
        n_images=n_images
    )
    starting_model = os.path.join(
        config['output_path'],
        '{:}-start.pt'.format(config['model_name'])
    )
    net.save_model(starting_model)

    for test_n, seed in enumerate(config['seeds']):
        print(
            '{:}[{:}] {:}Starting cross-validation{:} (seed {:d}){:}'.format(
                c['c'], strftime("%H:%M:%S"), c['g'], c['nc'] + c['y'],
                seed, c['nc']
            )
        )
        np.random.seed(seed)
        torch.manual_seed(seed)
        # model_name = 'presnet.{:02d}-e{:d}.n{:d}.pt'.format(
        #     t, epochs, i
        # )
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
            testing_set = [
                p for t in subjects_fold.values()
                for p in t['list'][t['ini']:t['end']]
            ]

            # Baseline model (full continuum access)
            net.load_model(starting_model)
            training_set = [
                p for p_list in training_validation
                for p in p_list[int(len(p_list) * val_split):]
            ]
            validation_set = [
                p for p_list in training_validation
                for p in p_list[:int(len(p_list) * val_split)]
            ]
            model_name = os.path.join(
                config['output_path'],
                '{:}-bl.n{:d}.pt'.format(
                    config['model_name'], i
                )
            )
            train(config, net, training_set, validation_set, model_name, 2)


if __name__ == '__main__':
    main()
