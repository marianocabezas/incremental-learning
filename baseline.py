import argparse
import os
import time
import importlib
import nibabel
import numpy as np
import pandas as pd
import yaml
import torch
from functools import partial
from torch.utils.data import DataLoader
from time import strftime
from utils import find_file, get_mask, get_normalised_image, get_bb
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


def load_image_list(path, image_list, roi, masked):
    images = [
        get_normalised_image(find_file(image, path), roi, masked=masked)
        for image in image_list
    ]

    return np.stack(images).astype(np.float16)


def get_images_seg(experiment_config, subject, session=None):
    # Init
    d_path = experiment_config['path']
    p_path = os.path.join(d_path, subject)
    if session is not None:
        p_path = os.path.join(p_path, session)

    # Data loading
    roi = get_mask(find_file(experiment_config['roi'], p_path))
    label = get_mask(find_file(experiment_config['labels'], p_path))
    if isinstance(experiment_config['files'], tuple):
        images = tuple(
            load_image_list(p_path, file_i, roi, False)
            for file_i in experiment_config['files']
        )
    elif isinstance(experiment_config['files'], list):
        images = load_image_list(
            p_path, experiment_config['files'], roi, False
        )
    else:
        images = load_image_list(
            p_path, [experiment_config['files']], roi, False
        )
    return roi, label, images


def get_images_class(experiment_config, subject, session=None):
    # Init
    d_path = experiment_config['path']
    p_path = os.path.join(d_path, subject)

    # Data loading
    if session is not None:
        p_path = os.path.join(p_path, session)
    roi = get_mask(find_file(experiment_config['roi'], p_path))

    label_csv = os.path.join(d_path, experiment_config['labels'])
    dx_df = pd.read_csv(label_csv)
    label_dict = dx_df.set_index(dx_df.columns[0])[dx_df.columns[1]].to_dict()
    label = label_dict[subject]
    if isinstance(experiment_config['files'], tuple):
        images = tuple(
            load_image_list(p_path, file_i, roi, True)
            for file_i in experiment_config['files']
        )
    elif isinstance(experiment_config['files'], list):
        images = load_image_list(
            p_path, experiment_config['files'], roi, True
        )
    else:
        images = load_image_list(
            p_path, [experiment_config['files']], roi, True
        )
    return roi, label, images


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
    epochs = config['epochs']
    patience = config['patience']

    try:
        net.load_model(os.path.join(path, model_name))
    except IOError:

        if verbose > 1:
            print('Preparing the training datasets / dataloaders')

        datasets = importlib.import_module('datasets')
        training_class = getattr(datasets, config['training'])
        validation_class = getattr(datasets, config['validation'])

        # Training
        if verbose > 1:
            print('< Training dataset >')
        dtrain, ltrain, rtrain = get_data(config, training)
        if 'train_patch' in config and 'train_overlap' in config:
            train_dataset = training_class(
                dtrain, ltrain, rtrain, patch_size=config['train_patch'],
                overlap=config['train_overlap']
            )
        elif 'train_patch' in config:
            train_dataset = training_class(
                dtrain, ltrain, rtrain, patch_size=config['train_patch']
            )
        else:
            train_dataset =training_class(dtrain, ltrain, rtrain)

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
            val_dataset = validation_class(
                dval, lval, rval, patch_size=config['test_patch'],
                overlap=config['train_overlap']
            )
        elif 'test_patch' in config:
            val_dataset = validation_class(
                dval, lval, rval, patch_size=config['test_patch']
            )
        else:
            val_dataset = validation_class(dval, lval, rval)

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


def test_images_seg(
    config, mask_name, net, subject, session=None, case=0, n_cases=1,
    t_start=time.time()
):
    masks_path = config['masks_path']
    if not os.path.isdir(masks_path):
        os.mkdir(masks_path)
    d_path = os.path.join(config['path'], subject)
    p_path = os.path.join(masks_path, subject)
    if not os.path.isdir(p_path):
        os.mkdir(p_path)
    if session is not None:
        p_path = os.path.join(p_path, session)
        d_path = os.path.join(d_path, session)
        if not os.path.isdir(p_path):
            os.mkdir(p_path)

    prediction_file = find_file(mask_name, p_path)
    if prediction_file is None:

        roi, label, images = get_images_seg(
            config, subject, session
        )
        bb = get_bb(roi, 2)

        prediction_file = os.path.join(p_path, mask_name)
        segmentation = np.zeros_like(label)
        none_slice = (slice(None, None),)

        if isinstance(images, tuple):
            data = tuple(
                data_i[none_slice + bb].astype(np.float32)
                for data_i in images
            )
        else:
            data = images[none_slice + bb].astype(np.float32)

        try:
            prediction = net.inference(data)
        except RuntimeError:
            patch_size = config['test_patch']
            batch_size = config['test_batch']
            prediction = net.patch_inference(
                data, patch_size, batch_size, case=case, n_cases=n_cases,
                t_start=t_start
            )
        segmentation[bb] = prediction
        segmentation[np.logical_not(roi)] = 0

        ref_nii = nibabel.load(find_file(config['labels'], d_path))
        segmentation_nii = nibabel.Nifti1Image(
            segmentation, ref_nii.get_qform(), ref_nii.header
        )
        segmentation_nii.to_filename(prediction_file)


def test(
    config, seed, net, base_name, testing_subjects, verbose=0
):
    # Init
    options = parse_inputs()
    mask_base = os.path.splitext(os.path.basename(options['config']))[0]
    mask_name = '{:}-{:}.s{:05d}.nii.gz'.format(
        mask_base, base_name, seed
    )
    type_dict = {
        'class': None,
        'seg': partial(test_images_seg, mask_name=mask_name),
        'patch': None
    }
    type_tag = config['type']

    test_start = time.time()
    for sub_i, subject in enumerate(testing_subjects):
        tests = len(testing_subjects) - sub_i
        test_elapsed = time.time() - test_start
        test_eta = tests * test_elapsed / (sub_i + 1)
        if config['multisession']:
            sessions = subject['sessions']
            subject = subject['subject']
            for sess_j, session in enumerate(sessions):
                if verbose:
                    print(
                        '\033[KTesting subject {:} [{:}]'
                        ' ({:d}/{:d} - {:d}/{:d}) {:} ETA {:}'.format(
                            subject, session,
                            sub_i + 1, len(testing_subjects),
                            sess_j + 1, len(sessions),
                            time_to_string(test_elapsed),
                            time_to_string(test_eta),
                        ), end='\r'
                    )
                if type_dict[type_tag] is not None:
                    type_dict[type_tag](
                        config=config, net=net, subject=subject,
                        session=session, case=sub_i,
                        n_cases=len(testing_subjects),
                        t_start=test_start
                    )
        else:
            if verbose > 0:
                print(
                    '\033[KTesting subject {:} ({:d}/{:d}) {:} ETA {:}'.format(
                        subject, sub_i + 1, len(testing_subjects),
                        time_to_string(test_elapsed),
                        time_to_string(test_eta),
                    ), end='\r'
                )
            if type_dict[type_tag] is not None:
                type_dict[type_tag](
                    config=config, net=net, subject=subject,
                    case=sub_i, n_cases=len(testing_subjects),
                    t_start=test_start
                )


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
    model_base = os.path.splitext(os.path.basename(options['config']))[0]
    models = importlib.import_module('models')
    network_class = getattr(models, config['network'])
    seeds = config['seeds']

    print(
        '{:}[{:}] {:}<Incremental learning framework>{:}'.format(
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
            '{:}[{:}] {:}Starting cross-validation (model: {:}){:}'
            ' (seed {:d}){:}'.format(
                c['clr'] + c['c'], strftime("%H:%M:%S"), c['g'], model_base,
                c['nc'] + c['y'], seed, c['nc']
            )
        )
        # Network init (random weights)
        np.random.seed(seed)
        torch.manual_seed(seed)
        net = network_class(
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

        print(
            '{:}Testing initial weights{:} - {:02d}/{:02d} '
            '({:} parameters)'.format(
                c['clr'] + c['c'], c['nc'], test_n + 1, len(config['seeds']),
                c['b'] + str(n_param) + c['nc']
            )
        )
        # We will save the initial results pre-training
        all_subjects = [p for t_list in subjects.values() for p in t_list]
        test(
            config, seed, net, 'init', all_subjects, verbose=1
        )

        # Cross-validation loop
        subjects_fold = {
            t_key: {
                'list': np.random.permutation([
                    sub for sub in t_list
                ]).tolist()
            }
            for t_key, t_list in subjects.items()
        }

        for i in range(n_folds):
            for t_key, t_dict in subjects_fold.items():
                t_list = t_dict['list']
                subjects_fold[t_key]['ini'] = len(t_list) * i // n_folds
                subjects_fold[t_key]['end'] = len(t_list) * (i + 1) // n_folds
            # Training
            # Here we'll do the training / validation / testing split...
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
            else:
                training_tasks = validation_tasks = training_validation

            # Testing set for the current fold
            testing_set = [
                p for t in subjects_fold.values()
                for p in t['list'][t['ini']:t['end']]
            ]

            # Baseline model (full continuum access)
            net = network_class(
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
                    c['clr'] + c['c'], c['g'] + str(i) + c['nc'],
                    test_n + 1, len(config['seeds']),
                    c['b'] + str(n_param) + c['nc']
                )
            )
            train(config, net, training_set, validation_set, model_name, 2)

            # Testing for the baseline. We want to reduce repeating the same
            # experiments to save time if the algorithm crashes.
            test(
                config, seed, net, 'baseline', all_subjects, verbose=1
            )


if __name__ == '__main__':
    main()
