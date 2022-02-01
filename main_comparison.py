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
            load_image_list(p_path, file_i, roi)
            for file_i in experiment_config['files']
        )
    else:
        images = load_image_list(
            p_path, [experiment_config['files']], roi
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
            load_image_list(p_path, file_i, roi)
            for file_i in experiment_config['files']
        )
    else:
        images = load_image_list(
            p_path, [experiment_config['files']], roi
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


def test_images_class(config, net, subject, session=None):
    roi, label, images = get_images_class(
        config, subject, session
    )
    if isinstance(images, tuple):
        data = tuple(
            data_i.astype(np.float32)
            for data_i in images
        )
    else:
        data = images.astype(np.float32)
    prediction = net.inference(data, nonbatched=True) > 0.5

    no_prediction = np.logical_not(prediction)

    label_csv = os.path.join(config['path'], config['labels'])
    dx_df = pd.read_csv(label_csv)
    label_dict = dx_df.set_index(dx_df.columns[0])[dx_df.columns[1]].to_dict()
    label = label_dict[subject]
    target = np.array(label).astype(bool)
    no_target = np.logical_not(target)

    tp = int(np.sum(np.logical_and(target, prediction)))
    tn = int(np.sum(np.logical_and(no_target, no_prediction)))
    fp = int(np.sum(np.logical_and(no_target, prediction)))
    fn = int(np.sum(np.logical_and(target, no_prediction)))

    results = {
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
    }
    return results


def test_images_patch(config, net, subject, session=None):
    roi, labels, images = get_images_seg(
        config, subject, session
    )
    if 'test_patch' in config and 'test_overlap' in config:
        val_dataset = config['validation'](
            [images], [labels], [roi], patch_size=config['test_patch'],
            overlap=config['test_overlap'], balanced=False
        )
    elif 'test_patch' in config:
        val_dataset = config['validation'](
            [images], [labels], [roi], patch_size=config['test_patch'],
            balanced=False
        )
    else:
        val_dataset = config['validation'](
            [images], [labels], [roi], balanced=False
        )

    test_loader = DataLoader(
        val_dataset, config['test_batch'], num_workers=32
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
    return results


def test_images_seg(config, mask_name, net, subject, session=None):
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
            prediction = net.inference(data) > 0.5
        except RuntimeError:
            patch_size = config['test_patch']
            batch_size = config['test_batch']
            prediction = net.patch_inference(
                data, patch_size, batch_size
            ) > 0.5
        segmentation[bb] = prediction
        segmentation[np.logical_not(roi)] = 0

        ref_nii = nibabel.load(find_file(config['labels'], d_path))
        segmentation_nii = nibabel.Nifti1Image(
            segmentation, ref_nii.get_qform(), ref_nii.header
        )
        segmentation_nii.to_filename(prediction_file)
    else:
        roi = get_mask(find_file(config['roi'], d_path))
        label = get_mask(find_file(config['labels'], d_path))
        bb = get_bb(roi, 2)
        segmentation = nibabel.load(prediction_file).get_fdata()
        prediction = segmentation[bb].astype(bool)

    try:
        min_size = config['min_size']
        prediction = remove_small_regions(prediction, min_size)
    except KeyError:
        pass

    target = label[bb].astype(bool)
    no_target = np.logical_not(target)
    target_regions, gtr = bwlabeln(target, return_num=True)
    no_prediction = np.logical_not(prediction)
    prediction_regions, r = bwlabeln(prediction, return_num=True)
    true_positive = np.logical_and(target, prediction)
    no_false_positives = np.unique(prediction_regions[true_positive])
    false_positive_regions = np.logical_not(
        np.isin(prediction_regions, no_false_positives.tolist() + [0])
    )
    false_positive = np.logical_and(no_target, prediction)

    results = {
        'TPV': int(np.sum(true_positive)),
        'TNV': int(np.sum(np.logical_and(no_target, no_prediction))),
        'FPV': int(np.sum(false_positive)),
        'FNV': int(np.sum(np.logical_and(target, no_prediction))),
        'TPR': len(np.unique(target_regions[true_positive])),
        'FPR': len(np.unique(prediction_regions[false_positive_regions])),
        'GTR': gtr,
        'R': r
    }
    return results


def test(
    config, seed, net, base_name, testing_results, testing_subjects,
    verbose=0
):
    # Init
    options = parse_inputs()
    mask_base = os.path.splitext(os.path.basename(options['config']))[0]
    mask_name = '{:}-{:}.s{:05d}.nii.gz'.format(
        mask_base, base_name, seed
    )
    type_dict = {
        'class': test_images_class,
        'seg': partial(test_images_seg, mask_name=mask_name),
        'patch': test_images_patch
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
                results = type_dict[type_tag](
                    config=config, net=net, subject=subject, session=session
                )
                for r_key, r_value in results.items():
                    testing_results[subject][session][str(seed)][r_key].append(
                        r_value
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
            results = type_dict[type_tag](
                config=config, net=net, subject=subject
            )
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
    config, seed, json_name, base_name, net, results, subjects
):
    path = config['json_path']
    json_file = find_file(json_name, path)
    if json_file is None:
        json_file = os.path.join(path, json_name)
        test(
            config, seed, net, base_name, results,
            subjects, verbose=1
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
    subjects = get_subjects(config)

    # We prepare the dictionaries that will hold the relevant segmentation and
    # detection measures. That includes all positive combinations of positives
    # and negatives. Most relevant metrics like DSC come from there.
    baseline_testing = empty_test_results(config, subjects)
    naive_testing = empty_test_results(config, subjects)
    ewc_testing = empty_test_results(config, subjects)
    ewcplus_testing = empty_test_results(config, subjects)
    init_testing = empty_test_results(config, subjects)

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
    ewcplus_training = deepcopy(baseline_training)

    if isinstance(config['files'], tuple):
        n_images = len(config['files'][0])
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

        print(
            '{:}Testing initial weights{:} - {:02d}/{:02d} '
            '({:} parameters)'.format(
                c['clr'] + c['c'], c['nc'], test_n + 1, len(config['seeds']),
                c['b'] + str(n_param) + c['nc']
            )
        )
        # We will save the initial results pre-training
        all_subjects = [p for t_list in subjects.values() for p in t_list]
        json_name = '{:}-init_testing.s{:d}.json'.format(
            model_base, seed
        )
        init_testing = get_test_results(
            config, seed, json_name, 'init', net,
            init_testing, all_subjects
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
                fold_val_baseline = empty_task_results(config, validation_tasks)
                fold_val_naive = empty_task_results(config, validation_tasks)
                fold_val_ewc = empty_task_results(config, validation_tasks)
                fold_val_ewcplus = empty_task_results(config, validation_tasks)
            else:
                training_tasks = validation_tasks = training_validation
                fold_val_baseline = None
                fold_val_naive = None
                fold_val_ewc = None
                fold_val_ewcplus = None
            fold_tr_baseline = empty_task_results(config, training_tasks)
            fold_tr_naive = empty_task_results(config, training_tasks)
            fold_tr_ewc = empty_task_results(config, training_tasks)
            fold_tr_ewcplus = empty_task_results(config, training_tasks)

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
            json_name = '{:}-baseline-init_training.s{:d}.json'.format(
                model_base, seed
            )
            fold_tr_baseline = get_task_results(
                config, json_name, 'baseline-train.init', net,
                fold_tr_baseline
            )
            json_name = '{:}-naive-init_training.s{:d}.json'.format(
                model_base, seed
            )
            fold_tr_naive = get_task_results(
                config, json_name, 'naive-train.init', net, fold_tr_naive
            )
            json_name = '{:}-ewc-init_training.s{:d}.json'.format(
                model_base, seed
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

            ewc_net = models.MetaModel(
                config['network'](
                    conv_filters=config['filters'],
                    n_images=n_images
                ),
                ewc_weight, ewc_binary
            )
            ewc_net.model.load_model(starting_model)

            # EWC approach. We use a penalty term / regularization loss
            # to ensure previous data isn't forgotten.
            try:
                ewc_alpha = config['ewc_alpha']
            except KeyError:
                ewc_alpha = 0.9

            ewcplus_net = models.MetaModel(
                config['network'](
                    conv_filters=config['filters'],
                    n_images=n_images
                ),
                ewc_weight, ewc_binary, ewc_alpha
            )
            ewcplus_net.model.load_model(starting_model)

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
                model_name = os.path.join(
                    model_path,
                    '{:}-ewcpp-t{:02d}.n{:d}.s{:05d}.pt'.format(
                        model_base, ti, i, seed
                    )
                )
                train(
                    config, ewcplus_net, training_set, validation_set,
                    model_name, 2
                )
                ewcplus_net.reset_optimiser()

                # Then we test it against all the datasets and tasks
                json_name = '{:}-ewcpp_test.f{:d}.s{:d}.t{:02d}.json'.format(
                    model_base, i, seed, ti
                )
                ewcplus_testing = get_test_results(
                    config, seed, json_name, 'ewcpp-test.t{:02d}'.format(ti),
                    ewcplus_net, ewcplus_testing, testing_set
                )

                json_name = '{:}-ewcpp-training.' \
                            'f{:d}.s{:d}.t{:02d}.json'.format(
                                model_base, i, seed, ti
                            )
                fold_tr_ewcplus = get_task_results(
                    config, json_name, 'ewcpp-train.f{:d}.t{:02d}'.format(
                        i, ti
                    ), ewcplus_net, fold_tr_ewcplus
                )
                if fold_val_naive is not None:
                    json_name = '{:}-ewcpp-validation.' \
                                'f{:d}.s{:d}.t{:02d}.json'.format(
                                    model_base, i, seed, ti
                                )
                    fold_val_ewcplus = get_task_results(
                        config, json_name, 'ewcpp-val.f{:d}.t{:02d}'.format(
                            i, ti
                        ),
                        ewcplus_net, fold_val_ewcplus
                    )

            # Now it's time to push the results
            baseline_training[str(seed)]['training'].append(fold_tr_baseline)
            naive_training[str(seed)]['training'].append(fold_tr_naive)
            ewc_training[str(seed)]['training'].append(fold_tr_ewc)
            ewcplus_training[str(seed)]['training'].append(fold_tr_ewcplus)
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
                ewcplus_training[str(seed)]['validation'].append(
                    fold_val_ewcplus
                )

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
    save_results(
        config, '{:}-ewcpp_testing.json'.format(model_base),
        ewcplus_testing
    )
    save_results(
        config, '{:}-ewcpp_training.json'.format(model_base),
        ewcplus_training
    )


if __name__ == '__main__':
    main()
