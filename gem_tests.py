import argparse
import os
import numpy as np
import importlib
import yaml
import torch
from time import strftime
from continual import LoggingGEM
from utils import color_codes
from natural_comparison import train, load_datasets, parse_inputs


"""
> Network functions
"""


def process_net(
    config, net, model_name, seed, training_set, validation_set,
    task, offset1, offset2, epochs
):
    net.to(net.device)
    train(
        config, seed, net, training_set, validation_set,
        model_name, epochs, epochs, task, offset1, offset2, 2
    )
    net.reset_optimiser()
    net.to(torch.device('cpu'))


"""
> Main function
"""


def main():
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
    epochs = config['epochs']

    try:
        pretrained = config['pretrained']
    except KeyError:
        pretrained = False
    try:
        lr = config['lr']
    except KeyError:
        lr = 1e-3

    # GEM parameters
    try:
        gem_weight = config['gem_weight']
    except KeyError:
        gem_weight = 0.5
    try:
        gem_memories = config['gem_memories']
    except KeyError:
        gem_memories = 256


    print(
        '{:}[{:}] {:}<GEM logging experiments>{:}'.format(
            c['c'], strftime("%H:%M:%S"), c['y'], c['nc']
        )
    )

    # We want a common starting point
    d_tr, d_te = load_datasets(config)
    n_tasks = len(d_tr)
    n_classes = max(d_tr[-1][0][-1], d_te[-1][0][-1])
    nc_per_task = n_classes // n_tasks

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
        models = importlib.import_module('models')
        network = getattr(models, config['network'])
        np.random.seed(seed)
        torch.manual_seed(seed)

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
                for _, x, y in d_tr
            ]
            validation_tasks = [
                (x[:int(len(x) * val_split)], y[:int(len(x) * val_split)])
                for _, x, y in d_tr
            ]
        else:
            training_tasks = validation_tasks = [(x, y) for _, x, y in d_tr]

        # GEM approaches. We group all the GEM-related approaches here for
        # simplicity. All parameters should be shared for a fair comparison.
        net = LoggingGEM(
            network(
                n_outputs=n_classes, lr=lr, pretrained=pretrained
            ), best=False,
            n_memories=gem_memories, memory_strength=gem_weight,
            n_tasks=n_tasks, n_classes=n_classes
        )
        net.to(torch.device('cpu'))
        n_param = sum(
            p.numel() for p in net.model.parameters() if p.requires_grad
        )

        for t_i, (training_set, validation_set) in enumerate(
                zip(training_tasks, validation_tasks)
        ):

            offset1 = t_i * nc_per_task
            offset2 = (t_i + 1) * nc_per_task

            # < GEM >
            # Original GEM
            print(
                '{:}Starting task - GEM {:02d}/{:02d}{:} - {:02d}/{:02d} '
                '({:} parameters)'.format(
                    c['clr'] + c['c'], t_i + 1, n_tasks, c['nc'],
                    test_n + 1, len(config['seeds']),
                    c['b'] + str(n_param) + c['nc']
                )
            )
            # We train the gem model on the current task
            model_name = os.path.join(
                model_path,
                '{:}-log_gem-t{:02d}.s{:05d}.pt'.format(
                    model_base, t_i, seed
                )
            )
            process_net(
                config, net, model_name, seed, training_set, validation_set,
                t_i, offset1, offset2, epochs
            )


if __name__ == '__main__':
    torch.backends.cudnn.enabled = False
    main()
