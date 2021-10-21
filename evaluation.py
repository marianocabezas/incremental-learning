import seaborn as sn
import os
import re
import nibabel as nib
import numpy as np
from shutil import copyfile
import matplotlib.pyplot as plt


def plot_bands(
    x, y, yinf, ysup, ax, xmin=None, xmax=None, ymin=0, ymax=1,
    title='', xlabel='Epoch', ylabel='Metric', legend=None
):
    # Init
    if xmin is None:
        xmin = np.min(x)
    if xmax is None:
        xmax = np.max(x)
    if ymin is None:
        ymin = np.min(yinf)
    if ymax is None:
        ymax = np.max(ysup)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    colomap = ['b', 'g', 'c', 'r', 'm', 'y', 'k']

    if yinf is not None and ysup is not None:
        for yi, yinfi, ysupi, ci in zip(y, yinf, ysup, colomap):
            ax.plot(x, yi, '-', color=ci)
            ax.fill_between(x, yinfi, ysupi, alpha=0.2, color=ci)
    else:
        for yi, ci in zip(y, colomap):
            ax.plot(x, yi, '-', color=ci, linewidth=2.0)

    ax.set_xlim(xmin=xmin, xmax=xmax)
    ax.set_ylim(ymin=ymin, ymax=ymax)

    if legend is not None:
        ax.legend(legend)


def plot_metrics(baseline, naive, dataset, filename, n_tasks=10, plot=False, seeds=None):
    if seeds is None:
        seeds = ['42', '80702', '74794', '62021', '48497']
    dsc_bl = [
        [
            [
                2 * tpv / (2 * tpv + fnv + fpv)
                for tpv, fnv, fpv in zip(
                sub_data[seed]['TPV'],
                sub_data[seed]['FNV'],
                sub_data[seed]['FPV']
            )
            ] * n_tasks
            for sub_data in baseline.values()
        ]
        for seed in seeds
    ]
    vtpf_bl = [
        [
            [
                tpv / (tpv + fnv)
                for tpv, fnv in zip(
                sub_data[seed]['TPV'],
                sub_data[seed]['FNV']
            )
            ] * n_tasks
            for sub_data in baseline.values()
        ]
        for seed in seeds
    ]
    vfpf_bl = [
        [
            [
                fpv / (tpv + fpv) if (tpv + fpv) > 0 else 0
                for fpv, tpv in zip(
                sub_data[seed]['FPV'],
                sub_data[seed]['TPV']
            )
            ] * n_tasks
            for sub_data in baseline.values()
        ]
        for seed in seeds
    ]
    dtpf_bl = [
        [
            [
                tpr / gtr
                for tpr, gtr in zip(
                sub_data[seed]['TPR'],
                sub_data[seed]['GTR']
            )
            ] * n_tasks
            for sub_data in baseline.values()
        ]
        for seed in seeds
    ]
    dfpf_bl = [
        [
            [
                fpr / r if r > 0 else 0
                for fpr, r in zip(
                sub_data[seed]['FPR'],
                sub_data[seed]['R']
            )
            ] * n_tasks
            for sub_data in baseline.values()
        ]
        for seed in seeds
    ]
    f1_bl = [
        [
            [
                2 * tpr / (r + gtr)
                for tpr, gtr, r in zip(
                sub_data[seed]['TPR'],
                sub_data[seed]['GTR'],
                sub_data[seed]['R']
            )
            ] * n_tasks
            for sub_data in baseline.values()
        ]
        for seed in seeds
    ]

    dsc_naive = [
        [
            [
                2 * tpv / (2 * tpv + fnv + fpv)
                for tpv, fnv, fpv in zip(
                sub_data[seed]['TPV'],
                sub_data[seed]['FNV'],
                sub_data[seed]['FPV']
            )
            ]
            for sub_data in naive.values()
        ]
        for seed in seeds
    ]
    vtpf_naive = [
        [
            [
                tpv / (tpv + fnv)
                for tpv, fnv in zip(
                sub_data[seed]['TPV'],
                sub_data[seed]['FNV']
            )
            ]
            for sub_data in naive.values()
        ]
        for seed in seeds
    ]
    vfpf_naive = [
        [
            [
                fpv / (tpv + fpv) if (tpv + fpv) > 0 else 0
                for fpv, tpv in zip(
                sub_data[seed]['FPV'],
                sub_data[seed]['TPV']
            )
            ]
            for sub_data in naive.values()
        ]
        for seed in seeds
    ]
    dtpf_naive = [
        [
            [
                tpr / gtr
                for tpr, gtr in zip(
                sub_data[seed]['TPR'],
                sub_data[seed]['GTR']
            )
            ]
            for sub_data in naive.values()
        ]
        for seed in seeds
    ]
    dfpf_naive = [
        [
            [
                fpr / r if r > 0 else 0
                for fpr, r in zip(
                sub_data[seed]['FPR'],
                sub_data[seed]['R']
            )
            ]
            for sub_data in naive.values()
        ]
        for seed in seeds
    ]
    f1_naive = [
        [
            [
                2 * tpr / (r + gtr)
                for tpr, gtr, r in zip(
                sub_data[seed]['TPR'],
                sub_data[seed]['GTR'],
                sub_data[seed]['R']
            )
            ]
            for sub_data in naive.values()
        ]
        for seed in seeds
    ]

    fig = plt.figure(figsize=(18, 12))
    ax = plt.subplot(2, 3, 1)
    mean_tasks_bl = np.mean(dsc_bl, axis=1)
    mean_tasks_naive = np.mean(dsc_naive, axis=1)
    x = list(range(n_tasks))
    y = np.stack([
        np.mean(mean_tasks_bl, axis=0),
        np.mean(mean_tasks_naive, axis=0)
    ], axis=0)
    yinf = np.stack([
        np.min(mean_tasks_bl, axis=0),
        np.min(mean_tasks_naive, axis=0)
    ], axis=0)
    ysup = np.stack([
        np.max(mean_tasks_bl, axis=0),
        np.max(mean_tasks_naive, axis=0)
    ], axis=0)
    plot_bands(
        x, y, yinf, ysup, ax, title='DSC ({:})'.format(dataset),
        xlabel='Task', ylabel='DSC',
        legend=['Baseline', 'Naive']
    )

    ax = plt.subplot(2, 3, 2)
    mean_tasks_bl = np.mean(vtpf_bl, axis=1)
    mean_tasks_naive = np.mean(vtpf_naive, axis=1)
    x = list(range(n_tasks))
    y = np.stack([
        np.mean(mean_tasks_bl, axis=0),
        np.mean(mean_tasks_naive, axis=0)
    ], axis=0)
    yinf = np.stack([
        np.min(mean_tasks_bl, axis=0),
        np.min(mean_tasks_naive, axis=0)
    ], axis=0)
    ysup = np.stack([
        np.max(mean_tasks_bl, axis=0),
        np.max(mean_tasks_naive, axis=0)
    ], axis=0)
    plot_bands(
        x, y, yinf, ysup, ax, title='TPF segmentation ({:})'.format(dataset),
        xlabel='Task', ylabel='TPF%',
        legend=['Baseline', 'Naive']
    )

    ax = plt.subplot(2, 3, 3)
    mean_tasks_bl = np.mean(vfpf_bl, axis=1)
    mean_tasks_naive = np.mean(vfpf_naive, axis=1)
    x = list(range(n_tasks))
    y = np.stack([
        np.mean(mean_tasks_bl, axis=0),
        np.mean(mean_tasks_naive, axis=0)
    ], axis=0)
    yinf = np.stack([
        np.min(mean_tasks_bl, axis=0),
        np.min(mean_tasks_naive, axis=0)
    ], axis=0)
    ysup = np.stack([
        np.max(mean_tasks_bl, axis=0),
        np.max(mean_tasks_naive, axis=0)
    ], axis=0)
    plot_bands(
        x, y, yinf, ysup, ax, title='FPF segmentation ({:})'.format(dataset),
        xlabel='Task', ylabel='FPF%',
        legend=['Baseline', 'Naive']
    )

    ax = plt.subplot(2, 3, 4)
    mean_tasks_bl = np.mean(f1_bl, axis=1)
    mean_tasks_naive = np.mean(f1_naive, axis=1)
    x = list(range(n_tasks))
    y = np.stack([
        np.mean(mean_tasks_bl, axis=0),
        np.mean(mean_tasks_naive, axis=0)
    ], axis=0)
    yinf = np.stack([
        np.min(mean_tasks_bl, axis=0),
        np.min(mean_tasks_naive, axis=0)
    ], axis=0)
    ysup = np.stack([
        np.max(mean_tasks_bl, axis=0),
        np.max(mean_tasks_naive, axis=0)
    ], axis=0)
    plot_bands(
        x, y, yinf, ysup, ax, title='F1 ({:})'.format(dataset), xlabel='Task', ylabel='F1',
        legend=['Baseline', 'Naive']
    )

    ax = plt.subplot(2, 3, 5)
    mean_tasks_bl = np.mean(dtpf_bl, axis=1)
    mean_tasks_naive = np.mean(dtpf_naive, axis=1)
    x = list(range(n_tasks))
    y = np.stack([
        np.mean(mean_tasks_bl, axis=0),
        np.mean(mean_tasks_naive, axis=0)
    ], axis=0)
    yinf = np.stack([
        np.min(mean_tasks_bl, axis=0),
        np.min(mean_tasks_naive, axis=0)
    ], axis=0)
    ysup = np.stack([
        np.max(mean_tasks_bl, axis=0),
        np.max(mean_tasks_naive, axis=0)
    ], axis=0)
    plot_bands(
        x, y, yinf, ysup, ax, title='TPF detection ({:})'.format(dataset), xlabel='Task', ylabel='TPF%',
        legend=['Baseline', 'Naive']
    )

    ax = plt.subplot(2, 3, 6)
    mean_tasks_bl = np.mean(dfpf_bl, axis=1)
    mean_tasks_naive = np.mean(dfpf_naive, axis=1)
    x = list(range(n_tasks))
    y = np.stack([
        np.mean(mean_tasks_bl, axis=0),
        np.mean(mean_tasks_naive, axis=0)
    ], axis=0)
    yinf = np.stack([
        np.min(mean_tasks_bl, axis=0),
        np.min(mean_tasks_naive, axis=0)
    ], axis=0)
    ysup = np.stack([
        np.max(mean_tasks_bl, axis=0),
        np.max(mean_tasks_naive, axis=0)
    ], axis=0)
    plot_bands(
        x, y, yinf, ysup, ax, title='FPF detection ({:})'.format(dataset), xlabel='Task', ylabel='FPF%',
        legend=['Baseline', 'Naive']
    )

    plt.tight_layout()
    plt.savefig(filename)

def plot_train_metrics(baseline, naive, dataset, filename, seed, fold, n_tasks=10, plot=False):
    dsc_bl = np.swapaxes([
        np.mean([
            [
                2 * tpv / (2 * tpv + fnv + fpv)
                for tpv, fnv, fpv in zip(
                sub_data['TPV'][1:],
                sub_data['FNV'][1:],
                sub_data['FPV'][1:]
            )
            ]
            for sub_data in task.values()
        ], axis=0)
        for task in baseline
    ], 0, 1)
    dtpf_bl = np.swapaxes([
        np.mean([
            [
                tpr / gtr
                for tpr, gtr in zip(
                sub_data['TPR'][1:],
                sub_data['GTR'][1:]
            )
            ]
            for sub_data in task.values()
        ], axis=0)
        for task in baseline
    ], 0, 1)
    f1_bl = np.swapaxes([
        np.mean([
            [
                2 * tpr / (r + gtr)
                for tpr, gtr, r in zip(
                sub_data['TPR'][1:],
                sub_data['GTR'][1:],
                sub_data['R'][1:]
            )
            ]
            for sub_data in task.values()
        ], axis=0)
        for task in baseline
    ], 0, 1)

    dsc_naive = np.swapaxes([
        np.mean([
            [
                2 * tpv / (2 * tpv + fnv + fpv)
                for tpv, fnv, fpv in zip(
                sub_data['TPV'][1:],
                sub_data['FNV'][1:],
                sub_data['FPV'][1:]
            )
            ]
            for sub_data in task.values()
        ], axis=0)
        for task in naive
    ], 0, 1)
    dtpf_naive = np.swapaxes([
        np.mean([
            [
                tpr / gtr
                for tpr, gtr in zip(
                sub_data['TPR'][1:],
                sub_data['GTR'][1:]
            )
            ]
            for sub_data in task.values()
        ], axis=0)
        for task in naive
    ], 0, 1)
    f1_naive = np.swapaxes([
        np.mean([
            [
                2 * tpr / (r + gtr)
                for tpr, gtr, r in zip(
                sub_data['TPR'][1:],
                sub_data['GTR'][1:],
                sub_data['R'][1:]
            )
            ]
            for sub_data in task.values()
        ], axis=0)
        for task in naive
    ], 0, 1)

    if plot:
        fig = plt.figure(figsize=(18, 6))
        ax = plt.subplot(1, 3, 1)
        sn.heatmap(
            np.concatenate([dsc_bl, dsc_naive]), cmap='jet', vmin=0,
            xticklabels=['Task{:02d}'.format(si) for si in range(len(naive))],
            yticklabels=['Baseline'] + ['Step{:02d}'.format(si) for si in range(len(naive))]
        )
        plt.title('DSC (segmentation) - Seed {:} - Fold {:02d}'.format(seed, fold))

        ax = plt.subplot(1, 3, 2)
        sn.heatmap(
            np.concatenate([dtpf_bl, dtpf_naive]), cmap='jet', vmin=0,
            xticklabels=['Task{:02d}'.format(si) for si in range(len(naive))],
            yticklabels=['Baseline'] + ['Step{:02d}'.format(si) for si in range(len(naive))]
        )
        plt.title('TPF% (detection) - Seed {:} - Fold {:02d}'.format(seed, fold))

        ax = plt.subplot(1, 3, 3)
        sn.heatmap(
            np.concatenate([f1_bl, f1_naive]), cmap='jet', vmin=0,
            xticklabels=['Task{:02d}'.format(si) for si in range(len(naive))],
            yticklabels=['Baseline'] + ['Step{:02d}'.format(si) for si in range(len(naive))]
        )
        plt.title('F1 score (detection) - Seed {:} - Fold {:02d}'.format(seed, fold))

        plt.tight_layout()
        plt.savefig(filename)

    return dsc_bl, dsc_naive, dtpf_bl, dtpf_naive, f1_bl, f1_naive
