import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sn
from math import gcd
from functools import reduce


def lcm(x, y):
    if x > 0 and y > 0:
        value = int(x * y / gcd(x, y))
    else:
        value = max(x, y, 1)
    return value


def plot_bands_sb(
        methods, legend, ymin=0, ymax=1,
        title='', xlabel='Time', ylabel='Metric', base_name='baseline',
        grouped=True
):
    # Init
    mean_tasks = [
        m if m.shape[1] > 1 else np.repeat(m, 2, axis=1) for m in methods
    ]
    tasks = [m.shape[1] for m in mean_tasks]
    steps = reduce(lcm, [t - 1 for t in tasks])
    x = [
        list(range(0, steps, int(steps / (t - 1)))) + [steps]
        if t > 1 else [0, steps] for t in tasks
    ]

    if grouped:
        data = [
            [x_i, val_i, m] for m, m_vals, m_x in zip(legend, mean_tasks, x)
            for seed_val_i in m_vals for val_i, x_i in zip(seed_val_i, m_x)
        ]
    else:
        data = [
                   [x_i, val_i, '{:}-{:02d}'.format(m, s_i)]
                   for m, m_vals, m_x in zip(legend, mean_tasks, x)
                   for s_i, (seed_val_i) in enumerate(m_vals)
                   for val_i, x_i in zip(seed_val_i, m_x)
                   if m is not base_name
               ] + [
                   [x_i, val_i, base_name]
                   for m, m_vals, m_x in zip(legend, mean_tasks, x)
                   for x_i, val_i in zip(m_x, np.max(m_vals, axis=0))
                    if m is base_name
               ]

    data_df = pd.DataFrame(data, columns=[xlabel, ylabel, 'Method'])

    for t, mean_ti in zip(legend, mean_tasks):
        print(
            '{:}  {:} [{:5.3f}, {:5.3f}] - mean {:5.3f}'.format(
                t, ylabel, np.min(mean_ti[:, 1:]), np.max(mean_ti[:, 1:]),
                np.max(np.mean(mean_ti[:, 1:], axis=0))
            )
        )

    plt.ylim(ymin, ymax)
    sn.lineplot(x=xlabel, y=ylabel, hue='Method', ci=100, data=data_df)
    plt.gca().set_xticks([])
    plt.gca().set_title(title)


def plot_metrics_diff(
    metrics_vec, metrics_dict, dataset, filename,
        base_name='baseline', rows=2, cols=3,
    grouped=True, ymin=0, ymax=1
):
    fig = plt.figure(figsize=(cols * 6, rows * 6))
    for cell, (metric, (_, metric_long)) in enumerate(metrics_dict.items()):
        values = [
            np.array([
                seed_vals[metric] if m is base_name else seed_vals[metric][1:]
                for seed_vals in m_vals.values()
            ])
            for m, m_vals in metrics_vec.items()
        ]

        plt.subplot(rows, cols, cell + 1)
        plot_bands_sb(
            values, metrics_vec.keys(),
            title='{:} ({:})'.format(metric_long, dataset),
            xlabel='Task (time)', ylabel=metric_long,
            base_name=base_name, grouped=grouped,
            ymin=ymin, ymax=ymax
        )

    plt.tight_layout()
    plt.savefig(filename)